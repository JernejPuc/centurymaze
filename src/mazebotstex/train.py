"""Tasks auxiliary to RL"""

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import log_softmax, one_hot

from discit.data import TensorDict, ExperienceBuffer
from discit.distr import Categorical, FixedVarNormal
from discit.marl import AuxTask

import config as cfg


# ------------------------------------------------------------------------------
# MARK: BeliefAuxTask

class BeliefAuxTask(AuxTask):
    """DIABL & InfER"""

    STAT_KEYS = ('Aux/loc_nll_off',)

    def __init__(
        self,
        policy: Module,
        optimizer,
        rng,
        n_envs: int,
        env_bot_idcs: Tensor,
        buffer_size: int,
        n_truncated_steps: int,
        online_only: bool,
        detach_com: bool = False
    ):
        super().__init__(True, not online_only)

        self.policy = policy
        self.optimizer = optimizer
        self.rng = rng
        self.detach_com = detach_com

        self.n_envs = n_envs
        self.env_bot_idcs = env_bot_idcs
        self.n_truncated_steps = n_truncated_steps

        self.buffer = ExperienceBuffer(buffer_size)

        # At most, there is a temp. buffer for each env. slice and goal idx. (events can overlap)
        self.temp_buffers = [[] for _ in range(n_envs)]

    def clear(self):
        self.buffer.clear(self.n_truncated_steps * self.n_envs * cfg.N_GOAL_CLRS)

    # --------------------------------------------------------------------------
    # MARK: collect

    def collect(self, data: TensorDict, obs: 'tuple[Tensor, ...]', mem: 'tuple[Tensor, ...]'):

        # Discard all unifinished seqs. on reset
        for i, nrst in enumerate(data['nrst'][self.env_bot_idcs].tolist()):
            if not nrst:
                self.temp_buffers[i].clear()

        for i, prio in enumerate(data['prio'].tolist()):
            buffers = self.temp_buffers[i]

            # Start adding to new temp. buffer on prio. signal
            if prio:
                buffers.append(ExperienceBuffer(self.n_truncated_steps))

            if buffers:
                chunk = data.chunk(i, 1)
                j = 0

                while j != len(buffers):
                    buffer = buffers[j]
                    buffer.append(chunk)

                    # When full length is reached, the seq. is added to the actual buffer
                    if buffer.is_full():
                        self.buffer.extend(buffer)

                        # Drop temp. buffer
                        buffer.clear()
                        del buffers[j]

                    else:
                        j += 1

    # --------------------------------------------------------------------------
    # MARK: update

    def update(self, batches: 'list[TensorDict]', stats: 'dict[str, Tensor]'):
        try:
            seq = self.buffer.sample(self.rng, self.n_truncated_steps, 1, 1, True)

        except RuntimeError:
            return

        qkv, memp, _ = seq[0]['mem']

        loss = 0.
        self.optimizer.zero_grad()

        for batch in seq:
            xp = batch['obs'][0]

            # Force open com. channel for bots with any obj. in frame (who should be speaking)
            forced_com_mask = batch['vaux'][:, 1:cfg.AUX_VAL_SPLIT[0]].any(1)

            _, bg, qkv, memp, _ = self.policy(xp, qkv, memp, forced_com_mask)

            if self.detach_com:
                qkv = qkv.detach()

            bg = FixedVarNormal(bg, skip_log_shift=True)

            loss = loss + self.offline_loss(batch, bg, stats)

        loss = loss * (cfg.AUX_WEIGHT / self.n_truncated_steps)
        loss.backward()
        self.optimizer.step()

    # --------------------------------------------------------------------------
    # MARK: offline_loss

    def offline_loss(
        self,
        batch: TensorDict,
        aux: FixedVarNormal,
        stats: 'dict[str, Tensor]'
    ) -> Tensor:

        # Extract goal coords. and masks
        goal_found_mask = batch['obs'][1][:, cfg.GOAL_FOUND_IDX]
        task_undone_mask = batch['obs'][1][:, cfg.GOAL_SPEC_SLICE].any(-1)
        goal_pos = batch['vaux'][:, -cfg.AUX_VAL_SPLIT[1]:]

        # Compare belief to target coords.
        aux_log_prob = aux.log_prob(goal_pos)

        # Only propagate error for bots with goal found and not reached
        prop_err_mask = goal_found_mask * task_undone_mask

        aux_loss = (aux_log_prob * prop_err_mask).mean()

        # Stats for logging
        with torch.no_grad():
            stats['Aux/loc_nll_off'] -= aux_loss

        return -aux_loss

    # --------------------------------------------------------------------------
    # MARK: loss

    def loss(
        self,
        batch: TensorDict,
        act: Categorical,
        vals: None,
        auxs: 'tuple[FixedVarNormal]',
        stats: 'dict[str, Tensor]'
    ) -> Tensor:

        # Extract goal coords. and masks
        goal_found_mask = batch['obs'][1][:, cfg.GOAL_FOUND_IDX]
        task_undone_mask = batch['obs'][1][:, cfg.GOAL_SPEC_SLICE].any(-1)
        goal_pos = batch['vaux'][:, -cfg.AUX_VAL_SPLIT[1]:]

        # Compare beliefs to targets
        goal_log_prob = auxs[0].log_prob(goal_pos)

        if self.detach_com:
            goal_log_prob = goal_log_prob.detach()

        # Only propagate error for bots with goal found and not reached
        prop_err_mask = goal_found_mask * task_undone_mask

        goal_loss = (goal_log_prob * prop_err_mask).mean()

        return -goal_loss


# ------------------------------------------------------------------------------
# MARK: VisionAuxTask

class VisionAuxTask(AuxTask):
    """Image encoder pretraining & retraining"""

    STAT_KEYS = (
        'VisNet/loss',
        'VisNet/dep_mse',
        'VisNet/clr_fce',
        'VisNet/ent_fce',
        'VisNet/obj_nll',
        'VisNet/enc_mae')

    def __init__(self, visnet: Module, optimizer, refnet: Module = None):
        super().__init__(False, True)

        self.visnet = visnet
        self.optimizer = optimizer
        self.refnet = refnet

    def clear(self):
        pass

    # --------------------------------------------------------------------------
    # MARK: collect

    def get_clr_seg(self, ent_seg: Tensor) -> Tensor:

        # Get colours from extended segmentation tags
        clr_seg = torch.where(
            ent_seg >= cfg.OBJ_CLS_OFFSET,
            ent_seg - cfg.OBJ_CLS_OFFSET,
            cfg.CLR_CLS_UNSPEC)

        # Revert to original segmentation tags
        ent_seg[ent_seg >= cfg.OBJ_CLS_OFFSET] = cfg.ENT_CLS_OBJ

        return clr_seg, ent_seg

    def get_px_weights(self, ent_seg: Tensor) -> Tensor:
        weights = torch.zeros(ent_seg.shape, device=ent_seg.device)

        for i in range(cfg.N_ENT_CLASSES):
            weights[ent_seg == i] = cfg.PX_WEIGHT_MAP[i]

        return weights.unsqueeze(1)

    def collect(self, data: TensorDict, obs: 'tuple[Tensor, ...]', mem: 'tuple[Tensor, ...]'):
        obs_img, obs_vec, _ = data['obs']

        ipt = obs_img[:, :4]
        dep = obs_img[:, 4:5]
        ent_seg = obs_img[:, 5].long()

        obj_in_frame = data['vaux'].split(cfg.AUX_VAL_SPLIT, dim=-1)[0].argmax(-1, keepdim=True)
        bot_pos = obs_vec[:, cfg.BOT_POS_SLICE]

        clr_seg, ent_seg = self.get_clr_seg(ent_seg)

        data['obs'] = (ipt, dep, ent_seg, clr_seg, self.get_px_weights(ent_seg), obj_in_frame, bot_pos)

    # --------------------------------------------------------------------------
    # MARK: update

    def update(self, batches: 'list[TensorDict]', stats: 'dict[str, Tensor]'):
        for b in batches:
            self.optimizer.zero_grad()

            out = self.visnet(b['obs'][0])
            refenc = self.refnet.enc(b['obs'][0].chunk(2)[1]).flatten() if self.refnet is not None else None

            loss = self.loss(b, None, None, out, stats, refenc)
            loss.backward()

            self.optimizer.step()

    # --------------------------------------------------------------------------
    # MARK: loss

    def loss(
        self,
        batch: TensorDict,
        act: None,
        vals: None,
        auxs: 'tuple[Tensor, ...]',
        stats: 'dict[str, Tensor]',
        refenc: 'Tensor | None' = None
    ) -> Tensor:

        # Unpack targets
        _, dep_target, ent_target, clr_target, weights, obj_in_frame, _ = batch['obs']

        # Unpack model outputs
        img_out, enc, pred_obj, _ = auxs
        clr_logits, ent_logits, dep = img_out.split(cfg.DEC_IMG_CHANNEL_SPLIT, dim=1)

        # Logits to probs.
        clr_log_probs = log_softmax(clr_logits, dim=1)
        clr_probs = clr_logits.softmax(dim=1)

        ent_log_probs = log_softmax(ent_logits, dim=1)
        ent_probs = ent_logits.softmax(dim=1)

        # Expand indices to probs. (one-hot)
        clr_probs_target = one_hot(clr_target, cfg.N_CLR_CLASSES)
        clr_probs_target = torch.movedim(clr_probs_target, -1, 1)

        ent_probs_target = one_hot(ent_target, cfg.N_ENT_CLASSES)
        ent_probs_target = torch.movedim(ent_probs_target, -1, 1)

        # Using weighted sum / weight_sum instead of mean, weighting across the whole batch
        weight_sum = weights.sum()

        # Segmentation: focal cross entropy (gamma 2)
        clr_fce = ((1. - clr_probs).square() * (clr_log_probs * clr_probs_target) * weights).sum() / weight_sum

        # Segmentation: focal cross entropy (gamma 2)
        ent_fce = ((1. - ent_probs).square() * (ent_log_probs * ent_probs_target) * weights).sum() / weight_sum

        # Reconstruction: mean squared error
        dep_mse = ((dep - dep_target).square() * weights.squeeze(1)).sum() / weight_sum

        # Prediction: neg. log. likelihood
        obj_nll = Categorical.from_raw(None, pred_obj).log_prob(None, obj_in_frame).mean()

        # Full loss
        loss = 0.05 * dep_mse - clr_fce - ent_fce - obj_nll

        # Consistency with pretrained encoder on the original data: mean absolute error
        if refenc is not None:
            enc = enc.chunk(2)[1].flatten()
            enc_mae = (enc - refenc).abs().mean()

            loss = loss + enc_mae

        else:
            enc_mae = 0.

        # Stats for logging
        with torch.no_grad():
            stats['VisNet/loss'] += loss.detach()
            stats['VisNet/dep_mse'] += dep_mse.detach()
            stats['VisNet/clr_fce'] -= clr_fce.detach()
            stats['VisNet/ent_fce'] -= ent_fce.detach()
            stats['VisNet/obj_nll'] -= obj_nll.detach()

            if not isinstance(enc_mae, float):
                stats['VisNet/enc_mae'] += enc_mae.detach()

        return loss
