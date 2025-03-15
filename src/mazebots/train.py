"""Tasks auxiliary to RL"""

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import log_softmax, one_hot

from discit.data import TensorDict, ExperienceBuffer
from discit.distr import Categorical, FixedVarNormal
from discit.marl import AuxTask

import config as cfg
from utils_torch import rgb_to_hsv


# ------------------------------------------------------------------------------
# MARK: BeliefAuxTask

class BeliefAuxTask(AuxTask):
    STAT_KEYS = ('Aux/loc_nll_off',)

    def __init__(
        self,
        policy: Module,
        optimizer,
        rng,
        n_envs: int,
        n_bots: int,
        batch_size: int,
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
        self.n_bots = n_bots
        self.n_truncated_steps = n_truncated_steps
        self.n_envs_per_batch, leftover_samples = divmod(batch_size, n_bots)

        if leftover_samples:
            raise ValueError(f'Batch size ({batch_size}) incompatible with num. of actors ({n_bots}).')

        self.buffer = ExperienceBuffer(buffer_size)

        # At most, there is a temp. buffer for each env. slice and goal idx. (events can overlap)
        self.temp_buffers = [[] for _ in range(n_envs)]

    def clear(self):
        self.buffer.clear(self.n_truncated_steps * self.n_envs * cfg.N_GOAL_CLRS)

    # --------------------------------------------------------------------------
    # MARK: collect

    def collect(self, data: TensorDict, obs: 'tuple[Tensor, ...]', mem: 'tuple[Tensor, ...]'):

        # Discard all unifinished seqs. on reset
        for i, nrst in enumerate(data['nrst'][::self.n_bots].tolist()):
            if not nrst:
                self.temp_buffers[i].clear()

        for i, prio in enumerate(data['prio'].tolist()):
            buffers = self.temp_buffers[i]

            # Start adding to new temp. buffer on prio. signal
            if prio:
                buffers.append(ExperienceBuffer(self.n_truncated_steps))

            if buffers:
                chunk = data.chunk(i, self.n_envs)
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
            seq = self.buffer.sample(self.rng, self.n_truncated_steps, 1, self.n_envs_per_batch, True)

        except RuntimeError:
            return

        qkv, memp, _ = seq[0]['mem']

        loss = 0.
        self.optimizer.zero_grad()

        for batch in seq:
            xp, _, x_map = batch['obs']
            n_envs = len(x_map)

            # Force open com. channel for bots with any obj. in frame (who should be speaking)
            forced_com_mask = batch['vaux'][:, 1:cfg.AUX_VAL_SPLIT[0]].any(1)

            _, bg, qkv, memp, _ = self.policy(xp, qkv, memp, forced_com_mask, n_envs)

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
        task_done_mask = batch['obs'][1][:, cfg.TASK_DONE_IDX]
        goal_pos = batch['vaux'][:, -cfg.AUX_VAL_SPLIT[1]:]

        # Compare belief to target coords.
        aux_log_prob = aux.log_prob(goal_pos)

        # Only propagate error for bots with goal found and not reached
        prop_err_mask = goal_found_mask * (1. - task_done_mask)

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
        task_done_mask = batch['obs'][1][:, cfg.TASK_DONE_IDX]
        goal_pos = batch['vaux'][:, -cfg.AUX_VAL_SPLIT[1]:]

        # Compare beliefs to targets
        goal_log_prob = auxs[0].log_prob(goal_pos)

        if self.detach_com:
            goal_log_prob = goal_log_prob.detach()

        # Only propagate error for bots with goal found and not reached
        prop_err_mask = goal_found_mask * (1. - task_done_mask)

        goal_loss = (goal_log_prob * prop_err_mask).mean()

        return -goal_loss


# ------------------------------------------------------------------------------
# MARK: VisionAuxTask

class VisionAuxTask(AuxTask):
    PX_WEIGHT_LIST = tuple(cfg.PX_WEIGHT_MAP.values())

    WALL_CLR_IDX_OFFSET = len(cfg.COLOURS['background'])
    GOAL_CLR_IDX_OFFSET = len(cfg.COLOURS['background']) + len(cfg.COLOURS['wall'])

    STAT_KEYS = (
        'VisNet/loss',
        'VisNet/dep_mse',
        'VisNet/clr_fce',
        'VisNet/ent_fce',
        'VisNet/obj_nll',
        'VisNet/loc_nll',
        'VisNet/enc_kl')

    def __init__(self, visnet: Module, optimizer, device: str):
        super().__init__(False, True)

        self.visnet = visnet
        self.optimizer = optimizer

        self.hues_wall = rgb_to_hsv(torch.tensor(cfg.COLOURS['wall']))[:, 0].to(device, dtype=torch.float32)
        self.hues_goal = rgb_to_hsv(torch.tensor(cfg.COLOURS['goal']))[:, 0].to(device, dtype=torch.float32)

    def clear(self):
        pass

    # --------------------------------------------------------------------------
    # MARK: collect

    def get_clr_seg(self, hsv: Tensor, ent_seg: Tensor) -> Tensor:

        # Match to nearest known hue
        wall_clr_seg = (hsv[:, 0, ..., None] - self.hues_wall).abs_().argmin(-1) + self.WALL_CLR_IDX_OFFSET
        main_clr_seg = (hsv[:, 0, ..., None] - self.hues_goal).abs_().argmin(-1) + self.GOAL_CLR_IDX_OFFSET

        # Distinguish black & white from red (same hue)
        bw_mask = ent_seg == cfg.ENT_CLS_BEACON
        b_mask = bw_mask & (hsv[:, 2] < 0.5)
        w_mask = bw_mask & ~b_mask

        main_clr_seg[b_mask] = cfg.BLACK_CLR_IDX
        main_clr_seg[w_mask] = cfg.WHITE_CLR_IDX

        clr_seg = torch.where(ent_seg == cfg.ENT_CLS_WALL, wall_clr_seg, main_clr_seg)

        # Explicitly set const. clr. elements
        clr_seg[ent_seg == cfg.ENT_CLS_CHASSIS] = cfg.GREY_CLR_IDX
        clr_seg[ent_seg == cfg.ENT_CLS_FLOOR] = cfg.FLOOR_CLR_IDX
        clr_seg[ent_seg == cfg.ENT_CLS_SKY] = cfg.SKY_CLR_IDX

        return clr_seg

    def get_px_weights(self, ent_seg: Tensor) -> Tensor:
        weights = torch.zeros(ent_seg.shape, device=ent_seg.device)

        for i in range(cfg.N_ENT_CLASSES):
            weights[ent_seg == i] = self.PX_WEIGHT_LIST[i]

        return weights.unsqueeze(1)

    def collect(self, data: TensorDict, obs: 'tuple[Tensor, ...]', mem: 'tuple[Tensor, ...]'):
        obs_img, obs_vec, _ = data['obs']

        hsv = obs_img[:, :3]
        ipt = obs_img[:, :4]
        dep = obs_img[:, 3:4]
        ent_seg = obs_img[:, -1].long()

        obj_in_frame = data['vaux'].split(cfg.AUX_VAL_SPLIT, dim=-1)[0].argmax(-1, keepdim=True)
        bot_pos = obs_vec[:, cfg.BOT_POS_SLICE]

        data['obs'] = (
            ipt, dep, ent_seg, self.get_clr_seg(hsv, ent_seg), self.get_px_weights(ent_seg), obj_in_frame, bot_pos)

    # --------------------------------------------------------------------------
    # MARK: update

    def update(self, batches: 'list[TensorDict]', stats: 'dict[str, Tensor]'):
        for b in batches:
            self.optimizer.zero_grad()

            out = self.visnet(b['obs'][0])

            loss = self.loss(b, None, None, out, stats)
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
        stats: 'dict[str, Tensor]'
    ) -> Tensor:

        # Unpack targets
        _, dep_target, ent_target, clr_target, weights, obj_in_frame, bot_pos = batch['obs']

        # Unpack model outputs
        img_out, _, pred_obj, pred_loc = auxs
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
        loc_nll = FixedVarNormal(pred_loc, skip_log_shift=True).log_prob(bot_pos).mean()

        # Full loss
        loss = 0.05 * dep_mse - clr_fce - ent_fce - obj_nll - loc_nll

        # Stats for logging
        with torch.no_grad():
            stats['VisNet/loss'] += loss.detach()
            stats['VisNet/dep_mse'] += dep_mse.detach()
            stats['VisNet/clr_fce'] -= clr_fce.detach()
            stats['VisNet/ent_fce'] -= ent_fce.detach()
            stats['VisNet/obj_nll'] -= obj_nll.detach()
            stats['VisNet/loc_nll'] -= loc_nll.detach()

        return loss
