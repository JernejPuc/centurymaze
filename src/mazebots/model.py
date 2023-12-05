"""MazeBots AI"""

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.functional import one_hot, scaled_dot_product_attention

from discit.distr import FixedVarNormal, MultiCategorical
from discit.func import GradScaler
from discit.rl import ActorCritic as ActorCriticTemplate

import config as cfg


class VisEncoder(Module):
    """Compresses image data into a vector encoding of visual information."""

    ENC_SIZE = 224

    def __init__(self):
        super().__init__()

        # ReLU quickly leads to dying neurons
        self.activ = nn.CELU(0.1)

        # 96x48x4 -> 24x12x20
        self.conv1 = nn.Conv2d(cfg.OBS_IMG_CHANNELS, 20, 4, 4, bias=False)
        self.bias1_h = nn.Parameter(torch.zeros(1, 20, 12, 1))
        self.bias1_w = nn.Parameter(torch.zeros(1, 20, 1, 24))

        # 24x12x20 -> 8x4x60 -> 8x4x40
        self.conv2_dw = nn.Conv2d(20, 60, 3, 3, groups=20, bias=False)
        self.conv2_cw = nn.Conv2d(60, 40, 1, 1, bias=False)
        self.bias2_h = nn.Parameter(torch.zeros(1, 40, 4, 1))
        self.bias2_w = nn.Parameter(torch.zeros(1, 40, 1, 8))

        # 8x4x40 -> 4x2x120 -> 4x2x80
        self.conv3_dw = nn.Conv2d(40, 120, 2, 2, groups=40, bias=False)
        self.conv3_cw = nn.Conv2d(120, 80, 1, 1, bias=False)
        self.bias3_h = nn.Parameter(torch.zeros(1, 80, 2, 1))
        self.bias3_w = nn.Parameter(torch.zeros(1, 80, 1, 4))

        # 4x2x80 -> 1x1x240 -> 1x1xE
        self.conv4_dw = nn.Conv2d(80, 240, (2, 4), 1, groups=80, bias=False)
        self.conv4_cw = nn.Conv2d(240, self.ENC_SIZE, 1, 1)

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2_dw.weight)
        nn.init.xavier_normal_(self.conv2_cw.weight)
        nn.init.xavier_normal_(self.conv3_dw.weight)
        nn.init.xavier_normal_(self.conv3_cw.weight)
        nn.init.zeros_(self.conv4_dw.weight)
        nn.init.zeros_(self.conv4_cw.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.activ(self.conv1(x) + self.bias1_h + self.bias1_w)
        x = self.activ(self.conv2_cw(self.conv2_dw(x)) + self.bias2_h + self.bias2_w)
        x = self.activ(self.conv3_cw(self.conv3_dw(x)) + self.bias3_h + self.bias3_w)
        x = self.activ(self.conv4_cw(self.conv4_dw(x)))

        return x


class UpBlock(Module):
    def __init__(self, in_channels: int, out_channels: int, r_factor: 'int | tuple[int, ...]', n_groups: int = 1):
        super().__init__()

        self.norm = nn.GroupNorm(min(n_groups, in_channels), in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

        if isinstance(r_factor, int):
            self.reshape = nn.PixelShuffle(r_factor)

        else:
            self.reshape = lambda x: torch.reshape(x, r_factor)

        nn.init.xavier_normal_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.reshape(self.conv(self.norm(x)))


class ResBlock(Module):
    def __init__(self, in_channels: int, mid_channels: int, n_groups: int = 1):
        super().__init__()

        self.activ = nn.CELU(0.1)

        self.norm1 = nn.GroupNorm(min(n_groups, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(min(n_groups, mid_channels), mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, in_channels, 3, padding=1)

        self.id_mul = nn.Parameter(torch.ones(1, in_channels, 1, 1))

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x0: Tensor) -> Tensor:
        x = self.conv1(self.activ(self.norm1(x0)))
        x = self.conv2(self.activ(self.norm2(x)))

        return x + self.id_mul * x0


class VisDecoder(Module):
    """Reproduces image data from a vector encoding of visual information."""

    def __init__(self):
        super().__init__()

        # 1x1xE -> 1x1x1024 -> 4x2x128
        # 4x2x128 -> 4x2x256 -> 4x2x128 (2)
        self.up4 = UpBlock(VisEncoder.ENC_SIZE, 1024, (-1, 128, 2, 4), 1)
        self.res4_1 = ResBlock(128, 256, 1)
        self.res4_2 = ResBlock(128, 256, 1)

        # 4x2x128 -> 4x2x256 -> 8x4x64
        # 8x4x64 -> 8x4x128 -> 8x4x64 (2)
        self.up3 = UpBlock(128, 256, 2, 1)
        self.res3_1 = ResBlock(64, 128, 2)
        self.res3_2 = ResBlock(64, 128, 2)

        # 8x4x64 -> 8x4x288 -> 24x12x32
        # 24x12x32 -> 24x12x64 -> 24x12x32 (2)
        self.up2 = UpBlock(64, 288, 3, 2)
        self.res2_1 = ResBlock(32, 64, 4)
        self.res2_2 = ResBlock(32, 64, 4)

        # 24x12x32 -> 24x12x256 -> 96x48x16
        # 96x48x16 -> 96x48x32 -> 96x48x16 (2)
        self.up1 = UpBlock(32, 256, 4, 16)
        self.res1_1 = ResBlock(16, 32, 32)
        self.res1_2 = ResBlock(16, 32, 32)

        # 96x48x16 -> 96x48xC
        self.norm0 = nn.GroupNorm(16, 16)
        self.conv0 = nn.Conv2d(16, sum(cfg.DEC_IMG_CHANNEL_SPLIT), 1)

        nn.init.normal_(self.conv0.weight, std=0.001)
        nn.init.zeros_(self.conv0.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.res4_2(self.res4_1(self.up4(x)))
        x = self.res3_2(self.res3_1(self.up3(x)))
        x = self.res2_2(self.res2_1(self.up2(x)))
        x = self.res1_2(self.res1_1(self.up1(x)))
        x = self.conv0(self.norm0(x))

        return x


class VisNet(Module):
    """
    Visual auto-encoder network for image processing via compact form.

    Here, the decoders can be more complex than the encoder,
    which targets specific, limited hardware.
    """

    def __init__(self):
        super().__init__()

        self.enc = VisEncoder()
        self.dec = VisDecoder()

    def forward(self, x: Tensor) -> Tensor:
        return self.dec(self.enc(x))


class Policy(Module):
    MEM_SIZE = 256
    MEM_HORIZON = 10 * cfg.STEPS_PER_SECOND

    FEAT_TRAINED = 2
    FEAT_EXTERNAL = 1
    FEAT_DISABLED = 0

    def __init__(self, n_agents_per_env: int, n_envs: int, com_state: int, guide_state: int, silence_bias: bool):
        super().__init__()

        self.n_bots = n_agents_per_env
        self.n_envs = n_envs
        self.n_all_bots = n_agents_per_env * n_envs
        self.com_state = com_state
        self.guide_state = guide_state

        self.communicating = com_state > self.FEAT_DISABLED
        self.guided = guide_state > self.FEAT_DISABLED

        self.input_sizes = (VisEncoder.ENC_SIZE + cfg.IPT_VEC_SPLIT[0], *cfg.IPT_VEC_SPLIT[1:])

        trq_values = torch.tensor(cfg.ACT_DOF_MODES_BASE)
        clr_values = torch.tensor(cfg.RCVR_CLR_CLASSES)

        self.trq_values = nn.Parameter(trq_values, requires_grad=False)
        self.clr_values = nn.Parameter(clr_values, requires_grad=False)
        self.out_value_tpl = (self.trq_values, self.clr_values)
        self.output_sizes = (len(trq_values), len(clr_values))

        comp_silent = torch.zeros(1, len(clr_values))
        comp_silent[:, 0] = 1.
        coml_silent = torch.full((1, len(clr_values)), -20.)
        coml_silent[:, 0] = 0.

        self.comp_initial = nn.Parameter(comp_silent, requires_grad=False)
        self.coml_initial = nn.Parameter(coml_silent, requires_grad=False)

        self.activ = nn.Tanh()

        # E+27 -> 256
        self.fcin = nn.Linear(self.input_sizes[0], 256)

        # 256 + 10 + 0|50 -> M
        self.rnn = nn.GRUCell(
            256 + cfg.AUX_VEC_SPLIT[0] + (cfg.OBS_COM_SIZE if self.communicating else 0),
            self.MEM_SIZE)

        # M -> A
        self.fcaux = nn.Linear(self.MEM_SIZE, cfg.AUX_VEC_SPLIT[1])

        # M + 0|A -> 5 + 11
        self.fcout = nn.Linear(
            self.MEM_SIZE + (cfg.AUX_VEC_SPLIT[1] if self.guided else 0),
            sum(self.output_sizes))

        # https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
        self.mem_initial = nn.Parameter(torch.zeros(1, self.MEM_SIZE).uniform_(-1., 1.))

        nn.init.orthogonal_(self.fcin.weight)
        nn.init.zeros_(self.fcin.bias)
        nn.init.orthogonal_(self.fcout.weight, gain=0.001)
        nn.init.zeros_(self.fcout.bias)
        nn.init.orthogonal_(self.fcaux.weight, gain=0.001)
        nn.init.zeros_(self.fcaux.bias)

        # Init. bias twd. silence over active com.
        if silence_bias:
            coml_silent_bias = torch.tensor([[0.9] + 10*[0.1/10.]]).log_()
            coml_silent_bias -= coml_silent_bias.max()

            with torch.no_grad():
                self.fcout.bias[self.output_sizes[0]:] = coml_silent_bias

        nn.init.orthogonal_(self.rnn.weight_ih)
        nn.init.orthogonal_(self.rnn.weight_hh)
        nn.init.zeros_(self.rnn.bias_ih)
        nn.init.zeros_(self.rnn.bias_hh)

        # Chrono init. wrt. 1 second (starting from level 1)
        with torch.no_grad():
            self.rnn.bias_hh[self.MEM_SIZE:-self.MEM_SIZE].uniform_(1, self.MEM_HORIZON-1).log_()

    def weigh_signals(self, sig: Tensor, weights: Tensor, scale_agg: float = 10., norm: bool = True) -> Tensor:
        """
        Weigh signal transmissions by strength (proximity) and incoming angle
        wrt. 4 oriented receivers, each covering an angle of 90 degrees.
        """

        # NxS -> Ex1xBxS -> ExBxBxS -> NxBxS (repeat_interleave cannot be used directly)
        sig = sig.reshape(self.n_envs, 1, self.n_bots, -1).expand(-1, self.n_bots, -1, -1)
        sig = sig.reshape(self.n_all_bots, self.n_bots, -1)

        # NxBxS, NxBx4 -> NxSx4
        sig_agg = torch.einsum('ijk,ijl->ikl', sig, weights)

        # Limit range of aggregate
        if norm:
            sig_norm = torch.linalg.norm(sig_agg, dim=-1, keepdim=True)
            sig_agg = sig_agg / sig_norm.clip(1e-6)

            # NxSx4, NxSx1 -> NxSx5
            sig_norm = torch.tanh(sig_norm / scale_agg) * scale_agg
            sig_agg = torch.cat((sig_agg, sig_norm), dim=-1)

        else:
            sig_agg = torch.tanh(sig_agg / scale_agg) * scale_agg

        # NxSx4|5-> Nx(S*4|5)
        return sig_agg.flatten(1)

    def forward(self, x: Tensor, com_weights: Tensor, com_probs: Tensor, memp: Tensor) -> 'tuple[Tensor, ...]':
        x, obj_ext, guide_ext, meta = x.split(self.input_sizes, dim=-1)
        nearest_obj_dist, nearest_obj_idx, com_idx = meta[:, 0], meta[:, 1].long(), meta[:, 2].long()

        # Process observations
        x = self.activ(self.fcin(x))

        # Process incoming messages
        if self.communicating:
            com_sig = com_probs / com_probs.detach().clip(1e-9) * one_hot(com_idx, cfg.N_RCVR_CLR_CLASSES).float()

            com_agg_far = self.weigh_signals(com_sig[:, 2:], com_weights[0])
            com_agg_near = self.weigh_signals(com_sig[:, 1:2], com_weights[1])

            x = torch.cat((x, obj_ext, com_agg_far, com_agg_near), dim=-1)

        else:
            x = torch.cat((x, obj_ext), dim=-1)

        # Update memory/state
        memp = self.rnn(x, memp)

        # Auxiliary path
        a = self.fcaux(GradScaler.apply(memp, 0.05) if self.guide_state == self.FEAT_TRAINED else memp.detach())

        a_dir, a_prox = a[:, :2], a[:, 2:]
        a_dir = a_dir / torch.linalg.norm(a_dir, dim=-1, keepdim=True)
        a_prox = (a_prox - 1.).sigmoid()

        a = torch.cat((a_dir, a_prox), dim=-1)

        if self.guide_state == self.FEAT_DISABLED:
            x = memp

        elif self.guide_state == self.FEAT_EXTERNAL:
            x = torch.cat((memp, guide_ext), dim=-1)

        else:
            x = torch.cat((memp, a.detach()), dim=-1)

        # Output
        x = self.fcout(x)

        # Override - silent
        if self.com_state == self.FEAT_DISABLED:
            trq_logits, clr_logits = x.split(self.output_sizes, dim=-1)
            clr_logits = self.coml_initial.expand(len(clr_logits), -1)

            x = torch.cat((trq_logits, clr_logits), dim=-1)

        # Override - heuristic
        elif self.com_state == self.FEAT_EXTERNAL:
            trq_logits, clr_logits = x.split(self.output_sizes, dim=-1)

            # Obj. colour if seen and near, white if obstructed, black otherwise
            com_idx = torch.where(
                nearest_obj_dist < (3. * cfg.GOAL_RADIUS),
                nearest_obj_idx + 2,
                (com_weights[1].sum(1)[:, 0] > 1e-6).long())

            clr_logits = -20. * torch.logical_not(one_hot(com_idx, cfg.N_RCVR_CLR_CLASSES))

            x = torch.cat((trq_logits, clr_logits), dim=-1)

        return x, a, memp


class AttentionBlock(Module):
    def __init__(self, in_dim: int, embed_dim: int, n_heads: int, n_per_slice: int, out_dim: int = None):
        super().__init__()

        self.qkv_dim = embed_dim * 3
        self.res_dim = n_heads * embed_dim
        self.n_heads = n_heads
        self.n_per_slice = n_per_slice

        if out_dim is None:
            out_dim = self.res_dim

        self.fcin = nn.Linear(in_dim, self.res_dim*3)
        self.fcout = nn.Linear(self.res_dim, out_dim)

        self.norm = nn.LayerNorm(out_dim)

        nn.init.orthogonal_(self.fcin.weight)
        nn.init.zeros_(self.fcin.bias)
        nn.init.zeros_(self.fcout.weight)
        nn.init.zeros_(self.fcout.bias)

    def forward(self, x0: Tensor) -> Tensor:
        qkv = self.fcin(x0)

        # BxH3D -> NxLxHx3D -> NxHxLx3D -> 3x NxHxLxD
        qkv = qkv.reshape(-1, self.n_per_slice, self.n_heads, self.qkv_dim)
        qkv = qkv.transpose(1, 2)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # 3x NxHxLxD -> NxHxLxD
        x = scaled_dot_product_attention(q, k, v)

        # NxHxLxD -> NxLxHxD -> BxHD
        x = x.transpose(1, 2)
        x = x.reshape(-1, self.res_dim)

        x = self.fcout(x)
        x = self.norm(x)

        return x


class Valuator(Module):
    """
    Critic network attending to a variable number of agents per environment.

    Besides the agents' memory vectors, the valuator has access to some other
    variables that are used in reward evaluation and thus, while unknowable
    to the agents, are critical to reliably anticipate future rewards.
    """

    MEM_SIZE = 256
    MEM_HORIZON = 20 * cfg.STEPS_PER_SECOND

    def __init__(self, n_agents_per_env: int):
        super().__init__()

        self.activ = nn.Tanh()

        # 60 + M -> 256
        self.fcin = nn.Linear(cfg.STATE_VEC_SIZE + Policy.MEM_SIZE, 256)

        # 256 + M -> 768 -> 3x 256 -> 256
        self.atten = AttentionBlock(256 + self.MEM_SIZE, 64, 4, n_agents_per_env)

        # 256 + 256 -> M
        self.rnn = nn.GRUCell(256 + 256, self.MEM_SIZE)

        # M -> V
        self.fcout = nn.Linear(self.MEM_SIZE, 1)

        self.mem_initial = nn.Parameter(torch.zeros(1, self.MEM_SIZE).uniform_(-1., 1.))

        nn.init.orthogonal_(self.fcin.weight)
        nn.init.zeros_(self.fcin.bias)
        nn.init.orthogonal_(self.fcout.weight, gain=0.001)
        nn.init.zeros_(self.fcout.bias)

        nn.init.orthogonal_(self.rnn.weight_ih)
        nn.init.orthogonal_(self.rnn.weight_hh)
        nn.init.zeros_(self.rnn.bias_ih)
        nn.init.zeros_(self.rnn.bias_hh)

        with torch.no_grad():
            self.rnn.bias_hh[self.MEM_SIZE:-self.MEM_SIZE].uniform_(1, self.MEM_HORIZON-1).log_()

    def forward(self, state: Tensor, memp: Tensor, memv: Tensor) -> 'tuple[Tensor, Tensor]':
        x = torch.cat((state, memp), dim=-1)
        x = self.activ(self.fcin(x))

        x_atn = torch.cat((x, memv), dim=-1)
        x_atn = self.atten(x_atn)

        x = torch.cat((x, x_atn), dim=-1)
        memv = self.rnn(x, memv)

        x = self.fcout(memv)

        return x, memv


class ActorCritic(ActorCriticTemplate):
    """Wrapper around the visual encoder, policy, and valuator networks."""

    def __init__(
        self,
        n_agents_per_env: int = 1,
        n_envs: int = 1,
        com_state: int = Policy.FEAT_TRAINED,
        guide_state: int = Policy.FEAT_DISABLED,
        silence_bias: bool = True,
        prob_actor: bool = True
    ):
        super().__init__()

        self.prob_actor = prob_actor

        self.visencoder = VisEncoder()
        self.policy = Policy(n_agents_per_env, n_envs, com_state, guide_state, silence_bias)
        self.valuator = Valuator(n_agents_per_env)

        for param in self.visencoder.parameters():
            param.requires_grad = False

    def init_mem(self, batch_size: int = 1) -> 'tuple[Tensor, Tensor, Tensor]':
        comp = self.policy.comp_initial.expand(batch_size, -1).clone()
        memp = self.policy.mem_initial.detach().expand(batch_size, -1).clone()
        memv = self.valuator.mem_initial.detach().expand(batch_size, -1).clone()

        return comp, memp, memv

    def reset_mem(
        self,
        mem: 'tuple[Tensor, Tensor, Tensor]',
        reset_mask: Tensor
    ) -> 'tuple[Tensor, Tensor, Tensor]':

        reset_mask = reset_mask.unsqueeze(-1)
        comp, memp, memv = mem

        comp = torch.lerp(comp, self.policy.comp_initial, reset_mask)
        memp = torch.lerp(memp, self.policy.mem_initial, reset_mask)
        memv = torch.lerp(memv, self.valuator.mem_initial, reset_mask)

        return comp, memp, memv

    def get_distr(self, args: 'Tensor | tuple[Tensor, ...]', from_raw: bool) -> MultiCategorical:
        if from_raw:
            return MultiCategorical.from_raw(self.policy.out_value_tpl, args.split(self.policy.output_sizes, dim=-1))

        return MultiCategorical(self.policy.out_value_tpl, args)

    def unwrap_sample(self, sample: 'tuple[Tensor, Tensor]') -> 'tuple[Tensor, Tensor]':
        return sample[0], sample[1][1, :, 0]

    def act_partial(
        self,
        img: Tensor,
        vec_etc: Tensor,
        com_weights: Tensor,
        comp: Tensor,
        memp: Tensor,
        _memv: Tensor
    ) -> 'tuple[Tensor, Tensor]':

        with torch.no_grad():
            img_enc = self.visencoder(img).flatten(1)

            ipt = torch.cat((img_enc, vec_etc[:, cfg.OBS_EXT_SLICE], vec_etc[:, cfg.META_VEC_SLICE]), dim=-1)

            x, _, memp = self.policy(ipt, com_weights, comp, memp)

        return x, memp

    def act(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        _sample: bool = None
    ) -> 'tuple[Tensor, Tensor, tuple[Tensor, ...]]':

        x, memp = self.act_partial(*obs, *mem)

        act = self.get_distr(x, from_raw=True)
        comp = act.distrs[1].probs

        if self.prob_actor:
            x, com_idx = self.unwrap_sample(act.sample())

        else:
            x = act.mode
            com_idx = comp.argmax(dim=-1)

        return x, com_idx, (comp, memp, mem[-1])

    def fwd_partial(
        self,
        ipt: Tensor,
        state: Tensor,
        com_weights: Tensor,
        comp: Tensor,
        memp: Tensor,
        memv: Tensor,
        detach: bool
    ) -> 'tuple[Tensor, ...]':

        x, a, memp = self.policy(ipt, com_weights, comp, memp)
        v, memv = self.valuator(state, memp.detach() if detach else GradScaler.apply(memp, 0.05), memv)

        return x, v, a, memp, memv

    def collect_static(
        self,
        img: Tensor,
        vec_etc: Tensor,
        com_weights: Tensor,
        comp: Tensor,
        memp: Tensor,
        memv: Tensor
    ) -> 'tuple[Tensor, ...]':

        with torch.no_grad():
            img_enc = self.visencoder(img).flatten(1)

            ipt = torch.cat((img_enc, vec_etc[:, cfg.OBS_EXT_SLICE], vec_etc[:, cfg.META_VEC_SLICE]), dim=-1)
            state = vec_etc[:, cfg.STATE_VEC_SLICE]

            x, v, _, memp, memv = self.fwd_partial(ipt, state, com_weights, comp, memp, memv, detach=False)

            val_mean = FixedVarNormal(v).mean.flatten()
            comp = x[:, self.policy.output_sizes[0]:].softmax(dim=-1)

        return x, val_mean, ipt, state, comp, memp, memv

    def collect_copied(
        self,
        ipt: Tensor,
        state: Tensor,
        com_weights: Tensor,
        comp: Tensor,
        memp: Tensor,
        memv: Tensor
    ) -> 'tuple[Tensor, ...]':

        with torch.no_grad():
            x, v, _, memp, memv = self.fwd_partial(ipt, state, com_weights, comp, memp, memv, detach=False)

            val_mean = FixedVarNormal(v).mean.flatten()
            comp = x[:, self.policy.output_sizes[0]:].softmax(dim=-1)

        return x, val_mean, comp, memp, memv

    def collect(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        encode: bool
    ) -> 'tuple[Tensor, Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]':

        if encode:
            x, val_mean, ipt, state, comp, memp, memv = self.collect_static(*obs, *mem)
            obs = ipt, state, obs[-1].clone()

        else:
            x, val_mean, comp, memp, memv = self.collect_copied(*obs, *mem)

        return x, val_mean, obs, (comp, memp, memv)

    def forward(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        detach: bool = False
    ) -> 'tuple[MultiCategorical, FixedVarNormal, FixedVarNormal, tuple[Tensor, ...]]':

        x, v, a, memp, memv = self.fwd_partial(*obs, *mem, detach=detach)

        act = self.get_distr(x, from_raw=True)
        comp = act.distrs[1].probs

        val = FixedVarNormal(v, skip_log_shift=True)
        aux = FixedVarNormal(a, skip_log_shift=True)

        return act, val, aux, (comp, memp, memv)
