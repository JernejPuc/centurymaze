"""MazeBots AI"""

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.functional import one_hot, scaled_dot_product_attention

from discit.distr import FixedVarNormal, MultiCategorical, MultiNormal
from discit.func import WeightedGradDiscretise, symexp
from discit.rl import ActorCritic as ActorCriticTemplate

import config as cfg


VISENC_SIZE = 224
RNNMEM_SIZE = 256
CRITIC_VALS = 1


class VisEncoder(Module):
    """Compresses image data into a vector encoding of visual information."""

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
        self.conv4_cw = nn.Conv2d(240, VISENC_SIZE, 1, 1)

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
        self.up4 = UpBlock(VISENC_SIZE, 1024, (-1, 128, 2, 4), 1)
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
    def __init__(self, n_agents_per_env: int, n_envs: int, communicating: bool, guided: bool):
        super().__init__()

        self.n_bots = n_agents_per_env
        self.n_envs = n_envs
        self.n_all_bots = n_agents_per_env * n_envs
        self.communicating = communicating
        self.unguided = not guided

        self.act_in_sizes = (
            VISENC_SIZE,
            cfg.OBS_VEC_SIZE + (cfg.STATE_VEC_SIZE if guided else 0),
            cfg.OBS_COM_SIZE if communicating else 0)

        trq_values = torch.tensor(cfg.ACT_DOF_MODES_BASE)
        clr_values = torch.tensor(cfg.RCVR_CLR_CLASSES)

        comp_initial = torch.zeros(1, len(clr_values))
        comp_initial[:, -1] = 1.
        coml_initial = torch.full((1, len(clr_values)), -20.)
        coml_initial[:, -1] = 0.

        self.trq_values = nn.Parameter(trq_values, requires_grad=False)
        self.clr_values = nn.Parameter(clr_values, requires_grad=False)
        self.act_value_tpl = (self.trq_values, self.clr_values)
        self.act_out_sizes = (len(trq_values), len(clr_values))

        self.comp_initial = nn.Parameter(comp_initial, requires_grad=False)
        self.coml_initial = nn.Parameter(coml_initial, requires_grad=False)
        self.com_out_idx = self.act_out_sizes[0]

        self.activ = nn.Tanh()

        # E + 0|20 + 68|24 -> 256
        self.fcin = nn.Linear(sum(self.act_in_sizes), 256)

        # 256 -> M
        self.rnn = nn.GRUCell(256, RNNMEM_SIZE)

        # M -> 5 + 11
        self.fcout = nn.Linear(RNNMEM_SIZE, sum(self.act_out_sizes))

        # https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
        self.mem_initial = nn.Parameter(torch.zeros(1, RNNMEM_SIZE).uniform_(-1., 1.))

        nn.init.orthogonal_(self.fcin.weight)
        nn.init.zeros_(self.fcin.bias)
        nn.init.orthogonal_(self.fcout.weight, gain=0.001)
        nn.init.zeros_(self.fcout.bias)

        nn.init.orthogonal_(self.rnn.weight_ih)
        nn.init.orthogonal_(self.rnn.weight_hh)
        nn.init.zeros_(self.rnn.bias_ih)
        nn.init.zeros_(self.rnn.bias_hh)

        # Chrono init. wrt. 1 second (starting from level 1)
        with torch.no_grad():
            self.rnn.bias_hh[RNNMEM_SIZE:-RNNMEM_SIZE].uniform_(1, max(1, cfg.STEPS_PER_SECOND-1)).log_()

    def get_input(self, img_enc: Tensor, vec_etc: Tensor) -> Tensor:
        """Get a single vector input from multiple sources (img, vec, state, com)."""

        if self.unguided:
            vec_etc = vec_etc[:, :-cfg.STATE_VEC_SIZE]

        if not self.communicating:
            return torch.cat((img_enc, vec_etc), dim=-1)

        # Restore colour classes from RGB actions
        rgb = vec_etc[:, cfg.OBS_RGB_SLICE]

        # Nx3, 11x3 -> Nx1x3 - 1x11x3 -> Nx11x3
        diff_to_clrs = rgb.unsqueeze(1) - self.clr_values.unsqueeze(0)

        # Nx11x3 -> Nx11 -> N
        dist_to_clrs = torch.linalg.norm(diff_to_clrs, dim=-1)
        clr_idcs = torch.argmin(dist_to_clrs, dim=-1)

        # N -> Nx11
        com_mask = one_hot(clr_idcs, len(self.clr_values)).float()

        return torch.cat((img_enc, vec_etc, com_mask), dim=-1)

    def weigh_signals(self, sig: Tensor, weights: Tensor, scale_agg: float = 10.) -> Tensor:
        """
        Weigh signal transmissions by strength (proximity) and incoming angle
        wrt. 4 oriented receivers, each covering an angle of 90 degrees.
        """

        # Nx3 -> Ex1xBx3 -> ExBxBx3 -> NxBx3 (repeat_interleave cannot be used directly)
        sig = sig.reshape(self.n_envs, 1, -1, 3).expand(-1, self.n_bots, -1, -1).reshape(self.n_all_bots, -1, 3)

        # NxBx4, NxBxS -> Nx4xS -> Nx(4xS)
        sig_agg = torch.einsum('ijk,ijl->ikl', weights, sig).flatten(1)

        # Limit range of aggregate
        sig_agg = torch.tanh(sig_agg / scale_agg) * scale_agg

        return sig_agg

    def forward(self, x: Tensor, com_weights: Tensor, com_probs: Tensor, memp: Tensor) -> 'tuple[Tensor, Tensor]':
        if self.communicating:
            x, com_mask = x[:, :self.com_out_idx], x[:, self.com_out_idx:]

            com_sig = WeightedGradDiscretise.apply(com_probs, com_mask)
            com_agg = self.weigh_signals(com_sig, com_weights)

            x = torch.cat((x, com_agg), dim=-1)

        x = self.activ(self.fcin(x))

        memp = self.rnn(x, memp)

        x = self.fcout(memp)

        return x, memp


class AttentionBlock(Module):
    def __init__(self, in_dim: int, embed_dim: int, n_heads: int, n_per_slice: int):
        super().__init__()

        self.qkv_dim = embed_dim * 3
        self.res_dim = n_heads * embed_dim
        self.n_heads = n_heads
        self.n_per_slice = n_per_slice

        self.fcin = nn.Linear(in_dim, self.res_dim*3)
        self.fcout = nn.Linear(self.res_dim, in_dim)

        self.norm = nn.LayerNorm(in_dim)
        self.id_mul = nn.Parameter(torch.ones((1, in_dim)))

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
        x = self.norm(x + self.id_mul * x0)

        return x


class Valuator(Module):
    """
    Critic network attending to a variable number of agents per environment.

    Besides the agents' memory vectors, the valuator has access to some other
    variables that are used in reward evaluation and thus, while unknowable
    to the agents, are critical to reliably anticipate future rewards.
    """

    def __init__(self, n_agents_per_env: int):
        super().__init__()

        self.activ = nn.Tanh()

        # 20 + 2*M -> 256
        self.fcin = nn.Linear(cfg.STATE_VEC_SIZE + RNNMEM_SIZE*2, 256)

        # 256 -> 768 -> 3x 256 -> 256
        self.atten = AttentionBlock(256, 64, 4, n_agents_per_env)

        # 256 -> M
        self.rnn = nn.GRUCell(256, RNNMEM_SIZE)

        # M -> V
        self.fcout = nn.Linear(RNNMEM_SIZE, CRITIC_VALS)

        self.mem_initial = nn.Parameter(torch.zeros(1, RNNMEM_SIZE).uniform_(-1., 1.))
        nn.init.orthogonal_(self.fcin.weight)
        nn.init.zeros_(self.fcin.bias)
        nn.init.orthogonal_(self.fcout.weight, gain=0.001)
        nn.init.zeros_(self.fcout.bias)

        nn.init.orthogonal_(self.rnn.weight_ih)
        nn.init.orthogonal_(self.rnn.weight_hh)
        nn.init.zeros_(self.rnn.bias_ih)
        nn.init.zeros_(self.rnn.bias_hh)

        # Chrono init. wrt. 1 second (starting from level 1)
        with torch.no_grad():
            self.rnn.bias_hh[RNNMEM_SIZE:-RNNMEM_SIZE].uniform_(1, max(1, cfg.STEPS_PER_SECOND-1)).log_()

    def forward(self, state: Tensor, memp: Tensor, memv: Tensor) -> 'tuple[Tensor, Tensor]':
        x = torch.cat((state, memp, memv), dim=-1)
        x = self.activ(self.fcin(x))

        x = self.atten(x)
        memv = self.rnn(x, memv)

        x = self.fcout(memv)

        return x, memv


class Aligner(Module):
    def __init__(self):
        super().__init__()

        self.aux_out_sizes = (sum(cfg.AUX_VEC_SPLIT),)*2
        self.fc = nn.Linear(RNNMEM_SIZE, sum(self.aux_out_sizes))

        nn.init.orthogonal_(self.fc.weight, gain=0.001)
        nn.init.zeros_(self.fc.bias)

    def forward(self, memp: Tensor) -> MultiNormal:
        x = self.fc(memp)
        mean, pseudo_log_dev = x.split(self.aux_out_sizes, dim=-1)

        return MultiNormal.from_raw(mean, pseudo_log_dev)


class ActorCritic(ActorCriticTemplate):
    """Wrapper around the visual encoder, policy, valuator, and aligner networks."""

    def __init__(
        self,
        n_agents_per_env: int = 1,
        n_envs: int = 1,
        communicating: bool = True,
        guided: bool = False,
        prob_actor: bool = True
    ):
        super().__init__()

        self.prob_actor = prob_actor
        self.ignore_com = not communicating

        self.visencoder = VisEncoder()
        self.policy = Policy(n_agents_per_env, n_envs, communicating, guided)
        self.valuator = Valuator(n_agents_per_env)
        self.aligner = Aligner()

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
        if not from_raw:
            return MultiCategorical(self.policy.act_value_tpl, args)

        trq_logits, clr_logits = args.split(self.policy.act_out_sizes, dim=-1)

        if self.ignore_com:
            clr_logits = self.policy.coml_initial.expand(len(clr_logits), -1).clone()

        return MultiCategorical.from_raw(self.policy.act_value_tpl, (trq_logits, clr_logits))

    def act_partial(
        self,
        img: Tensor,
        vec_etc: Tensor,
        com_weights: Tensor,
        comp: Tensor,
        memp: Tensor,
        _
    ) -> Tensor:

        with torch.no_grad():
            img_enc = self.visencoder(img).flatten(1)

            x = self.policy.get_input(img_enc, vec_etc)

            x, memp = self.policy(x, com_weights, comp, memp)

        return x, memp

    def act(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        _sample: bool = None
    ) -> 'tuple[Tensor, tuple[Tensor, ...]]':

        comp, memp, memv = mem

        x, memp = self.act_partial(*obs, comp, memp)

        act = self.get_distr(x, from_raw=True)
        comp = act.distrs[1].probs

        x = act.sample()[0] if self.prob_actor else act.mode

        return x, (comp, memp, memv)

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

        x, memp = self.policy(ipt, com_weights, comp, memp)
        v, memv = self.valuator(state, memp.detach() if detach else memp, memv)

        return x, v, memp, memv

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

            ipt = self.policy.get_input(img_enc, vec_etc)
            state = vec_etc[:, -cfg.STATE_VEC_SIZE:]

            x, v, memp, memv = self.fwd_partial(ipt, state, com_weights, comp, memp, memv, detach=False)

            val_mean = FixedVarNormal(symexp(v)).mean.flatten()
            comp = x[:, self.policy.com_out_idx:].softmax(dim=-1)

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
            x, v, memp, memv = self.fwd_partial(ipt, state, com_weights, comp, memp, memv, detach=False)

            val_mean = FixedVarNormal(symexp(v)).mean.flatten()
            comp = x[:, self.policy.com_out_idx:].softmax(dim=-1)

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
    ) -> 'tuple[MultiCategorical, FixedVarNormal, MultiNormal, tuple[Tensor, ...]]':

        x, v, memp, memv = self.fwd_partial(*obs, *mem, detach=detach)
        aux = self.aligner(memp)

        act = self.get_distr(x, from_raw=True)
        val = FixedVarNormal(symexp(v))
        comp = act.distrs[1].probs

        return act, val, aux, (comp, memp, memv)
