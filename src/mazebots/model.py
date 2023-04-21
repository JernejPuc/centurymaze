"""MazeBots AI"""

from typing import Callable

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.functional import scaled_dot_product_attention

from discit.distr import ClippedNormal, FixedVarNormal, MultiCategorical, MultiMixed
from discit.func import symexp
from discit.rl import ActorCritic as ActorCriticTemplate

import config as cfg


VISENC_SIZE = 192
RNNMEM_SIZE = 256
CRITIC_VALS = 1


class VisEncoder(Module):
    def __init__(self):
        super().__init__()

        # ReLU quickly leads to dying neurons
        self.activ = nn.CELU(0.1)

        # 96x48x4 -> 24x12x16
        self.conv1 = nn.Conv2d(cfg.OBS_IMG_CHANNELS, 16, 4, 4, bias=False)
        self.bias1_h = nn.Parameter(torch.zeros(1, 16, 12, 1))
        self.bias1_w = nn.Parameter(torch.zeros(1, 16, 1, 24))

        # 24x12x16 -> 8x4x48 -> 8x4x32
        self.conv2_dw = nn.Conv2d(16, 48, 3, 3, groups=16, bias=False)
        self.conv2_cw = nn.Conv2d(48, 32, 1, 1, bias=False)
        self.bias2_h = nn.Parameter(torch.zeros(1, 32, 4, 1))
        self.bias2_w = nn.Parameter(torch.zeros(1, 32, 1, 8))

        # 8x4x32 -> 4x2x96 -> 4x2x64
        self.conv3_dw = nn.Conv2d(32, 96, 2, 2, groups=32, bias=False)
        self.conv3_cw = nn.Conv2d(96, 64, 1, 1, bias=False)
        self.bias3_h = nn.Parameter(torch.zeros(1, 64, 2, 1))
        self.bias3_w = nn.Parameter(torch.zeros(1, 64, 1, 4))

        # 4x2x64 -> 1x1x192 -> 1x1xE
        self.conv4_dw = nn.Conv2d(64, 192, (2, 4), 1, groups=64, bias=False)
        self.conv4_cw = nn.Conv2d(192, VISENC_SIZE, 1, 1)

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2_dw.weight)
        nn.init.xavier_normal_(self.conv2_cw.weight)
        nn.init.xavier_normal_(self.conv3_dw.weight)
        nn.init.xavier_normal_(self.conv3_cw.weight)
        nn.init.zeros_(self.conv4_dw.weight)
        nn.init.zeros_(self.conv4_cw.bias)

    def forward(self, x0: Tensor) -> Tensor:
        x1 = self.activ(self.conv1(x0) + self.bias1_h + self.bias1_w)
        x2 = self.activ(self.conv2_cw(self.conv2_dw(x1)) + self.bias2_h + self.bias2_w)
        x3 = self.activ(self.conv3_cw(self.conv3_dw(x2)) + self.bias3_h + self.bias3_w)
        x4 = self.activ(self.conv4_cw(self.conv4_dw(x3)))

        return x4, x3, x2, x1

    def encode(self, x: Tensor) -> Tensor:
        return self(x)[0].flatten(1)


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


class VisDecoderRec(Module):
    """
    Reconstruction decoder must extract information only from
    its most compressed representation.
    """

    def __init__(self):
        super().__init__()

        self.activ = nn.SiLU()

        # 1x1xE -> 1x1x512 -> 4x2x64
        # 4x2x64 -> 4x2x128 -> 4x2x64 (2)
        self.up4 = UpBlock(VISENC_SIZE, 512, (-1, 64, 2, 4), 1)
        self.res4_1 = ResBlock(64, 128, 1)
        self.res4_2 = ResBlock(64, 128, 1)

        # 4x2x64 -> 4x2x256 -> 8x4x64
        # 8x4x64 -> 8x4x128 -> 8x4x64 (2)
        self.up3 = UpBlock(64, 256, 2, 1)
        self.res3_1 = ResBlock(64, 128, 2)
        self.res3_2 = ResBlock(64, 128, 2)

        # 8x4x64 -> 8x4x288 -> 24x12x32
        # 24x12x32 -> 24x12x64 -> 24x12x32 (2)
        self.up2 = UpBlock(64, 288, 3, 2)
        self.res2_1 = ResBlock(32, 64, 4)
        self.res2_2 = ResBlock(32, 64, 4)

        # 24x12x32 -> 24x12x128 -> 96x48x8
        # 96x48x8 -> 96x48x16 -> 96x48x8 (2)
        self.up1 = UpBlock(32, 128, 4, 4)
        self.res1_1 = ResBlock(8, 16, 16)
        self.res1_2 = ResBlock(8, 16, 16)

        # 96x48x8 -> 96x48xC
        self.norm0 = nn.GroupNorm(8, 8)
        self.conv0 = nn.Conv2d(8, cfg.OBS_IMG_CHANNELS, 1)

        nn.init.normal_(self.conv0.weight, std=0.001)
        nn.init.zeros_(self.conv0.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.res4_2(self.res4_1(self.up4(x)))
        x = self.res3_2(self.res3_1(self.up3(x)))
        x = self.res2_2(self.res2_1(self.up2(x)))
        x = self.res1_2(self.res1_1(self.up1(x)))
        x = self.conv0(self.norm0(x))

        return x


class VisDecoderSeg(Module):
    """Segmentation decoder can rely on skip connections."""

    def __init__(self):
        super().__init__()

        self.activ = nn.CELU(0.1)

        # 1x1xE -> 1x1x512 -> 4x2x64
        # 4x2x64 -> 4x2x128 -> 4x2x64 (2)
        self.up4 = UpBlock(VISENC_SIZE, 512, (-1, 64, 2, 4), 1)
        self.res4_1 = ResBlock(64, 128, 1)
        self.res4_2 = ResBlock(64, 128, 1)

        # 4x2x(64+64) -> 4x2x256 -> 8x4x64
        # 8x4x64 -> 8x4x128 -> 8x4x64 (2)
        self.up3 = UpBlock(128, 256, 2, 2)
        self.res3_1 = ResBlock(64, 128, 2)
        self.res3_2 = ResBlock(64, 128, 2)

        # 8x4x(64+32) -> 8x4x288 -> 24x12x32
        # 24x12x32 -> 24x12x64 -> 24x12x32 (2)
        self.up2 = UpBlock(96, 288, 3, 4)
        self.res2_1 = ResBlock(32, 64, 4)
        self.res2_2 = ResBlock(32, 64, 4)

        # 24x12x(32+16) -> 24x12x128 -> 96x48x8
        # 96x48x8 -> 96x48x16 -> 96x48x8 (2)
        self.up1 = UpBlock(48, 128, 4, 8)
        self.res1_1 = ResBlock(8, 16, 16)
        self.res1_2 = ResBlock(8, 16, 16)

        # 96x48x8 -> 96x48xC
        self.norm0 = nn.GroupNorm(8, 8)
        self.conv0 = nn.Conv2d(8, cfg.N_SEG_CLASSES, 1)

        nn.init.normal_(self.conv0.weight, std=0.001)
        nn.init.zeros_(self.conv0.bias)

    def forward(self, e4: Tensor, e3: Tensor, e2: Tensor, e1: Tensor) -> Tensor:
        x = self.res4_2(self.res4_1(self.up4(e4)))
        x = torch.cat((x, e3), dim=1)
        x = self.res3_2(self.res3_1(self.up3(x)))
        x = torch.cat((x, e2), dim=1)
        x = self.res2_2(self.res2_1(self.up2(x)))
        x = torch.cat((x, e1), dim=1)
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
        self.decr = VisDecoderRec()
        self.decs = VisDecoderSeg()

    def forward(self, x: Tensor) -> 'tuple[Tensor, Tensor]':
        e = self.enc(x)
        return self.decr(e[0]), self.decs(*e)


class Policy(Module):
    def __init__(self, visual: bool = True, communicating: bool = True, guided: bool = False):
        super().__init__()

        self.act_in_sizes = (
            VISENC_SIZE if visual else 0,
            cfg.OBS_VEC_SIZE - (0 if communicating else cfg.REC_VEC_SIZE),
            cfg.GUIDE_VEC_SIZE if guided else 0)

        act_values_1 = torch.tensor(cfg.ACT_DOF_MODES_BASE)
        act_values_2 = torch.tensor(cfg.ACT_DOF_VAL_MODS)
        act_values = (act_values_1.T.unsqueeze(-1) * act_values_2.T.unsqueeze(-2)).flatten(-2).T

        self.act_values = nn.Parameter(act_values, requires_grad=False)
        self.act_out_sizes = (len(act_values_1), len(act_values_2), cfg.ACT_VEC_SPLIT[1], cfg.ACT_VEC_SPLIT[1])

        self.activ = nn.Tanh()

        # E|0 + 34|25 + 6|0 -> 256
        self.fcin = nn.Linear(sum(self.act_in_sizes), 256)

        # 256 -> M
        self.rnn = nn.GRUCell(256, RNNMEM_SIZE)

        # M -> 5 + 2 + 3*2
        self.fcout = nn.Linear(RNNMEM_SIZE, sum(self.act_out_sizes))

        # https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
        self.mem = nn.Parameter(torch.zeros(1, RNNMEM_SIZE).uniform_(-1., 1.))

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

        # Policy variations
        # NOTE: Bad practice, but covers all combinations without overhead of branching
        lambda_input_str = (
            'lambda img, vec, aux: torch.cat(('
            f'{"img, " if visual else ""}'
            f'{"vec, " if communicating else "vec[:, :-cfg.REC_VEC_SIZE], "}'
            f'{"aux[:, :cfg.GUIDE_VEC_SIZE]" if guided else ""}'
            '), dim=-1)')

        self.get_input: Callable[[Tensor, Tensor, Tensor], Tensor] = eval(lambda_input_str)

    def forward(self, x: Tensor, mem: Tensor) -> 'tuple[Tensor, Tensor]':
        x = self.activ(self.fcin(x))

        mem = self.rnn(x, mem)

        x = self.fcout(mem)

        return x, mem


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

        # M + 12 + M -> 256
        self.fcin = nn.Linear(RNNMEM_SIZE*2 + cfg.STATE_VEC_SIZE, 256)

        # 256 -> 768 -> 3x 256 -> 256
        self.atten = AttentionBlock(256, 64, 4, n_agents_per_env)

        # 256 -> M
        self.rnn = nn.GRUCell(256, RNNMEM_SIZE)

        # M -> V
        self.fcout = nn.Linear(RNNMEM_SIZE, CRITIC_VALS)

        self.mem = nn.Parameter(torch.zeros(1, RNNMEM_SIZE).uniform_(-1., 1.))
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

    def forward(self, x: Tensor, aux: Tensor, mem: Tensor) -> 'tuple[Tensor, Tensor]':
        x = torch.cat((x, aux, mem), dim=-1)
        x = self.activ(self.fcin(x))

        x = self.atten(x)
        mem = self.rnn(x, mem)

        x = self.fcout(mem)

        return x, mem


class ActorCritic(ActorCriticTemplate):
    """Wrapper around the visual encoder, policy, and valuator networks."""

    def __init__(
        self,
        n_agents_per_env: int = 1,
        visual: bool = True,
        communicating: bool = True,
        guided: bool = False,
        prob_actor: bool = True
    ):
        super().__init__()

        self.prob_actor = prob_actor
        self.visual_policy = visual

        self.visencoder = VisEncoder()
        self.policy = Policy(visual, communicating, guided)
        self.valuator = Valuator(n_agents_per_env)

        for param in self.visencoder.parameters():
            param.requires_grad = False

    def init_mem(self, batch_size: int = 1) -> 'tuple[Tensor, Tensor]':
        memp = self.policy.mem.detach().expand(batch_size, -1).clone()
        memv = self.valuator.mem.detach().expand(batch_size, -1).clone()

        return memp, memv

    def reset_mem(
        self,
        mem: 'tuple[Tensor, Tensor]',
        reset_mask: Tensor
    ) -> 'tuple[Tensor, Tensor]':

        reset_mask = reset_mask.unsqueeze(-1)
        memp, memv = mem

        memp = torch.lerp(memp, self.policy.mem, reset_mask)
        memv = torch.lerp(memv, self.valuator.mem, reset_mask)

        return memp, memv

    def get_distr(self, args: 'Tensor | tuple[Tensor, ...]', from_raw: bool) -> MultiMixed:
        if from_raw:
            mcat_logits_1, mcat_logits_2, mnor_mean, mnor_log_dev = args.split(self.policy.act_out_sizes, dim=1)

            mcat = MultiCategorical.from_raw(mcat_logits_1, mcat_logits_2, values=self.policy.act_values)
            mnor = ClippedNormal.from_raw(mnor_mean, mnor_log_dev, -1., 0.01, 3., low=0., high=1.)

        else:
            mcat_log_probs, mnor_mean, mnor_log_dev = args

            mcat = MultiCategorical(self.policy.act_values, mcat_log_probs)
            mnor = ClippedNormal(mnor_mean, mnor_log_dev, low=0., high=1.)

        return MultiMixed(mcat, mnor)

    def act_partial(
        self,
        obs_img: Tensor,
        obs_vec: Tensor,
        obs_aux: Tensor,
        memp: Tensor,
        _
    ) -> Tensor:

        with torch.no_grad():
            if self.visual_policy:
                obs_img = self.visencoder.encode(obs_img)

            obs_vec = self.policy.get_input(obs_img, obs_vec, obs_aux)

            x, new_memp = self.policy(obs_vec, memp)
            memp.copy_(new_memp)

            act = self.get_distr(x, from_raw=True)
            x = act.sample()[0] if self.prob_actor else act.mode

        return x

    def act(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        _sample: bool = None
    ) -> 'tuple[Tensor, tuple[Tensor, ...]]':

        return self.act_partial(*obs, *mem), mem

    def fwd_partial(
        self,
        obs_vec: Tensor,
        obs_aux: Tensor,
        memp: Tensor,
        memv: Tensor,
        detach: bool
    ) -> 'tuple[Tensor, ...]':

        x, memp = self.policy(obs_vec, memp)
        v, memv = self.valuator(memp.detach() if detach else memp, obs_aux, memv)

        return x, v, memp, memv

    def collect_static(
        self,
        obs_img: Tensor,
        obs_vec: Tensor,
        obs_aux: Tensor,
        memp: Tensor,
        memv: Tensor
    ) -> 'tuple[Tensor, ...]':

        with torch.no_grad():
            if self.visual_policy:
                obs_img = self.visencoder.encode(obs_img)

            obs_vec = self.policy.get_input(obs_img, obs_vec, obs_aux)

            x, v, memp, memv = self.fwd_partial(obs_vec, obs_aux, memp, memv, detach=False)

            val_mean = FixedVarNormal(symexp(v)).mean.flatten()

        return x, val_mean, obs_vec, obs_aux, memp, memv

    def collect_copied(
        self,
        obs_vec: Tensor,
        obs_aux: Tensor,
        memp: Tensor,
        memv: Tensor
    ) -> 'tuple[Tensor, ...]':

        with torch.no_grad():
            x, v, memp, memv = self.fwd_partial(obs_vec, obs_aux, memp, memv, detach=False)

            val_mean = FixedVarNormal(symexp(v)).mean.flatten()

        return x, val_mean, memp, memv

    def collect(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        encode: bool
    ) -> 'tuple[Tensor, Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]':

        if encode:
            x, val_mean, obs_vec, obs_aux, memp, memv = self.collect_static(*obs, *mem)
            obs = obs_vec, obs_aux

        else:
            x, val_mean, memp, memv = self.collect_copied(*obs, *mem)

        return x, val_mean, obs, (memp, memv)

    def forward(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        detach: bool = False
    ) -> 'tuple[ClippedNormal, FixedVarNormal, tuple[Tensor, ...]]':

        x, v, memp, memv = self.fwd_partial(*obs, *mem, detach=detach)

        act = self.get_distr(x, from_raw=True)
        val = FixedVarNormal(symexp(v))

        return act, val, (memp, memv)
