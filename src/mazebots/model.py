"""MazeBots AI"""

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.functional import scaled_dot_product_attention

from discit.distr import IndepNormal, OnlyMean
from discit.func import symexp
from discit.rl import ActorCriticTemplate

import config as cfg


VISENC_SIZE = 192
RNNMEM_SIZE = 256
CRITIC_VALS = 1


class VisEncoder(Module):
    def __init__(self):
        super().__init__()

        # ReLU quickly leads to dying neurons
        self.act = nn.CELU(0.1)

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
        x1 = self.act(self.conv1(x0) + self.bias1_h + self.bias1_w)
        x2 = self.act(self.conv2_cw(self.conv2_dw(x1)) + self.bias2_h + self.bias2_w)
        x3 = self.act(self.conv3_cw(self.conv3_dw(x2)) + self.bias3_h + self.bias3_w)
        x4 = self.act(self.conv4_cw(self.conv4_dw(x3)))

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

        self.act = nn.CELU(0.1)

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
        x = self.conv1(self.act(self.norm1(x0)))
        x = self.conv2(self.act(self.norm2(x)))

        return x + self.id_mul * x0


class VisDecoderRec(Module):
    """
    Reconstruction decoder must extract information only from
    its most compressed representation.
    """

    def __init__(self):
        super().__init__()

        self.act = nn.SiLU()

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

        self.act = nn.CELU(0.1)

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
    def __init__(self, communicating: bool = True, guided: bool = False):
        super().__init__()

        self.act = nn.Tanh()

        # E + 34|25 + 0|6 -> 256
        self.fcin = nn.Linear(
            VISENC_SIZE
            + cfg.OBS_VEC_SIZE - (0 if communicating else cfg.REC_VEC_SIZE)
            + (cfg.GUIDE_VEC_SIZE if guided else 0),
            256)

        # 256 -> M
        self.rnn = nn.GRUCell(256, RNNMEM_SIZE)

        # M -> 7 + 7
        self.fcout = nn.Linear(RNNMEM_SIZE, cfg.ACT_VEC_SIZE*2)

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
        if communicating and not guided:
            def get_vec_input(x: Tensor, aux: Tensor) -> Tensor:
                return x

        elif not communicating and guided:
            def get_vec_input(x: Tensor, aux: Tensor) -> Tensor:
                return torch.cat((x[:, :-cfg.REC_VEC_SIZE], aux[:, :cfg.GUIDE_VEC_SIZE]), dim=-1)

        elif not communicating and not guided:
            def get_vec_input(x: Tensor, aux: Tensor) -> Tensor:
                return x[:, :-cfg.REC_VEC_SIZE]

        else:
            def get_vec_input(x: Tensor, aux: Tensor) -> Tensor:
                return torch.cat((x, aux[:, :cfg.GUIDE_VEC_SIZE]), dim=-1)

        self.get_vec_input = get_vec_input

    def forward(self, x: Tensor, aux: Tensor, mem: Tensor) -> 'tuple[Tensor, Tensor]':
        x = self.get_vec_input(x, aux)
        x = self.act(self.fcin(x))

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

        self.act = nn.Tanh()

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
        x = self.act(self.fcin(x))

        x = self.atten(x)
        mem = self.rnn(x, mem)

        x = self.fcout(mem)

        return x, mem


class ActorCritic(ActorCriticTemplate):
    """Wrapper around the visual encoder, policy, and valuator networks."""

    def __init__(
        self,
        n_agents_per_env: int = 1,
        communicating: bool = True,
        guided: bool = False,
        prob_actor: bool = True
    ):
        super().__init__()

        self.prob_actor = prob_actor
        self.act_out_sizes = (cfg.ACT_VEC_SIZE,)*2

        self.visencoder = VisEncoder()
        self.policy = Policy(communicating, guided)
        self.valuator = Valuator(n_agents_per_env)

        for param in self.visencoder.parameters():
            param.requires_grad = False

    def init_mem(self, batch_size: int = 1, detach: bool = False) -> 'tuple[Tensor, Tensor]':
        memp = self.policy.mem.expand(batch_size, -1)
        memv = self.valuator.mem.expand(batch_size, -1)

        if detach:
            memp = memp.detach().clone()
            memv = memv.detach().clone()

        return memp, memv

    def reset_mem(
        self,
        mem: 'tuple[Tensor, Tensor]',
        reset_mask: Tensor,
        keep_mask: Tensor = None
    ) -> 'tuple[Tensor, Tensor]':

        reset_mask = reset_mask[..., None]
        keep_mask = (1. - reset_mask) if keep_mask is None else keep_mask[..., None]

        memp = keep_mask * mem[0] + reset_mask * self.policy.mem
        memv = keep_mask * mem[1] + reset_mask * self.valuator.mem

        return memp, memv

    def get_distr(self, args: 'tuple[Tensor, ...]') -> IndepNormal:
        return IndepNormal(*args, pseudo=False)

    def fwd_partial(
        self,
        obs_img: Tensor,
        obs_vec: Tensor,
        obs_aux: Tensor,
        memp: Tensor,
        memv: Tensor
    ) -> 'tuple[Tensor, ...]':

        with torch.no_grad():
            obs_img = self.visencoder.encode(obs_img)
            obs_vec = torch.cat((obs_img, obs_vec), dim=-1)

            x, memp = self.policy(obs_vec, obs_aux, memp)
            v, memv = self.valuator(memp, obs_aux, memv)

            val_mean = OnlyMean(symexp(v)).mean.flatten()

        return x, val_mean, obs_vec, obs_aux, memp, memv

    def fwd_partial_actor(
        self,
        obs_img: Tensor,
        obs_vec: Tensor,
        obs_aux: Tensor,
        memp: Tensor,
        _
    ) -> Tensor:

        with torch.no_grad():
            obs_img = self.visencoder.encode(obs_img)
            obs_vec = torch.cat((obs_img, obs_vec), dim=-1)

            x, new_memp = self.policy(obs_vec, obs_aux, memp)
            memp.copy_(new_memp)

        return x

    def fwd_actor(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]'
    ) -> 'tuple[Tensor, tuple[Tensor, ...]]':

        x = self.fwd_partial_actor(*obs, *mem)
        act = IndepNormal(*x.split(self.act_out_sizes, dim=1), pseudo=True)

        return act.sample() if self.prob_actor else act.loc, mem

    def fwd_critic(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]'
    ) -> 'tuple[Tensor, tuple[Tensor, ...]]':

        _, val_mean, _, _, memp, memv = self.fwd_partial(*obs, *mem)

        return val_mean, (memp, memv)

    def fwd_collector(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]'
    ) -> 'tuple[tuple[Tensor, ...], Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]':

        x, val_mean, obs_vec, obs_aux, memp, memv = self.fwd_partial(*obs, *mem)

        act = IndepNormal(*x.split(self.act_out_sizes, dim=1), pseudo=True)
        act_args = (act.loc, act.scale, act.sample())

        return act_args, val_mean, (obs_vec, obs_aux), (memp, memv)

    def fwd_learner(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]'
    ) -> 'tuple[IndepNormal, OnlyMean, tuple[Tensor, ...]]':

        obs_vec, obs_aux = obs
        memp, memv = mem

        x, memp = self.policy(obs_vec, obs_aux, memp)
        v, memv = self.valuator(memp, obs_aux, memv)

        act = IndepNormal(*x.split(self.act_out_sizes, dim=1), pseudo=True)
        val = OnlyMean(symexp(v))

        return act, val, (memp, memv)
