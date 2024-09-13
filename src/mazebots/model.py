"""MazeBots AI"""

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.functional import scaled_dot_product_attention

from discit.distr import FixedVarNormal, Categorical
from discit.func import GradScaler
from discit.rl import ActorCritic as ActorCriticTemplate

import config as cfg


# ------------------------------------------------------------------------------
# MARK: VisEncoder

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
        self.conv4_cw = nn.Conv2d(240, cfg.VIS_ENC_SIZE, 1, 1)

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


# ------------------------------------------------------------------------------
# MARK: UpBlock

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


# ------------------------------------------------------------------------------
# MARK: ResBlock

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


# ------------------------------------------------------------------------------
# MARK: VisDecoder

class VisDecoder(Module):
    """Reproduces image data from a vector encoding of visual information."""

    def __init__(self):
        super().__init__()

        # 1x1xE -> 1x1x1024 -> 4x2x128
        # 4x2x128 -> 4x2x256 -> 4x2x128 (2)
        self.up4 = UpBlock(cfg.VIS_ENC_SIZE, 1024, (-1, 128, 2, 4), 1)
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


# ------------------------------------------------------------------------------
# MARK: VisNet

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


# ------------------------------------------------------------------------------
# MARK: MapEncoder

class MapEncoder(Module):
    """Conv. encoder for processing global state information in spatial form."""

    def __init__(self):
        super().__init__()

        self.activ = nn.CELU(0.1)

        # 20x20x15 -> 20x20x16
        self.conv1 = nn.Conv2d(cfg.STATE_MAP_CHANNELS, 16, 3, padding=1)
        self.norm1 = nn.GroupNorm(4, 16)

        # 20x20x16 -> 10x10x24
        self.conv2 = nn.Conv2d(16, 24, 2, 2, bias=False)
        self.bias2_h = nn.Parameter(torch.zeros(1, 24, 10, 1))
        self.bias2_w = nn.Parameter(torch.zeros(1, 24, 1, 10))
        self.norm2 = nn.GroupNorm(2, 24)

        # 10x10x24 -> 5x5x48
        self.conv3 = nn.Conv2d(24, 48, 2, 2, bias=False)
        self.bias3_h = nn.Parameter(torch.zeros(1, 48, 5, 1))
        self.bias3_w = nn.Parameter(torch.zeros(1, 48, 1, 5))
        self.norm3 = nn.GroupNorm(1, 48)

        # 5x5x48 -> 1x1xE
        self.conv4 = nn.Conv2d(48, cfg.MAP_ENC_SIZE, 5)
        self.norm4 = nn.GroupNorm(1, cfg.MAP_ENC_SIZE)

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.conv3.weight)
        nn.init.xavier_normal_(self.conv4.weight)
        nn.init.zeros_(self.conv4.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.activ(self.norm1(self.conv1(x)))
        x = self.activ(self.norm2(self.conv2(x) + self.bias2_h + self.bias2_w))
        x = self.activ(self.norm3(self.conv3(x) + self.bias3_h + self.bias3_w))
        x = self.activ(self.norm4(self.conv4(x)))

        return x.flatten(1)


# ------------------------------------------------------------------------------
# MARK: TopMAC

class TopMAC(Module):

    def __init__(self, n_envs: int, n_bots: int):
        super().__init__()

        self.n_envs = n_envs
        self.n_bots = n_bots
        self.n_all_bots = n_envs * n_bots

        self.q = nn.Parameter(torch.empty(1, cfg.N_TOPICS, cfg.K_DIM))
        self.k = nn.Parameter(torch.empty(1, cfg.N_TOPICS, cfg.K_DIM))

    def forward(self, qkv: Tensor, mask: Tensor) -> Tensor:
        qkv = qkv.reshape(self.n_envs, self.n_bots, -1)
        q, k, v = qkv.split(cfg.QKV_SPLIT, dim=-1)

        top_q = self.q.expand(self.n_envs, -1, -1)
        top_k = self.k.expand(self.n_envs, -1, -1)

        # N -> Ex1xB
        mask = torch.where(mask.bool(), 0., -1e+6).reshape(self.n_envs, 1, self.n_bots)

        # ExTxQ, ExBxK, ExBxV -> ExTxV
        top_v = scaled_dot_product_attention(top_q, k, v, mask)

        # ExBxQ, ExTxK, ExTxV -> ExBxV
        x = scaled_dot_product_attention(q, top_k, top_v)

        return x.reshape(self.n_all_bots, -1)


# ------------------------------------------------------------------------------
# MARK: TarMAC

class TarMAC:
    def __init__(self, n_envs: int, n_bots: int):
        self.n_envs = n_envs
        self.n_bots = n_bots
        self.n_all_bots = n_envs * n_bots

    def __call__(self, qkv: Tensor, mask: Tensor) -> Tensor:
        qkv = qkv.reshape(self.n_envs, self.n_bots, -1)
        q, k, v = qkv.split(cfg.QKV_SPLIT, dim=-1)

        # N -> Ex1xB
        mask = torch.where(mask.bool(), 0., -1e+6).reshape(self.n_envs, 1, self.n_bots)

        # ExBxQ, ExBxK, ExBxV -> ExBxV
        x = scaled_dot_product_attention(q, k, v, mask)

        return x.reshape(self.n_all_bots, -1)


# ------------------------------------------------------------------------------
# MARK: NoMAC

class NoMAC(Module):
    def __init__(self, n_envs: int, n_bots: int):
        super().__init__()

        self.zeros = nn.Parameter(torch.empty(n_envs * n_bots, cfg.V_DIM), requires_grad=False)

    def forward(self, qkv: Tensor, mask: Tensor) -> Tensor:
        return self.zeros


# ------------------------------------------------------------------------------
# MARK: Policy

class Policy(Module):
    def __init__(self, com_mode: int, n_envs: int, n_bots: int):
        super().__init__()

        self.com_mode = com_mode
        self.n_envs = n_envs
        self.n_bots = n_bots
        self.n_all_bots = n_envs * n_bots

        self.com_mask_idx = cfg.VIS_ENC_SIZE + cfg.LED_ON_IDX
        self.qkv_initial = nn.Parameter(torch.zeros(1, sum(cfg.QKV_SPLIT)), requires_grad=False)

        act_values = torch.tensor(cfg.ACT_VALUES)
        self.act_values = nn.Parameter(act_values, requires_grad=False)
        self.out_sizes = (len(cfg.ACT_VALUES), *cfg.AUX_VAL_SPLIT, sum(cfg.QKV_SPLIT))

        self.activ = nn.Tanh()

        # 128 -> 64
        self.com = (NoMAC, TarMAC, TopMAC)[com_mode](n_envs, n_bots)

        # 224 + 28 + 64 -> 320
        self.fcin = nn.Linear(cfg.VIS_ENC_SIZE + cfg.OBS_VEC_SIZE + cfg.V_DIM, cfg.POL_ENC_SIZE)

        # 320 -> 256
        self.rnn = nn.GRUCell(cfg.POL_ENC_SIZE, cfg.MEM_SIZE)

        # 256 -> 10 + 13 + 128
        self.fcout = nn.Linear(cfg.MEM_SIZE, sum(self.out_sizes))

        # https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
        self.mem_initial = nn.Parameter(torch.zeros(1, cfg.MEM_SIZE).uniform_(-1., 1.))

    def random_init(self):
        nn.init.orthogonal_(self.fcin.weight)
        nn.init.zeros_(self.fcin.bias)
        nn.init.orthogonal_(self.fcout.weight, gain=0.001)
        nn.init.zeros_(self.fcout.bias)

        for weight in (self.rnn.weight_ih, self.rnn.weight_hh):
            nn.init.orthogonal_(weight[:cfg.MEM_SIZE])
            nn.init.orthogonal_(weight[cfg.MEM_SIZE:-cfg.MEM_SIZE])
            nn.init.orthogonal_(weight[-cfg.MEM_SIZE:])

        nn.init.zeros_(self.rnn.bias_ih)
        nn.init.zeros_(self.rnn.bias_hh)

        # Multi-horizon chrono init.
        with torch.no_grad():
            chrono_bias = torch.cat([
                torch.empty(cfg.CHRONO_SIZE).uniform_(ti, tj)
                for ti, tj in zip(cfg.CHRONO_RANGES[:-1], cfg.CHRONO_RANGES[1:])])

            self.rnn.bias_hh[cfg.MEM_SIZE:-cfg.MEM_SIZE] = chrono_bias.mul_(cfg.STEPS_PER_SECOND).log_()

    def forward(self, x: Tensor, qkv: Tensor, mem: Tensor) -> 'tuple[Tensor, ...]':
        goal_found_mask = x[:, cfg.GOAL_FOUND_SLICE]
        speaking_mask = x[:, self.com_mask_idx]

        # Process messages
        msg = self.com(qkv, speaking_mask)

        # Process observations
        x = self.activ(self.fcin(torch.cat((x, msg), dim=-1)))

        # Update memory/state and output
        mem = self.rnn(x, mem)

        # Split output into actions, beliefs, and message components
        a, bs, bg, qkv = self.fcout(mem).split(self.out_sizes, dim=-1)

        # Detach grads. for unfound goals
        bg = torch.lerp(bg.detach(), bg, goal_found_mask)

        return a, bs, bg, qkv, mem


# ------------------------------------------------------------------------------
# MARK: AttentionBlock

class AttentionBlock(Module):
    def __init__(self, in_dim: int, embed_dim: int, n_heads: int, n_per_group: int):
        super().__init__()

        self.qkv_dim = embed_dim * 3
        self.out_dim = n_heads * embed_dim
        self.n_heads = n_heads
        self.n_per_group = n_per_group

        self.fc = nn.Linear(in_dim, self.out_dim*3)
        self.norm = nn.LayerNorm(self.out_dim)

        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: Tensor) -> Tensor:
        qkv = self.fc(x)

        # Bx(H*3*E) -> NxSxHx(3*E) -> NxHxSx(3*E) -> 3x NxHxSxE
        qkv = qkv.reshape(-1, self.n_per_group, self.n_heads, self.qkv_dim)
        qkv = qkv.transpose(1, 2)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # 3x NxHxSxE -> NxHxSxE
        x = scaled_dot_product_attention(q, k, v)

        # NxHxSxE -> NxSxHxE -> Bx(H*E)
        x = x.transpose(1, 2)
        x = x.reshape(-1, self.out_dim)

        x = self.norm(x)

        return x


# ------------------------------------------------------------------------------
# MARK: Valuator

class Valuator(Module):
    """
    Critic network attending to a variable number of agents per environment.

    Besides the agents' memory vectors, the valuator has access to some other
    variables that are used in reward evaluation and thus, while unknowable
    to the agents, are critical to reliably anticipate future rewards.
    """

    def __init__(self, n_envs: int, n_bots: int):
        super().__init__()

        self.n_bots = n_bots
        self.n_all_bots = n_envs * n_bots

        self.activ = nn.Tanh()

        # 20x20x15 -> 128
        self.mapenc = MapEncoder()

        # 108 + 128 + 256 -> 512
        self.fcin = nn.Linear(cfg.STATE_VEC_SIZE + cfg.MAP_ENC_SIZE + cfg.MEM_SIZE, cfg.VAL_ENC_SIZE)

        # 512 -> 256
        self.rnn = nn.GRUCell(cfg.VAL_ENC_SIZE, cfg.MEM_SIZE)

        # 256 -> 256 -> 3
        self.mlpout = nn.Sequential(
            nn.Linear(cfg.MEM_SIZE, cfg.VAL_PEN_SIZE),
            self.activ,
            nn.Linear(cfg.VAL_PEN_SIZE, len(cfg.DISCOUNTS)))

        self.mem_initial = nn.Parameter(torch.zeros(1, cfg.MEM_SIZE).uniform_(-1., 1.))

    def random_init(self):
        nn.init.orthogonal_(self.fcin.weight)
        nn.init.zeros_(self.fcin.bias)
        nn.init.orthogonal_(self.mlpout[0].weight)
        nn.init.zeros_(self.mlpout[0].bias)
        nn.init.orthogonal_(self.mlpout[2].weight, gain=0.001)
        nn.init.zeros_(self.mlpout[2].bias)

        for weight in (self.rnn.weight_ih, self.rnn.weight_hh):
            nn.init.orthogonal_(weight[:cfg.MEM_SIZE])
            nn.init.orthogonal_(weight[cfg.MEM_SIZE:-cfg.MEM_SIZE])
            nn.init.orthogonal_(weight[-cfg.MEM_SIZE:])

        nn.init.zeros_(self.rnn.bias_ih)
        nn.init.zeros_(self.rnn.bias_hh)

        # Multi-horizon chrono init.
        with torch.no_grad():
            chrono_bias = torch.cat([
                torch.empty(cfg.CHRONO_SIZE).uniform_(ti, tj)
                for ti, tj in zip(cfg.CHRONO_RANGES[:-1], cfg.CHRONO_RANGES[1:])])

            self.rnn.bias_hh[cfg.MEM_SIZE:-cfg.MEM_SIZE] = chrono_bias.mul_(cfg.STEPS_PER_SECOND).log_()

    def forward(self, x: Tensor, x_map: Tensor, memp: Tensor, memv: Tensor) -> 'tuple[Tensor, Tensor]':

        # Encode spatial state information
        x_map = self.mapenc(x_map).repeat_interleave(self.n_bots, dim=0, output_size=self.n_all_bots)

        # Consolidate local observations with global state information
        x = torch.cat((x, x_map, memp), dim=-1)
        x = self.activ(self.fcin(x))

        # Update temporal variables
        memv = self.rnn(x, memv)

        # Map to value distribution parameters
        x = self.mlpout(memv)

        return x, memv


# ------------------------------------------------------------------------------
# MARK: ActorCritic

class ActorCritic(ActorCriticTemplate):
    """Wrapper around the visual encoder, policy, and valuator networks."""

    def __init__(self, n_envs: int, n_bots: int, com_mode: int):
        super().__init__()

        self.n_all_bots = n_envs * n_bots

        self.visencoder = VisEncoder()
        self.policy = Policy(com_mode, n_envs, n_bots)
        self.valuator = Valuator(n_envs, n_bots)

        for param in self.visencoder.parameters():
            param.requires_grad = False

        self.policy.random_init()
        self.valuator.random_init()

    def init_mem(self, batch_size: int = 1) -> 'tuple[Tensor, Tensor]':
        qkv = self.policy.qkv_initial.detach().expand(self.n_all_bots, -1).clone()
        memp = self.policy.mem_initial.detach().expand(self.n_all_bots, -1).clone()
        memv = self.valuator.mem_initial.detach().expand(self.n_all_bots, -1).clone()

        return qkv, memp, memv

    def reset_mem(self, mem: 'tuple[Tensor, Tensor]', nonreset_mask: Tensor) -> 'tuple[Tensor, Tensor]':
        qkv, memp, memv = mem

        qkv = torch.lerp(self.policy.qkv_initial, qkv, nonreset_mask)
        memp = torch.lerp(self.policy.mem_initial, memp, nonreset_mask)
        memv = torch.lerp(self.valuator.mem_initial, memv, nonreset_mask)

        return qkv, memp, memv

    def get_distr(self, args: Tensor, from_raw: bool) -> Categorical:
        return (Categorical.from_raw if from_raw else Categorical)(self.policy.act_values, args)

    def unwrap_sample(self, sample: 'tuple[Tensor, Tensor]') -> 'tuple[Tensor, Tensor]':
        return sample[0], None

    def act(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        sample: bool
    ) -> 'tuple[tuple[Tensor, ...], tuple[Tensor, ...]]':

        x_img, x_vec, _ = obs
        qkv, memp, memv = mem

        with torch.no_grad():
            x_img = self.visencoder(x_img).flatten(1)
            x = torch.cat((x_img, x_vec[:, :cfg.OBS_VEC_SIZE]), dim=-1)

            a, _, _, qkv, memp = self.policy(x, qkv, memp)

        act = self.get_distr(a, from_raw=True)

        if sample:
            a = act.sample()

        else:
            a = act.mode, a.log_probs.argmax(dim=-1).unsqueeze(-1)

        return a, (qkv, memp, memv)

    def collect(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        encode: bool
    ) -> 'tuple[Tensor, Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]':

        xp, xv, x_map = obs
        qkv, memp, memv = mem

        with torch.no_grad():
            if encode:
                x_img = self.visencoder(xp).flatten(1)
                xp = torch.cat((x_img, xv[:, :cfg.OBS_VEC_SIZE]), dim=-1)

            a, _, _, qkv, memp = self.policy(xp, qkv, memp)
            v, memv = self.valuator(xv, x_map, GradScaler.apply(memp, 0.01), memv)

        return a, v, (xp, xv, x_map), (qkv, memp, memv)

    def forward(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        detach: bool = False
    ) -> 'tuple[Categorical, FixedVarNormal, tuple[Categorical, FixedVarNormal], tuple[Tensor, ...]]':

        xp, xv, x_map = obs
        qkv, memp, memv = mem

        a, bs, bg, qkv, memp = self.policy(xp, qkv, memp)
        v, memv = self.valuator(xv, x_map, memp.detach() if detach else GradScaler.apply(memp, 0.01), memv)

        act = self.get_distr(a, from_raw=True)
        val = FixedVarNormal(v, skip_log_shift=True)
        bs = Categorical(bs)
        bg = FixedVarNormal(bg, skip_log_shift=True)

        return act, val, (bs, bg), (qkv, memp, memv)
