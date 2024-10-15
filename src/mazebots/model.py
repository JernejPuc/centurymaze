"""MazeBots AI"""

import torch
from torch import nn, Tensor
from torch.nn import Module
from torch.nn.functional import scaled_dot_product_attention, softplus

from discit.distr import FixedVarNormal, Categorical
from discit.marl import MultiActorCritic as ActorCriticTemplate

import config as cfg


# ------------------------------------------------------------------------------
# MARK: ResidualMLP

class ResidualMLP(Module):
    def __init__(self, dim_in: int, dim_mid: int, dim_out: int = None):
        super().__init__()

        if dim_out is None:
            dim_out = dim_mid

        self.activ = nn.Tanh()
        self.fcin = nn.Linear(dim_in, dim_mid)
        self.fcout = nn.Linear(dim_in + dim_mid, dim_out)

    def random_init(self, gain: float = 1.):
        nn.init.orthogonal_(self.fcin.weight)
        nn.init.zeros_(self.fcin.bias)
        nn.init.orthogonal_(self.fcout.weight, gain=gain)
        nn.init.zeros_(self.fcout.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.fcout(torch.cat((x, self.activ(self.fcin(x))), dim=-1))


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

        # 4x2x80 -> 1x1x240 -> 1x1x2E
        self.conv4_dw = nn.Conv2d(80, 240, (2, 4), 1, groups=80, bias=False)
        self.conv4_cw = nn.Conv2d(240, 2*cfg.VIS_ENC_SIZE, 1, 1)

    def random_init(self):
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
        return self.conv4_cw(self.conv4_dw(x))


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

    def __init__(self, sample: bool = False):
        super().__init__()

        self.sample = sample

        self.enc = VisEncoder()
        self.dec = VisDecoder()

        # E -> E + E -> 9 + 2
        self.mlp = ResidualMLP(cfg.VIS_ENC_SIZE, cfg.VIS_ENC_SIZE, sum(cfg.AUX_VAL_SPLIT))

        self.enc.random_init()
        self.mlp.random_init(gain=0.001)

    def forward(self, x: Tensor) -> 'tuple[Tensor, ...]':
        x_mean, x_scale = self.enc(x).chunk(2, dim=1)

        x_mean = x_mean.tanh()
        x_scale = softplus(x_scale)

        if self.sample:
            x = x_mean + torch.randn_like(x_scale) * x_scale

        else:
            x = x_mean

        b_obj, b_loc = self.mlp(x.flatten(1)).split(cfg.AUX_VAL_SPLIT, dim=-1)

        return self.dec(x), x_mean, x_scale, b_obj, b_loc


# ------------------------------------------------------------------------------
# MARK: RandomActorCritic

class RandomActorCritic(ActorCriticTemplate):
    """Static zero output, essentially random policy."""

    def __init__(self, n_envs: int, n_bots: int, com_mode: int, sample: bool = True):
        super().__init__()

        self.n_all_bots = n_envs * n_bots

        self.act_values = nn.Parameter(torch.tensor(cfg.ACT_VALUES), requires_grad=False)
        self.zero = nn.Parameter(torch.zeros(1, 1), requires_grad=False)

        self.visnet = VisNet(sample)

    def get_distr(self, args: 'Tensor | tuple[Tensor, ...]', from_raw: bool = False) -> Categorical:
        return Categorical.from_raw(self.act_values, args) if from_raw else Categorical(self.act_values, args[0])

    def collect(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        sample: 'tuple[Tensor, ...] | None'
    ) -> 'tuple[dict[str, Tensor | tuple[Tensor, ...]], None, tuple]':

        act = self.zero.expand(self.n_all_bots, cfg.ACT_SIZE)
        val = self.zero.expand(self.n_all_bots, len(cfg.DISCOUNTS))
        adv = self.zero.expand(self.n_all_bots, -1)

        act = self.get_distr(act, from_raw=True)

        if sample is None:
            sample = act.sample()

        d = {
            'act': sample,
            'args': act.args,
            'val': val,
            'adv': adv,
            'mem': mem,
            'obs': obs}

        return d, None, mem

    def forward(self, obs: 'tuple[Tensor, ...]', mem: 'tuple[Tensor, ...]', detach: bool = False):
        raise NotImplementedError


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
        x = self.conv4(x).tanh()

        return x.flatten(1)


# ------------------------------------------------------------------------------
# MARK: TopMAC

class TopMAC(Module):

    def __init__(self, n_envs: int, n_bots: int):
        super().__init__()

        self.n_envs = n_envs
        self.n_bots = n_bots

        self.q = nn.Parameter(torch.empty(1, cfg.N_TOPICS, cfg.K_DIM).uniform_(-1., 1.))
        self.k = nn.Parameter(torch.empty(1, cfg.N_TOPICS, cfg.K_DIM).uniform_(-1., 1.))
        self.s = nn.Parameter(torch.tensor(3., dtype=torch.float32))

    def forward(self, qkv: Tensor, mask: Tensor, n_envs: int = None) -> Tensor:
        if n_envs is None:
            n_envs = self.n_envs

        qkv = qkv.reshape(n_envs, self.n_bots, -1)
        q, k, v = qkv.split(cfg.QKV_SPLIT, dim=-1)

        # Apply trainable scale
        s = self.s.abs()

        top_q = (self.q * s).expand(n_envs, -1, -1)
        top_k = (self.k * s).expand(n_envs, -1, -1)

        # N -> Ex1xB
        mask = torch.where(mask.bool(), 0., -1e+6).reshape(n_envs, 1, self.n_bots)

        # ExTxQ, ExBxK, ExBxV -> ExTxV
        top_v = scaled_dot_product_attention(top_q, k, v, mask)

        # ExBxQ, ExTxK, ExTxV -> ExBxV
        x = scaled_dot_product_attention(q, top_k, top_v)

        return x.reshape(-1, cfg.V_DIM)


# ------------------------------------------------------------------------------
# MARK: TarMAC

class TarMAC:
    def __init__(self, n_envs: int, n_bots: int):
        self.n_envs = n_envs
        self.n_bots = n_bots

    def __call__(self, qkv: Tensor, mask: Tensor, n_envs: int = None) -> Tensor:
        if n_envs is None:
            n_envs = self.n_envs

        qkv = qkv.reshape(n_envs, self.n_bots, -1)
        q, k, v = qkv.split(cfg.QKV_SPLIT, dim=-1)

        # N -> Ex1xB
        mask = torch.where(mask.bool(), 0., -1e+6).reshape(n_envs, 1, self.n_bots)

        # ExBxQ, ExBxK, ExBxV -> ExBxV
        x = scaled_dot_product_attention(q, k, v, mask)

        return x.reshape(-1, cfg.V_DIM)


# ------------------------------------------------------------------------------
# MARK: NoMAC

class NoMAC(Module):
    def __init__(self, n_envs: int, n_bots: int):
        super().__init__()

        self.zeros = nn.Parameter(torch.zeros(1, cfg.V_DIM), requires_grad=False)

    def forward(self, qkv: Tensor, mask: Tensor, n_envs: int = None) -> Tensor:
        return self.zeros.expand(len(qkv), -1)


# ------------------------------------------------------------------------------
# MARK: Policy

class Policy(Module):
    def __init__(self, com_mode: int, n_envs: int, n_bots: int):
        super().__init__()

        self.com_mode = com_mode
        self.n_envs = n_envs
        self.n_bots = n_bots
        self.n_all_bots = n_envs * n_bots

        self.qkv_initial = nn.Parameter(torch.zeros(1, sum(cfg.QKV_SPLIT)), requires_grad=False)

        act_values = torch.tensor(cfg.ACT_VALUES)
        self.act_values = nn.Parameter(act_values, requires_grad=False)
        self.out_sizes = (len(cfg.ACT_VALUES), *cfg.AUX_VAL_SPLIT, sum(cfg.QKV_SPLIT))

        ipt_size = cfg.OBS_VEC_SIZE + cfg.VIS_ENC_SIZE + cfg.V_DIM

        # 128 -> 64
        self.com = (NoMAC, TarMAC, TopMAC)[com_mode](n_envs, n_bots)

        # 224 + 28 + 64 -> 316 + 320 -> 320
        self.mlpin = ResidualMLP(ipt_size, cfg.POL_ENC_SIZE)

        # 320 -> 256
        self.rnn = nn.GRUCell(cfg.POL_ENC_SIZE, cfg.MEM_SIZE)

        # 256 -> 256 + 256 -> 10 + 11 + 128
        self.mlpout = ResidualMLP(cfg.MEM_SIZE, cfg.PRE_OUT_SIZE, sum(self.out_sizes))

        # https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
        self.mem_initial = nn.Parameter(torch.zeros(1, cfg.MEM_SIZE).uniform_(-1., 1.))

    def random_init(self):
        self.mlpin.random_init()
        self.mlpout.random_init(gain=0.001)

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

    def forward(
        self, x: Tensor,
        qkv: Tensor,
        mem: Tensor,
        forced_com_mask: Tensor = None,
        n_envs: int = None
    ) -> 'tuple[Tensor, ...]':

        goal_found_mask = x[:, cfg.GOAL_FOUND_SLICE]
        speaking_mask = x[:, cfg.LED_ON_IDX]

        if forced_com_mask is not None:
            speaking_mask = speaking_mask.maximum(forced_com_mask)

        # Process messages
        msg = self.com(qkv, speaking_mask, n_envs)

        # Process observations
        x = self.mlpin(torch.cat((x, msg), dim=-1))

        # Update memory/state and output
        mem = self.rnn(x, mem)

        # Split output into actions, beliefs, and message components
        a, bo, bg, qkv = self.mlpout(mem).split(self.out_sizes, dim=-1)

        # Detach grads. for unfound goals
        bg = torch.lerp(bg.detach(), bg, goal_found_mask)

        return a, bo, bg, qkv, mem, msg.detach()


# ------------------------------------------------------------------------------
# MARK: EnvAttention

class EnvAttention(Module):
    def __init__(self, n_envs: int, n_bots: int, n_heads: int):
        super().__init__()

        self.n_envs = n_envs
        self.n_bots = n_bots
        self.n_heads = n_heads

        self.kv_dim = 2 * cfg.K_DIM

        self.q = nn.Parameter(torch.empty(1, n_heads, 1, cfg.K_DIM).uniform_(-1., 1.))
        self.s = nn.Parameter(torch.tensor(3., dtype=torch.float32))

    def forward(self, kv: Tensor) -> Tensor:

        # Nx(H*2*K) -> ExBxHx(2*K) -> ExHxBx(2*K) -> 2x ExHxBxK
        kv = kv.reshape(-1, self.n_bots, self.n_heads, self.kv_dim)
        kv = kv.transpose(1, 2)
        k, v = kv.contiguous().chunk(2, dim=-1)

        # Apply trainable scale
        s = self.s.abs()

        env_q = (self.q * s).expand(len(k), -1, -1, -1)

        # ExHx1xQ, 2x ExHxBxK -> ExHx1xK -> Ex(H*K)
        x = scaled_dot_product_attention(env_q, k, v)

        return x.flatten(1)


# ------------------------------------------------------------------------------
# MARK: SelfAttention

class SelfAttention:
    def __init__(self, n_envs: int, n_bots: int, n_heads: int):
        self.n_envs = n_envs
        self.n_bots = n_bots
        self.n_all_bots = n_envs * n_bots
        self.n_heads = n_heads

        self.qkv_dim = 3 * cfg.K_DIM
        self.out_dim = n_heads * cfg.K_DIM

    def __call__(self, qkv: Tensor) -> Tensor:

        # Nx(H*3*K) -> ExBxHx(3*K) -> ExHxBx(3*K) -> 3x ExHxBxK
        qkv = qkv.reshape(-1, self.n_bots, self.n_heads, self.qkv_dim)
        qkv = qkv.transpose(1, 2)
        q, k, v = qkv.contiguous().chunk(3, dim=-1)

        # 3x ExHxBxK -> ExHxBxK
        x = scaled_dot_product_attention(q, k, v)

        # ExHxBxK -> ExBxHxK -> Nx(H*K)
        x = x.transpose(1, 2)
        x = x.reshape(-1, self.out_dim)

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

        self.n_envs = n_envs
        self.n_bots = n_bots
        self.n_all_bots = n_envs * n_bots

        self.vikv_split = (len(cfg.DISCOUNTS) - 1, cfg.N_HEADS * 2 * cfg.K_DIM)
        self.aiqkv_split = (1, cfg.N_HEADS * 3 * cfg.K_DIM)

        # 20x20x15 -> 116
        self.mapenc = MapEncoder()

        # 108 + 224 + 64 + 116 -> 512 + 512 -> 512
        self.mlpin = ResidualMLP(cfg.STATE_VEC_SIZE + cfg.VIS_ENC_SIZE + cfg.V_DIM + cfg.MAP_ENC_SIZE, cfg.VAL_ENC_SIZE)

        # 512 -> 256
        self.rnn = nn.GRUCell(cfg.VAL_ENC_SIZE, cfg.MEM_SIZE)

        # 256 -> 256 + 256 -> 2 + 256
        self.mlp_vi = ResidualMLP(cfg.MEM_SIZE, cfg.PRE_OUT_SIZE, sum(self.vikv_split))

        # 51 + 116 + 128 -> 295 + 249 -> 1
        self.mlp_vj = ResidualMLP(cfg.STATE_ENV_SIZE + cfg.MAP_ENC_SIZE + cfg.ATN_ENC_SIZE, cfg.PRE_OUT_SIZE - 7, 1)

        # 256 + 5 -> 261 + 251 -> 1 + 384
        self.mlp_ai = ResidualMLP(cfg.MEM_SIZE + cfg.ACT_SIZE, cfg.PRE_OUT_SIZE - cfg.ACT_SIZE, sum(self.aiqkv_split))

        # 295 + 261 + 1 + 128 -> 685 + 243 -> 1
        self.mlp_li = ResidualMLP(295 + cfg.MEM_SIZE + cfg.ACT_SIZE + 1 + cfg.ATN_ENC_SIZE, cfg.PRE_OUT_SIZE - 13, 1)

        # 256 -> 4x 2x 32, 4x 32 -> 128
        self.atten_j = EnvAttention(n_envs, n_bots, cfg.N_HEADS)

        # 384 -> 4x 3x 32 -> 128
        self.atten_i = SelfAttention(n_envs, n_bots, cfg.N_HEADS)

        self.mem_initial = nn.Parameter(torch.zeros(1, cfg.MEM_SIZE).uniform_(-1., 1.))

    def random_init(self):
        self.mlpin.random_init()
        self.mlp_vi.random_init(gain=0.1)
        self.mlp_vj.random_init(gain=0.001)
        self.mlp_ai.random_init(gain=0.1)
        self.mlp_li.random_init(gain=0.001)

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

    def forward(self, x: Tensor, x_map: Tensor, memv: Tensor, act: Tensor) -> 'tuple[Tensor, ...]':
        x_env = x[::self.n_bots, cfg.STATE_ENV_SLICE]

        # Encode spatial state information
        x_map = self.mapenc(x_map)

        # Consolidate local observations with global state information
        x = torch.cat((x, x_map.repeat_interleave(self.n_bots, dim=0, output_size=len(memv))), dim=-1)
        x = self.mlpin(x)

        # Update temporal variables
        memv = self.rnn(x, memv)

        # Map to value distribution parameters
        # Indiv. branch (info. from other agents only implicit)
        vi, kv = self.mlp_vi(memv).split(self.vikv_split, dim=-1)

        mema = torch.cat((memv, act), dim=-1)
        ai, qkv = self.mlp_ai(mema).split(self.aiqkv_split, dim=-1)

        # Joint branch (info. from all agents collapsed into a single vector)
        x = torch.cat((x_env, x_map, self.atten_j(kv)), dim=-1)
        vj = self.mlp_vj(x)

        # Mixing branch (info. from other agents adjusts indiv. estimates)
        x = x.repeat_interleave(self.n_bots, dim=0, output_size=len(memv))
        x = torch.cat((x, mema, ai.detach(), self.atten_i(qkv)), dim=-1)

        li = self.mlp_li(x)
        ai = ai * li

        return vj, vi, ai, memv


# ------------------------------------------------------------------------------
# MARK: ActorCritic

class ActorCritic(ActorCriticTemplate):
    """Wrapper around the visual encoder, policy, and valuator networks."""

    def __init__(self, n_envs: int, n_bots: int, com_mode: int):
        super().__init__()

        self.n_bots = n_bots
        self.n_all_bots = n_envs * n_bots

        self.visencoder = VisEncoder()
        self.policy = Policy(com_mode, n_envs, n_bots)
        self.valuator = Valuator(n_envs, n_bots)

        for param in self.visencoder.parameters():
            param.requires_grad = False

        self.policy.random_init()
        self.valuator.random_init()

    def init_mem(self, n_envs: int = None, n_all_bots: int = None) -> 'tuple[Tensor, ...]':
        qkv = self.policy.qkv_initial.detach().expand(self.n_all_bots, -1).clone()
        memp = self.policy.mem_initial.detach().expand(self.n_all_bots, -1).clone()
        memv = self.valuator.mem_initial.detach().expand(self.n_all_bots, -1).clone()

        return qkv, memp, memv

    def reset_mem(self, mem: 'tuple[Tensor, ...]', nonreset_mask: Tensor) -> 'tuple[Tensor, ...]':
        qkv, memp, memv = mem

        qkv = torch.lerp(self.policy.qkv_initial, qkv, nonreset_mask)
        memp = torch.lerp(self.policy.mem_initial, memp, nonreset_mask)
        memv = torch.lerp(self.valuator.mem_initial, memv, nonreset_mask)

        return qkv, memp, memv

    def get_distr(self, args: 'Tensor | tuple[Tensor, ...]', from_raw: bool = False) -> Categorical:
        if from_raw:
            return Categorical.from_raw(self.policy.act_values, args)

        else:
            return Categorical(self.policy.act_values, args[0])

    def unwrap_sample(self, act: 'tuple[Tensor, Tensor]', belief: 'tuple[Tensor, Tensor]') -> 'tuple[Tensor, Tensor]':
        return act[0], torch.cat((belief[0][:, 1:], belief[1]), dim=-1)

    def act(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        sample: bool
    ) -> 'tuple[tuple[Tensor, ...], tuple[Tensor, ...]]':

        x_img, x_vec, _ = obs
        qkv, memp, memv = mem

        x_img = self.visencoder(x_img)[:, :cfg.VIS_ENC_SIZE, 0, 0].tanh()
        x = torch.cat((x_vec[:, :cfg.OBS_VEC_SIZE], x_img), dim=-1)

        a, bo, bg, qkv, memp, _ = self.policy(x, qkv, memp)

        act = self.get_distr(a, from_raw=True)

        a = act.sample()[0] if sample else act.mode
        b = torch.cat((bo[:, 1:], bg), dim=-1)

        return a, b, (qkv, memp, memv)

    def collect(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        sample: 'tuple[Tensor, ...] | None'
    ) -> 'tuple[dict[str, Tensor | tuple[Tensor, ...]], Tensor, tuple[Tensor, ...]]':

        qkv, memp, memv = mem

        if sample is None:
            x_img, x_vec, x_map = obs

            x_img = self.visencoder(x_img)[:, :cfg.VIS_ENC_SIZE, 0, 0].tanh()
            xp = torch.cat((x_vec[:, :cfg.OBS_VEC_SIZE], x_img), dim=-1)

            a, bo, bg, qkv, memp, x_msg = self.policy(xp, qkv, memp)
            xv = torch.cat((x_vec, x_img, x_msg), dim=-1)

        else:
            xp, xv, x_map = obs

            a, bo, bg, qkv, memp, _ = self.policy(xp, qkv, memp)

        act = self.get_distr(a, from_raw=True)

        if sample is None:
            sample = act.sample()

        vj, vi, ad, memv = self.valuator(xv, x_map, memv, sample[0])

        v = torch.cat((
            vj.repeat_interleave(self.n_bots, dim=0, output_size=self.n_all_bots),
            vi), dim=-1)

        d = {
            'act': sample,
            'args': act.args,
            'val': v,
            'adv': ad,
            'mem': mem,
            'obs': (xp, xv, x_map)}

        return d, (bo, bg), (qkv, memp, memv)

    def forward(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        sample: 'tuple[Tensor, ...]'
    ) -> 'dict[str, Categorical | tuple[FixedVarNormal, ...] | tuple[Tensor, ...]]':

        xp, xv, x_map = obs
        qkv, memp, memv = mem

        n_envs = len(x_map)

        a, bo, bg, qkv, memp, _ = self.policy(xp, qkv, memp, n_envs=n_envs)
        vj, vi, ai, memv = self.valuator(xv, x_map, memv, sample[0])

        aj = ai.reshape(n_envs, self.policy.n_bots, -1).sum(1)

        act = self.get_distr(a, from_raw=True)
        valj = FixedVarNormal(vj, skip_log_shift=True)
        vali = FixedVarNormal(vi, skip_log_shift=True)
        advj = FixedVarNormal(aj, skip_log_shift=True)
        bo = Categorical(None, bo)
        bg = FixedVarNormal(bg, skip_log_shift=True)

        return {
            'act': act,
            'valj': valj,
            'vali': vali,
            'advj': advj,
            'aux': (bo, bg),
            'mem': (qkv, memp, memv)}
