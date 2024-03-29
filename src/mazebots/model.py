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


class SpaEncoder(Module):
    """Conv. encoder for processing global state information in spatial form."""

    ENC_SIZE = 128

    def __init__(self):
        super().__init__()

        self.activ = nn.CELU(0.1)

        # 20x20x16 -> 20x20x16
        self.conv1 = nn.Conv2d(cfg.STATE_SPA_CHANNELS, 16, 3, padding=1)
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
        self.conv4 = nn.Conv2d(48, self.ENC_SIZE, 5)
        self.norm4 = nn.GroupNorm(1, self.ENC_SIZE)

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


class Policy(Module):
    """
    Bicameral policy, where the motor and speaker subnetworks interact through
    detached memory vectors to limit interference during optimisation.
    """

    MOT_MEM_SIZE = 192
    COM_MEM_SIZE = 64
    MEM_SIZE = MOT_MEM_SIZE + COM_MEM_SIZE

    SHORT_MEM_HALF_LIFE = 3 * cfg.STEPS_PER_SECOND
    LONG_MEM_HALF_LIFE = 60 * cfg.STEPS_PER_SECOND
    MAX_MEM_HORIZON = 300 * cfg.STEPS_PER_SECOND

    FEAT_CONDITIONAL = 3
    FEAT_TRAINED = 2
    FEAT_EXTERNAL = 1
    FEAT_DISABLED = 0

    def __init__(
        self,
        com_state: int,
        guide_state: int,
        com_out_size: int,
        com_bias: bool,
        n_agents_per_env: int,
        n_envs: int
    ):
        super().__init__()

        self.com_state = com_state
        self.guide_state = guide_state
        self.com_bias = com_bias

        self.n_bots = n_agents_per_env
        self.n_envs = n_envs
        self.n_all_bots = n_agents_per_env * n_envs

        self.input_sizes = (VisEncoder.ENC_SIZE + cfg.OBS_VEC_SIZE, cfg.AUX_VAL_SIZE, cfg.META_VAL_SIZE)
        self.role_slice = slice(cfg.OBS_ROLE_IDX + VisEncoder.ENC_SIZE, cfg.OBS_ROLE_IDX + VisEncoder.ENC_SIZE + 1)

        trq_values = torch.tensor(cfg.ACT_DOF_MODES_BASE)
        clr_values = torch.tensor(cfg.RCVR_CLR_CLASSES)
        self.trq_values = nn.Parameter(trq_values, requires_grad=False)
        self.clr_values = nn.Parameter(clr_values, requires_grad=False)
        self.out_values = (self.trq_values, self.clr_values)
        self.output_sizes = (len(trq_values), len(clr_values))

        trql_mask = torch.tensor((cfg.ACT_DOF_MOTION_MASK,), dtype=torch.float32)
        self.trql_mask = nn.Parameter(trql_mask, requires_grad=False)
        self.null_bias = nn.Parameter(torch.tensor(-20.), requires_grad=False)

        coml_mask = torch.ones((1, cfg.N_RCVR_CLR_CLASSES))
        coml_mask[0, :com_out_size] = 0.
        self.coml_mask = nn.Parameter(coml_mask, requires_grad=False)

        coml_silent = torch.full((1, len(clr_values)), -20.)
        coml_silent[:, 0] = 0.
        self.coml_initial = nn.Parameter(coml_silent, requires_grad=False)

        self.activ = nn.Tanh()

        # E + 94 + 46|0 -> 320
        self.fcin = nn.Linear(
            self.input_sizes[0]+(cfg.AUX_VAL_SIZE if guide_state in (self.FEAT_EXTERNAL, self.FEAT_CONDITIONAL) else 0),
            320)

        # 320 + Mc|Mm -> Mm|Mc
        self.rnn_m = nn.GRUCell(320 + self.COM_MEM_SIZE, self.MOT_MEM_SIZE)
        self.rnn_c = nn.GRUCell(320 + self.MOT_MEM_SIZE, self.COM_MEM_SIZE)

        # M -> A
        self.fcaux = nn.Linear(self.MEM_SIZE, cfg.AUX_VAL_SIZE)

        # Mm|Mc -> 5|11
        self.fcout_m = nn.Linear(self.MOT_MEM_SIZE, self.output_sizes[0])
        self.fcout_c = nn.Linear(self.COM_MEM_SIZE, self.output_sizes[1])

        # https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
        self.mem_initial = nn.Parameter(torch.zeros(1, self.MEM_SIZE).uniform_(-1., 1.))
        self.mem_sizes = (self.MOT_MEM_SIZE, self.COM_MEM_SIZE)

    def random_init(self):
        nn.init.orthogonal_(self.fcin.weight)
        nn.init.zeros_(self.fcin.bias)
        nn.init.orthogonal_(self.fcout_m.weight, gain=0.001)
        nn.init.zeros_(self.fcout_m.bias)
        nn.init.orthogonal_(self.fcout_c.weight, gain=0.001)
        nn.init.zeros_(self.fcout_c.bias)
        nn.init.orthogonal_(self.fcaux.weight, gain=0.001)
        nn.init.zeros_(self.fcaux.bias)

        # Init. bias twd. neutral com. (short-range, unassociated with any obj.)
        if self.com_bias:
            coml_bias = torch.tensor([0.1/10., 0.9] + 9*[0.1/10.]).log_()
            coml_bias -= coml_bias.max()

            with torch.no_grad():
                self.fcout_c.bias.copy_(coml_bias)

        for rnn, mem_size in zip((self.rnn_m, self.rnn_c), self.mem_sizes):
            for weight in (rnn.weight_ih, rnn.weight_hh):
                nn.init.orthogonal_(weight[:mem_size])
                nn.init.orthogonal_(weight[mem_size:-mem_size])
                nn.init.orthogonal_(weight[-mem_size:])

            nn.init.zeros_(rnn.bias_ih)
            nn.init.zeros_(rnn.bias_hh)

            # Chrono init. wrt. discount factors and max. episode length
            with torch.no_grad():
                gamma_short = 0.5 ** (1. / self.SHORT_MEM_HALF_LIFE)
                gamma_long = 0.5 ** (1. / self.LONG_MEM_HALF_LIFE)

                probs_short = torch.tensor([gamma_short**i for i in range(1, self.MAX_MEM_HORIZON)])
                probs_short /= probs_short.sum()

                probs_long = torch.tensor([gamma_long**i for i in range(1, self.MAX_MEM_HORIZON)])
                probs_long /= probs_long.sum()

                probs = 0.5 * probs_short + 0.5 * probs_long
                horizons = 1. + probs.multinomial(mem_size, replacement=True)

                rnn.bias_hh[mem_size:-mem_size] = horizons.log_()

    def forward(self, x: Tensor, memp: Tensor, detach: bool) -> 'tuple[Tensor, ...]':
        x, x_aux, heur_com_idx = x.split(self.input_sizes, dim=-1)
        mem_m, mem_c = memp.split(self.mem_sizes, dim=-1)
        speaker_mask = x[:, self.role_slice]

        # Process observations
        if self.guide_state == self.FEAT_CONDITIONAL:
            obj_in_frame, obj_dirprox, obj_found = x_aux.split(cfg.AUX_VAL_SPLIT, dim=-1)
            obj_dirprox = obj_dirprox.reshape(-1, cfg.N_DIM_POS, cfg.N_OBJ_COLOURS) * obj_found.unsqueeze(1)
            obj_dirprox = obj_dirprox.flatten(1)

            x = torch.cat((x, obj_in_frame, obj_dirprox, obj_found), dim=-1)

        elif self.guide_state == self.FEAT_EXTERNAL:
            x = torch.cat((x, x_aux), dim=-1)

        x = self.activ(self.fcin(x))

        # Update com. memory/state and output
        if self.com_state >= self.FEAT_TRAINED:
            mem_c = self.rnn_c(torch.cat((x, mem_m.detach()), dim=-1), mem_c)
            clr_logits = self.fcout_c(mem_c)

        # Override - heuristic
        # Obj. colour if seen and near, white if obstructed, black otherwise
        elif self.com_state == self.FEAT_EXTERNAL:
            clr_logits = -20. * torch.logical_not(one_hot(heur_com_idx.squeeze(-1).long(), cfg.N_RCVR_CLR_CLASSES))

        # Override - silent
        else:
            clr_logits = self.coml_initial.expand(len(x), -1)

        # Update mot. memory/state and output
        mem_m = self.rnn_m(torch.cat((x, mem_c.detach()), dim=-1), mem_m)
        trq_logits = self.fcout_m(mem_m)

        # Freeze translation and mask colours for speakers when com. training is emphasised
        if self.com_state == self.FEAT_CONDITIONAL:
            trq_logits = torch.lerp(trq_logits, self.null_bias, self.trql_mask * speaker_mask)
            clr_logits = torch.lerp(clr_logits, self.null_bias, self.coml_mask * speaker_mask)

        # Detach grads. from non-speakers
        elif self.com_state == self.FEAT_TRAINED:
            clr_logits = torch.lerp(clr_logits.detach(), clr_logits, speaker_mask)

        memp = torch.cat((mem_m, mem_c), dim=-1)
        x = torch.cat((trq_logits, clr_logits), dim=-1)

        # Auxiliary obj. estimates
        if self.guide_state == self.FEAT_TRAINED:
            a = self.fcaux(memp.detach() if detach else GradScaler.apply(memp, 0.01))
            a_01, a_dir = a.split((cfg.AUX_VAL_SIZE - 2*cfg.N_OBJ_COLOURS, 2*cfg.N_OBJ_COLOURS), dim=-1)

            a_01 = a_01.sigmoid()
            a_in_frame, a_prox, a_found = a_01.split((cfg.N_OBJ_COLOURS+1,) + (cfg.N_OBJ_COLOURS,)*2, dim=-1)

            a_dir = a_dir.reshape(len(x), 2, -1)
            a_dir = a_dir / torch.linalg.norm(a_dir, dim=1, keepdim=True)

            # Detach error grads. for unfound objs.
            obj_found = x_aux[:, -cfg.AUX_VAL_SPLIT[-1]:]
            any_found = obj_found.reshape(self.n_envs, self.n_bots, -1).any(1)
            any_found = any_found.repeat_interleave(self.n_bots, dim=0, output_size=self.n_all_bots)

            a_dirprox = torch.cat((a_dir, a_prox.unsqueeze(1)), dim=1) * any_found.unsqueeze(1)
            a_dirprox = a_dirprox.flatten(1)

            a = torch.cat((a_in_frame, a_dirprox, a_found), dim=-1)

        else:
            a = self.fcaux(memp.detach())

        # Detach grads. from non-speakers
        if self.com_state == self.FEAT_CONDITIONAL:
            x = torch.lerp(x.detach(), x, speaker_mask)
            a = torch.lerp(a.detach(), a, speaker_mask)
            memp = torch.lerp(memp.detach(), memp, speaker_mask)

        return x, a, memp


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


class Valuator(Module):
    """
    Critic network attending to a variable number of agents per environment.

    Besides the agents' memory vectors, the valuator has access to some other
    variables that are used in reward evaluation and thus, while unknowable
    to the agents, are critical to reliably anticipate future rewards.
    """

    MEM_SIZE = 256
    N_MEM_HALF_LIVES = 2

    SHORT_MEM_HALF_LIFE = Policy.SHORT_MEM_HALF_LIFE
    LONG_MEM_HALF_LIFE = MAX_MEM_HORIZON = Policy.MAX_MEM_HORIZON

    def __init__(self, com_state: int, n_agents_per_env: int, n_envs: int):
        super().__init__()

        self.detach_cond = com_state == Policy.FEAT_CONDITIONAL
        self.role_slice = slice(cfg.OBS_ROLE_IDX, cfg.OBS_ROLE_IDX + 1)

        self.n_bots = n_agents_per_env
        self.n_all_bots = n_agents_per_env * n_envs

        self.activ = nn.Tanh()

        # 20x20x16 -> 128
        self.spaencoder = SpaEncoder()

        # 122 + 128 + Mp -> 512
        self.fcin = nn.Linear(cfg.STATE_VEC_SIZE + SpaEncoder.ENC_SIZE + Policy.MEM_SIZE, 512)

        # 512 -> M
        self.rnn = nn.GRUCell(512, self.MEM_SIZE)

        # M -> 256 -> V
        self.mlpout = nn.Sequential(
            nn.Linear(self.MEM_SIZE, 256),
            self.activ,
            nn.Linear(256, self.N_MEM_HALF_LIVES))

        self.mem_initial = nn.Parameter(torch.zeros(1, self.MEM_SIZE).uniform_(-1., 1.))

    def random_init(self):
        nn.init.orthogonal_(self.fcin.weight)
        nn.init.zeros_(self.fcin.bias)
        nn.init.orthogonal_(self.mlpout[0].weight)
        nn.init.zeros_(self.mlpout[0].bias)
        nn.init.orthogonal_(self.mlpout[2].weight, gain=0.001)
        nn.init.zeros_(self.mlpout[2].bias)

        for weight in (self.rnn.weight_ih, self.rnn.weight_hh):
            nn.init.orthogonal_(weight[:self.MEM_SIZE])
            nn.init.orthogonal_(weight[self.MEM_SIZE:-self.MEM_SIZE])
            nn.init.orthogonal_(weight[-self.MEM_SIZE:])

        nn.init.zeros_(self.rnn.bias_ih)
        nn.init.zeros_(self.rnn.bias_hh)

        # Chrono init. wrt. discount factors and max. episode length
        with torch.no_grad():
            gamma_short = 0.5 ** (1. / self.SHORT_MEM_HALF_LIFE)
            gamma_long = 0.5 ** (1. / self.LONG_MEM_HALF_LIFE)

            probs_short = torch.tensor([gamma_short**i for i in range(1, self.MAX_MEM_HORIZON)])
            probs_short /= probs_short.sum()

            probs_long = torch.tensor([gamma_long**i for i in range(1, self.MAX_MEM_HORIZON)])
            probs_long /= probs_long.sum()

            probs = 0.5 * probs_short + 0.5 * probs_long
            horizons = 1. + probs.multinomial(self.MEM_SIZE, replacement=True)

            self.rnn.bias_hh[self.MEM_SIZE:-self.MEM_SIZE] = horizons.log_()

    def forward(self, x_vec: Tensor, x_spa: Tensor, memp: Tensor, memv: Tensor) -> 'tuple[Tensor, Tensor]':

        # Encode spatial state information
        x_spa = self.spaencoder(x_spa).repeat_interleave(self.n_bots, dim=0, output_size=self.n_all_bots)

        # Consolidate local observations with global state information
        x = torch.cat((x_vec, x_spa, memp), dim=-1)
        x = self.activ(self.fcin(x))

        # Update temporal variables
        memv = self.rnn(x, memv)

        # Map to value distribution parameters
        x = self.mlpout(memv)

        # Detach grads. from non-speakers
        if self.detach_cond:
            speaker_mask = x_vec[:, self.role_slice]
            x = torch.lerp(x.detach(), x, speaker_mask)
            memv = torch.lerp(memv.detach(), memv, speaker_mask)

        return x, memv


class ActorCritic(ActorCriticTemplate):
    """Wrapper around the visual encoder, policy, and valuator networks."""

    def __init__(
        self,
        n_agents_per_env: int = 1,
        n_envs: int = 1,
        com_state: int = Policy.FEAT_TRAINED,
        guide_state: int = Policy.FEAT_DISABLED,
        com_out_size: int = cfg.N_RCVR_CLR_CLASSES,
        com_bias: bool = True
    ):
        super().__init__()

        self.visencoder = VisEncoder()
        self.policy = Policy(com_state, guide_state, com_out_size, com_bias, n_agents_per_env, n_envs)
        self.valuator = Valuator(com_state, n_agents_per_env, n_envs)

        for param in self.visencoder.parameters():
            param.requires_grad = False

        self.policy.random_init()
        self.valuator.random_init()

    def init_mem(self, batch_size: int = 1) -> 'tuple[Tensor, Tensor]':
        memp = self.policy.mem_initial.detach().expand(batch_size, -1).clone()
        memv = self.valuator.mem_initial.detach().expand(batch_size, -1).clone()

        return memp, memv

    def reset_mem(self, mem: 'tuple[Tensor, Tensor]', nonreset_mask: Tensor) -> 'tuple[Tensor, Tensor]':
        memp, memv = mem

        memp = torch.lerp(self.policy.mem_initial, memp, nonreset_mask)
        memv = torch.lerp(self.valuator.mem_initial, memv, nonreset_mask)

        return memp, memv

    def get_distr(self, args: 'Tensor | tuple[Tensor, ...]', from_raw: bool) -> MultiCategorical:
        if from_raw:
            return MultiCategorical.from_raw(self.policy.out_values, args.split(self.policy.output_sizes, dim=-1))

        return MultiCategorical(self.policy.out_values, args)

    def unwrap_sample(self, sample: 'tuple[Tensor, Tensor]') -> 'tuple[Tensor, Tensor]':
        return sample[0], sample[1][:, 1]

    def act_partial(
        self,
        img: Tensor,
        vec: Tensor,
        _spa: Tensor,
        memp: Tensor,
        _memv: Tensor
    ) -> 'tuple[Tensor, Tensor]':

        with torch.no_grad():
            img_enc = self.visencoder(img).flatten(1)

            inp = torch.cat((img_enc, vec[:, :cfg.OBS_VEC_SIZE+cfg.AUX_VAL_SIZE], vec[:, -cfg.META_VAL_SIZE:]), dim=-1)

            x, _, memp = self.policy(inp, memp, False)

        return x, memp

    def act(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        sample: bool
    ) -> 'tuple[tuple[Tensor, ...], tuple[Tensor, ...]]':

        x, memp = self.act_partial(*obs, *mem)
        act = self.get_distr(x, from_raw=True)

        if sample:
            x = act.sample()

        else:
            x = act.mode, torch.stack([lp.argmax(dim=-1) for lp in act.log_prob_tpl], dim=-1)

        return x, (memp, mem[-1])

    def fwd_partial(
        self,
        inp: Tensor,
        inv: Tensor,
        spa: Tensor,
        memp: Tensor,
        memv: Tensor,
        detach: bool
    ) -> 'tuple[Tensor, ...]':

        x, a, memp = self.policy(inp, memp, detach)
        v, memv = self.valuator(inv, spa, memp.detach() if detach else GradScaler.apply(memp, 0.01), memv)

        return x, v, a, memp, memv

    def collect_static(
        self,
        img: Tensor,
        vec: Tensor,
        spa: Tensor,
        memp: Tensor,
        memv: Tensor
    ) -> 'tuple[Tensor, ...]':

        with torch.no_grad():
            img_enc = self.visencoder(img).flatten(1)

            inp = torch.cat((img_enc, vec[:, :cfg.OBS_VEC_SIZE+cfg.AUX_VAL_SIZE], vec[:, -cfg.META_VAL_SIZE:]), dim=-1)
            inv = torch.cat((vec[:, :cfg.OBS_LOC_SIZE], vec[:, cfg.STATE_VEC_SLICE]), dim=-1)

            x, v, _, memp, memv = self.fwd_partial(inp, inv, spa, memp, memv, detach=False)

        return x, v, inp, inv, memp, memv

    def collect_copied(
        self,
        inp: Tensor,
        inv: Tensor,
        spa: Tensor,
        memp: Tensor,
        memv: Tensor
    ) -> 'tuple[Tensor, ...]':

        with torch.no_grad():
            x, v, _, memp, memv = self.fwd_partial(inp, inv, spa, memp, memv, detach=False)

        return x, v, memp, memv

    def collect(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        encode: bool
    ) -> 'tuple[Tensor, Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]':

        if encode:
            x, val_mean, inp, inv, memp, memv = self.collect_static(*obs, *mem)
            obs = inp, inv, obs[-1].clone()

        else:
            x, val_mean, memp, memv = self.collect_copied(*obs, *mem)

        return x, val_mean, obs, (memp, memv)

    def forward(
        self,
        obs: 'tuple[Tensor, ...]',
        mem: 'tuple[Tensor, ...]',
        detach: bool = False
    ) -> 'tuple[MultiCategorical, FixedVarNormal, FixedVarNormal, tuple[Tensor, ...]]':

        x, v, a, memp, memv = self.fwd_partial(*obs, *mem, detach=detach)

        act = self.get_distr(x, from_raw=True)
        val = FixedVarNormal(v, skip_log_shift=True)
        aux = FixedVarNormal(a, skip_log_shift=True)

        return act, val, aux, (memp, memv)
