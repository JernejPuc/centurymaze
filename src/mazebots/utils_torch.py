"""PyTorch utilities"""

from functools import cached_property
from typing import Callable

import torch
from torch import Tensor
from torch.nn.functional import logsigmoid, log_softmax


# Truncated 1/x characteristic (1 at 0, 0 at max distance)
# Lower K corresponds to steeper slope, i.e. higher sensitivity in close range
MAX_DIST = 128.
DIST_NORM_K = 2
DIST_NORM_ADD = DIST_NORM_K
DIST_NORM_SUB = DIST_NORM_K/MAX_DIST
DIST_NORM_MUL = (MAX_DIST+DIST_NORM_ADD)*DIST_NORM_SUB


def adjust_depth_range(dep: Tensor):
    dep = torch.clamp(dep, 0., MAX_DIST)
    dep = DIST_NORM_MUL / (dep + DIST_NORM_ADD) - DIST_NORM_SUB

    return dep


def apply_quat_rot(q: Tensor, v: Tensor) -> Tensor:
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.

    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions. Returns a tensor of shape (*, 3).

    Sources:
    https://github.com/facebookresearch/QuaterNet/blob/main/common/quaternion.py#L33
    https://fgiesen.wordpress.com/2019/02/09/rotating-a-single-vector-using-a-quaternion/
    """

    qvec = q[:, :-1]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)

    return (v + 2. * (q[:, -1:] * uv + uuv))


def get_eulz_from_quat(q: Tensor) -> Tensor:
    """
    Infer Euler angle z-component (xyz sequence) from quaternion q.

    Based on trials:  # 1, 3, 5, 6 seem to work (xyz, zxy, yxz, zyx)
    z1 = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    z2 = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1, 1))
    z3 = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q1 * q1 + q3 * q3))
    z4 = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1, 1))
    z5 = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2*(q1 * q1 + q3 * q3))
    z6 = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    """

    q0 = q[:, -1]
    q1 = q[:, 0]
    q2 = q[:, 1]
    q3 = q[:, 2]

    return torch.atan2(2. * (q0 * q3 - q1 * q2), 1. - 2.*(q2 * q2 + q3 * q3))[..., None]


def symlog(x: Tensor) -> Tensor:
    return x.sign() * (x.abs() + 1.).log()


def symexp(x: Tensor) -> Tensor:
    return x.sign() * (x.abs().exp() - 1.)


class TaLU(torch.autograd.Function):
    """Tangential linear unit with param. 1/10."""

    @staticmethod
    def forward(ctx, i: Tensor) -> Tensor:
        x = i.mul(10.).tanh_().div_(10.).clamp_min_(i)
        ctx.save_for_backward(i)

        return x

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g: Tensor) -> Tensor:
        i, = ctx.saved_tensors
        x = i.mul(10.).clamp_max_(0.).cosh_().pow_(-2).mul_(g)

        return x


class ActionDistrTemplate:
    sample: 'Tensor | Callable[[], Tensor]'
    entropy: 'Tensor | Callable[[], Tensor]'
    log_prob: 'Tensor | Callable[[Tensor], Tensor]'
    kl_div: 'Tensor | Callable[[ActionDistrTemplate], Tensor]'


class ValueDistrTemplate:
    expect: 'Tensor | Callable[[], Tensor]'
    log_prob: 'Tensor | Callable[[Tensor], Tensor]'


class DiagNormal(ActionDistrTemplate):
    """
    Multivariate normal with diagonal covariance matrix,
    i.e. independent normal axes, where only diagonal elements are non-zero
    (no correlations between variables are intended).
    """

    _05log2pi = 0.9189385332046727
    _05p05log2pi = 1.4189385332046727

    def __init__(self, loc: Tensor, scale: Tensor, sample: Tensor = None, pseudo: bool = False):
        self.loc = loc

        # Shadow cached properties
        if not pseudo:
            self.scale = scale
            self.pseudo_scale = None

        else:
            self.pseudo_scale = scale

        if sample is not None:
            self.sample = sample

    @cached_property
    def log_scale(self) -> Tensor:
        return self.scale.log() if self.pseudo_scale is None else logsigmoid(self.pseudo_scale)

    @cached_property
    def scale(self) -> Tensor:
        return torch.sigmoid(self.pseudo_scale)

    @cached_property
    def var(self) -> Tensor:
        return self.scale ** 2

    @cached_property
    def dbl_var(self) -> Tensor:
        return 2. * self.var

    def sample(self) -> Tensor:
        return torch.normal(self.loc, self.scale)

    def entropy(self) -> Tensor:
        return (self._05p05log2pi + self.log_scale).sum(-1)

    def log_prob(self, values: Tensor) -> Tensor:
        return (-((values - self.loc)**2) / self.dbl_var - self.log_scale - self._05log2pi).sum(-1)

    def kl_div(self, othr: 'DiagNormal') -> Tensor:
        return (self.log_scale - othr.log_scale + ((othr.var + (othr.loc - self.loc)**2) / self.dbl_var) - 0.5).sum(-1)


class InterpDiscrete(ValueDistrTemplate):
    """
    Categorical distribution, where values represent the delimiters between bins.
    Log probability of a value is interpolated in proportion to log probabilities
    of the two bounding values.
    """

    def __init__(self, logits: Tensor, values: Tensor, values_as_indices: bool = False):
        self.logits = logits
        self.values = values
        self.values_as_indices = values_as_indices
        self.add_dims = tuple([None for _ in range(len(logits.shape)-1)])
        self.expect = self.mean

    @cached_property
    def dim_match_values(self) -> Tensor:
        return self.values[self.add_dims] if self.add_dims else self.values

    @cached_property
    def log_probs(self) -> Tensor:
        return log_softmax(self.logits, dim=-1)

    @cached_property
    def probs(self) -> Tensor:
        return torch.softmax(self.logits, dim=-1)

    def mean(self) -> Tensor:
        return (self.dim_match_values * self.probs).sum(-1)

    def mode(self) -> Tensor:
        return self.values[..., self.logits.argmax(dim=-1)]

    def sample(self) -> Tensor:
        return torch.multinomial(self.probs, 1)

    def entropy(self) -> Tensor:
        return -(self.probs * self.log_probs).sum(-1)

    def log_prob(self, values: Tensor) -> Tensor:
        ridx = torch.bucketize(values, self.values)
        lidx = ridx-1

        if self.values_as_indices:
            rval = ridx
            lval = lidx

        else:
            rval = self.values[ridx]
            lval = self.values[lidx]

        ratio = (values - lval) / (rval - lval)

        return self.log_probs[..., lidx] * (1. - ratio) + self.log_probs[..., ridx] * ratio


class AsymmetricLaplace:
    def __init__(self, loc: Tensor, scale: Tensor, skew: Tensor):
        self.loc = loc
        self.scale = scale
        self.skew = skew
        self.mean = self.expect

    @cached_property
    def lscale(self) -> Tensor:
        return self.scale * self.skew

    @cached_property
    def rscale(self) -> Tensor:
        return self.scale / self.skew

    def expect(self) -> Tensor:
        return self.loc + (1. - self.skew**2) * self.rscale

    def var(self) -> Tensor:
        return (1. + self.skew**4) * self.rscale**2

    def sample(self) -> Tensor:
        exp1, exp2 = self.loc.new_empty(2, *self.loc.shape).exponential_()

        return self.loc - self.lscale * exp1 + self.rscale * exp2

    def entropy(self) -> Tensor:
        return 1. + torch.log(1. + self.skew**2) + torch.log(self.rscale)

    def log_prob(self, value: Tensor) -> Tensor:
        d = value - self.loc
        d = -d.abs() / torch.where(d > 0., self.rscale, self.lscale)

        return d - torch.log(self.lscale + self.rscale)
