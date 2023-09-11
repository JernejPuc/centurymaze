"""PyTorch utilities"""

import torch
from torch import Tensor


def norm_depth_range(dep: Tensor, max_dep: float = 128., slope: float = 0.5) -> Tensor:
    """
    Truncated 1/x characteristic (1 at 0, 0 at max distance).
    Lower K corresponds to steeper slope, i.e. higher sensitivity in close range.
    """

    add_term = 1. / slope
    sub_term = add_term / max_dep
    mul_term = (max_dep + add_term) * sub_term

    dep = torch.clamp(dep, 0., max_dep)
    dep = mul_term / (dep + add_term) - sub_term

    return dep


def clip_angle_range(ang: Tensor):
    """Bound angles between -pi and pi."""

    return ang + ((ang < -torch.pi).float() - (ang > torch.pi).float()) * (2. * torch.pi)


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

    return v + 2. * (q[:, -1:] * uv + uuv)


def get_eulz_from_quat(q: Tensor, keepdim: bool = True) -> Tensor:
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

    q0 = q[..., -1]
    q1 = q[..., 0]
    q2 = q[..., 1]
    q3 = q[..., 2]

    eulz = torch.atan2(2. * (q0 * q3 - q1 * q2), 1. - 2.*(q2 * q2 + q3 * q3))

    return eulz.unsqueeze(-1) if keepdim else eulz


def rgb_to_hsv(img: Tensor, unstack_dim: int = -1, stack_dim: int = -1) -> Tensor:
    """
    See: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py#L262
    """

    r, g, b = img.unbind(dim=unstack_dim)
    max_val = torch.ones_like(r)

    max_channel_val = torch.max(img, dim=-1).values
    min_channel_val = torch.min(img, dim=-1).values

    eq_channel_mask_f = (max_channel_val == min_channel_val).float()
    channel_diff = max_channel_val - min_channel_val

    masked_diff = torch.lerp(channel_diff, max_val, eq_channel_mask_f)
    sat = channel_diff / torch.lerp(max_channel_val, max_val, eq_channel_mask_f)

    r_diff = (max_channel_val - r) / masked_diff
    g_diff = (max_channel_val - g) / masked_diff
    b_diff = (max_channel_val - b) / masked_diff

    r_hue = (max_channel_val == r) * (b_diff - g_diff)
    g_hue = ((max_channel_val == g) & (max_channel_val != r)) * (2. + r_diff - b_diff)
    b_hue = ((max_channel_val != g) & (max_channel_val != r)) * (4. + g_diff - r_diff)

    hue = r_hue + g_hue + b_hue
    hue = torch.fmod((hue / 6. + 1.), 1.)

    return torch.stack((hue, sat, max_channel_val), dim=stack_dim)
