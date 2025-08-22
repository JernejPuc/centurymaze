"""PyTorch utilities"""

import torch
from torch import Tensor


def norm_distance(dist: Tensor, max_dist: float = 24., dist_bias: float = 2., dist_scale: float = 1.) -> Tensor:
    """
    Truncated inverse characteristic: 1 at 0 (dist. bias), 0 at noise threshold (max dist.).
    Lower bias corresponds to steeper slope, i.e. higher sensitivity in close range,
    while scaling can be used to adjust values in distant range.
    """

    sub_term = dist_bias / max_dist
    mul_term = (max_dist + dist_bias) * sub_term

    if dist_scale != 1.:
        dist = dist * dist_scale

    dist = torch.clamp(dist, 0., max_dist)
    res = mul_term / (dist + dist_bias) - sub_term

    return res


def clip_angle_range(ang: Tensor):
    """Bound angles between -pi and pi."""

    return ang + (ang < -torch.pi) * (2.*torch.pi) + (ang > torch.pi) * (-2.*torch.pi)


def check_fov(diff_to_pt: Tensor, half_angle: float = torch.pi / 4.) -> Tensor:
    """
    Check whether the given difference (in the local frame) corresponds to
    a point within the given angle range (field of view).
    """

    return torch.atan(diff_to_pt[:, 1] / diff_to_pt[:, 0].clip(1e-6)).abs() < half_angle


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

    qx, qy, qz, qw = q.unbind(-1)

    eulz = torch.atan2(2. * (qw * qz - qx * qy), 1. - 2. * (qy**2 + qz**2))

    return eulz.unsqueeze(-1) if keepdim else eulz


def get_trigonz_from_quat(q: Tensor):
    """Infer Euler angle z-component sine and cosine from quaternion q."""

    qx, qy, qz, qw = q.unbind(-1)

    sin_z = 2. * (qw * qz - qx * qy)
    cos_z = 1. - 2. * (qy**2 + qz**2)

    return torch.stack((sin_z, cos_z), axis=-1)


def rgb_to_hsv(img: Tensor, unstack_dim: int = -1, stack_dim: int = -1) -> Tensor:
    """
    See: https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py#L262
    """

    r, g, b = img.unbind(dim=unstack_dim)
    max_val = torch.ones_like(r)

    max_channel_val = torch.max(img, dim=unstack_dim).values
    min_channel_val = torch.min(img, dim=unstack_dim).values

    eq_channel_mask = (max_channel_val == min_channel_val)
    channel_diff = max_channel_val - min_channel_val

    masked_diff = torch.where(eq_channel_mask, max_val, channel_diff)
    sat = channel_diff / torch.where(eq_channel_mask, max_val, max_channel_val)

    r_diff = (max_channel_val - r) / masked_diff
    g_diff = (max_channel_val - g) / masked_diff
    b_diff = (max_channel_val - b) / masked_diff

    r_hue = (max_channel_val == r) * (b_diff - g_diff)
    g_hue = ((max_channel_val == g) & (max_channel_val != r)) * (2. + r_diff - b_diff)
    b_hue = ((max_channel_val != g) & (max_channel_val != r)) * (4. + g_diff - r_diff)

    hue = r_hue + g_hue + b_hue
    hue = torch.fmod((hue / 6. + 1.), 1.)

    return torch.stack((hue, sat, max_channel_val), dim=stack_dim)
