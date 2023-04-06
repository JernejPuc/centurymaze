"""PyTorch utilities"""

import torch
from torch import Tensor

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
