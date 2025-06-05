import torch
import torch.nn.functional as F


def deg2rad_tensor(deg, dtype=torch.float32, device="cuda"):
    deg = torch.tensor(deg, dtype=dtype, device=device)
    return deg * torch.pi / 180.0


def rotation_matrix_x(theta:torch.Tensor) -> torch.Tensor:
    if theta.shape == torch.Size([]):
        theta = theta.unsqueeze(0)
    
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    n = theta.shape[0]
    T = torch.eye(3, dtype=theta.dtype, device=theta.device).unsqueeze(0).repeat(n, 1, 1)  # (n, 4, 4)
    
    T[:, 1, 1] = cos_theta
    T[:, 1, 2] = -sin_theta
    T[:, 2, 1] = sin_theta
    T[:, 2, 2] = cos_theta
    
    return T


def rotation_matrix_y(theta:torch.Tensor) -> torch.Tensor:
    if theta.shape == torch.Size([]):
        theta = theta.unsqueeze(0)
    
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    n = theta.shape[0]
    T = torch.eye(3, dtype=theta.dtype, device=theta.device).unsqueeze(0).repeat(n, 1, 1)  # (n, 4, 4)

    T[:, 0, 0] = cos_theta
    T[:, 0, 2] = sin_theta
    T[:, 2, 0] = -sin_theta
    T[:, 2, 2] = cos_theta
    
    return T


def rotation_matrix_z(theta, dtype=torch.float32, device="cuda"):
    if theta.shape == torch.Size([]):
        theta = theta.unsqueeze(0)
    
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    n = theta.shape[0]
    T = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).repeat(n, 1, 1)  # (n, 4, 4)

    T[:, 0, 0] = cos_theta
    T[:, 0, 1] = -sin_theta
    T[:, 1, 0] = sin_theta
    T[:, 1, 1] = cos_theta
    
    return T


def TTrans(trans:torch.Tensor) ->  torch.Tensor:
    if not (isinstance(trans, torch.Tensor) and trans.shape[-1] == 3):
        raise ValueError('trans must be a torch tensor with shape (, 3)')
    
    if trans.ndim == 1:
        trans = trans.unsqueeze(0)
    
    n = trans.shape[0]
    T = torch.eye(4, dtype=trans.dtype, device=trans.device).unsqueeze(0).repeat(n, 1, 1)  # (n, 4, 4)
    T[:,0,3] = trans[:,0]
    T[:,1,3] = trans[:,1]
    T[:,2,3] = trans[:,2]
    return T


def TTransX(x:torch.Tensor) -> torch.Tensor:
    if x.shape == torch.Size([]):
        x = x.unsqueeze(0)
    n = x.shape[0]
    T = torch.eye(4, dtype=x.dtype, device=x.device).unsqueeze(0).repeat(n, 1, 1)  # (n, 4, 4)
    T[:,0,3] = x
    return T


def TTransY(y:torch.Tensor) -> torch.Tensor:
    if y.shape == torch.Size([]):
        y = y.unsqueeze(0)
    n = y.shape[0]
    T = torch.eye(4, dtype=y.dtype, device=y.device).unsqueeze(0).repeat(n, 1, 1)  # (n, 4, 4)
    T[:,1,3] = y
    return T


def TTransZ(z:torch.Tensor) -> torch.Tensor:
    if z.shape == torch.Size([]):
        z = z.unsqueeze(0)
    n = z.shape[0]
    T = torch.eye(4, dtype=z.dtype, device=z.device).unsqueeze(0).repeat(n, 1, 1)  # (n, 4, 4)
    T[:,2,3] = z
    return T


def TRotX(theta:torch.Tensor) -> torch.Tensor:
    if theta.shape == torch.Size([]):
        theta = theta.unsqueeze(0)
    T = torch.eye(4, dtype=theta.dtype, device=theta.device).unsqueeze(0).repeat(theta.shape[0], 1, 1)  # (n, 4, 4)
    T[:,:3,:3] = rotation_matrix_x(theta)
    return T


def TRotY(theta:torch.Tensor) -> torch.Tensor:
    if theta.shape == torch.Size([]):
        theta = theta.unsqueeze(0)
    T = torch.eye(4, dtype=theta.dtype, device=theta.device).unsqueeze(0).repeat(theta.shape[0], 1, 1)  # (n, 4, 4)
    T[:,:3,:3] = rotation_matrix_y(theta)
    return T


def TRotZ(theta:torch.Tensor) -> torch.Tensor:
    if theta.shape == torch.Size([]):
        theta = theta.unsqueeze(0)
    T = torch.eye(4, dtype=theta.dtype, device=theta.device).unsqueeze(0).repeat(theta.shape[0], 1, 1)  # (n, 4, 4)
    T[:,:3,:3] = rotation_matrix_z(theta)
    return T


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling


if __name__ == '__main__':
    rotx = TRotX(torch.tensor([torch.pi/2.0,-torch.pi/2.0], dtype=torch.float32, device="cuda"))
    print(rotx)
    print(rotx.shape)
    quat = matrix_to_quaternion(rotx[:,:3,:3])
    print(quat)
