import torch


def _expand_pose_vector_for_action(action_block: torch.Tensor, pose_vec: torch.Tensor) -> torch.Tensor:
    """
    Expand a pose-level vector, e.g. shape [B, 3], to match an action block,
    e.g. [B, T, 3], by inserting singleton dimensions before the last dim.

    This is useful when p_t1 / p_t2 / R_t1 / R_t2 are pose-level quantities,
    while x_t1 / x_t2 are horizon-level action tensors.
    """
    while pose_vec.ndim < action_block.ndim:
        pose_vec = pose_vec.unsqueeze(-2)
    return pose_vec


def _expand_rotation_for_action(action_block: torch.Tensor, rot_mat: torch.Tensor) -> torch.Tensor:
    """Expand a pose-level rotation matrix [*,3,3] to match action block [*,T,3]."""
    while rot_mat.ndim < action_block.ndim + 1:
        rot_mat = rot_mat.unsqueeze(-3)
    return rot_mat


def _validate_rotation_repr(rotation_repr: str) -> str:
    valid = {"rotation_col_6d"}
    if rotation_repr not in valid:
        raise ValueError(
            "Unsupported rotation_repr: {} (expected one of {}).".format(
                rotation_repr, sorted(valid)
            )
        )
    return rotation_repr


def _transform_rot6d_t1_to_t2(
    rot6_t1: torch.Tensor,
    q_t2_from_t1: torch.Tensor,
    *,
    rotation_repr: str,
) -> torch.Tensor:
    """Linear orthogonal transform in noise space (distribution-preserving)."""
    _validate_rotation_repr(rotation_repr)
    q_t2_from_t1 = _expand_rotation_for_action(rot6_t1[..., :3], q_t2_from_t1)
    r1_t1 = rot6_t1[..., 0:3]
    r2_t1 = rot6_t1[..., 3:6]
    r1_t2 = torch.einsum("...ij,...j->...i", q_t2_from_t1, r1_t1)
    r2_t2 = torch.einsum("...ij,...j->...i", q_t2_from_t1, r2_t1)
    return torch.cat([r1_t2, r2_t2], dim=-1)


def _transform_rot6d_t2_to_t1(
    rot6_t2: torch.Tensor,
    q_t2_from_t1: torch.Tensor,
    *,
    rotation_repr: str,
) -> torch.Tensor:
    """Inverse linear orthogonal transform in noise space."""
    _validate_rotation_repr(rotation_repr)
    q_t2_from_t1 = _expand_rotation_for_action(rot6_t2[..., :3], q_t2_from_t1)
    q_t1_from_t2 = q_t2_from_t1.transpose(-1, -2)
    r1_t2 = rot6_t2[..., 0:3]
    r2_t2 = rot6_t2[..., 3:6]
    r1_t1 = torch.einsum("...ij,...j->...i", q_t1_from_t2, r1_t2)
    r2_t1 = torch.einsum("...ij,...j->...i", q_t1_from_t2, r2_t2)
    return torch.cat([r1_t1, r2_t1], dim=-1)


def convert_noise_10d_t1_to_t2(
    eps_t1: torch.Tensor,
    R_t1: torch.Tensor,
    R_t2: torch.Tensor,
    *,
    rotation_repr: str = "rotation_col_6d",
) -> torch.Tensor:
    """
    Convert 10D action noise from t1 TCP-local coordinate to t2 TCP-local coordinate.

    Rotation channels are transformed as two independent 3D blocks with an
    orthogonal map q, preserving Gaussian scale in diffusion latent space.

    Action/noise format:
        eps[..., 0:3]   = translation noise
        eps[..., 3:6]   = first row-vector block of PyTorch3D 6D rotation
        eps[..., 6:9]   = second row-vector block of PyTorch3D 6D rotation
        eps[..., 9:10]  = gripper width noise

    Coordinate transform:
        q = R_t2^T R_t1

    For column-vector translation noise:
        eps_pos_t2 = q eps_pos_t1

    For rotation noise blocks:
        r_t2 = q r_t1
    ``rotation_repr`` controls the semantic meaning of channels [3:9].
    Supported value: ``rotation_col_6d``.
    """
    _validate_rotation_repr(rotation_repr)
    assert eps_t1.shape[-1] == 10

    eps_pos = eps_t1[..., 0:3]
    eps_r1 = eps_t1[..., 3:6]
    eps_r2 = eps_t1[..., 6:9]
    eps_gripper = eps_t1[..., 9:10]

    q = R_t2.mT @ R_t1
    q = _expand_rotation_for_action(eps_pos, q)

    # Translation: column-vector convention.
    eps_pos_t2 = torch.einsum("...ij,...j->...i", q, eps_pos)

    rot6_t1 = torch.cat([eps_r1, eps_r2], dim=-1)
    rot6_t2 = _transform_rot6d_t1_to_t2(
        rot6_t1,
        q,
        rotation_repr=rotation_repr,
    )
    eps_r1_t2 = rot6_t2[..., 0:3]
    eps_r2_t2 = rot6_t2[..., 3:6]

    return torch.cat(
        [eps_pos_t2, eps_r1_t2, eps_r2_t2, eps_gripper],
        dim=-1,
    )


def convert_noise_10d_t2_to_t1(
    eps_t2: torch.Tensor,
    R_t1: torch.Tensor,
    R_t2: torch.Tensor,
    *,
    rotation_repr: str = "rotation_col_6d",
) -> torch.Tensor:
    """
    Convert 10D action noise from t2 TCP-local coordinate to t1 TCP-local coordinate.

    Inverse of convert_noise_10d_t1_to_t2.

    Coordinate transform:
        q = R_t2^T R_t1

    Inverse translation:
        eps_pos_t1 = q^T eps_pos_t2

    Inverse rotation noise blocks:
        r_t1 = q^T r_t2
    ``rotation_repr`` controls the semantic meaning of channels [3:9].
    Supported value: ``rotation_col_6d``.
    """
    _validate_rotation_repr(rotation_repr)
    assert eps_t2.shape[-1] == 10

    eps_pos = eps_t2[..., 0:3]
    eps_r1 = eps_t2[..., 3:6]
    eps_r2 = eps_t2[..., 6:9]
    eps_gripper = eps_t2[..., 9:10]

    q = R_t2.mT @ R_t1
    q = _expand_rotation_for_action(eps_pos, q)

    # Translation inverse: q^T maps t2 -> t1.
    eps_pos_t1 = torch.einsum("...ij,...j->...i", q.transpose(-1, -2), eps_pos)

    rot6_t2 = torch.cat([eps_r1, eps_r2], dim=-1)
    rot6_t1 = _transform_rot6d_t2_to_t1(
        rot6_t2,
        q,
        rotation_repr=rotation_repr,
    )
    eps_r1_t1 = rot6_t1[..., 0:3]
    eps_r2_t1 = rot6_t1[..., 3:6]

    return torch.cat(
        [eps_pos_t1, eps_r1_t1, eps_r2_t1, eps_gripper],
        dim=-1,
    )


def convert_noisy_sample_10d_t1_to_t2(
    x_t1: torch.Tensor,
    R_t1: torch.Tensor,
    R_t2: torch.Tensor,
    p_t1: torch.Tensor,
    p_t2: torch.Tensor,
    alpha_bar_t: torch.Tensor,
    *,
    position_scale,
    rotation_repr: str = "rotation_col_6d",
) -> torch.Tensor:
    """
    Convert a current noisy 10D action sample from t1 TCP-local coordinates
    to t2 TCP-local coordinates. This is the forward transform.

    Rotation channels are transformed with a linear orthogonal map q instead
    of projection through SO(3), preserving latent/noise statistics.

    Coordinate transform:
        q = R_t2^T R_t1

    Clean translation action transform in raw meters:
        a0_t2 = q a0_t1 + b_p

    where:
        b_p = R_t2^T (p_t1 - p_t2)

    Noisy sample transform:
        x_t2 = q x_t1 + sqrt(alpha_bar_t) b_p

    Since the diffusion model usually operates in normalized action space,
    b_p must be divided by position_scale before being added to x_t2.

    Important:
        position_scale should match the normalization used for translation
        channels, usually:

            position_scale = (action_trans_max - action_trans_min) / 2

        This implementation assumes the translation normalization is centered
        around zero and isotropic, or at least compatible with the rotation q.
    ``rotation_repr`` controls the semantic meaning of channels [3:9].
    Supported value: ``rotation_col_6d``.
    """
    _validate_rotation_repr(rotation_repr)
    assert x_t1.shape[-1] == 10

    x_pos_t1 = x_t1[..., 0:3]
    x_r1_t1 = x_t1[..., 3:6]
    x_r2_t1 = x_t1[..., 6:9]
    x_gripper = x_t1[..., 9:10]

    sqrt_alpha = torch.sqrt(alpha_bar_t)
    while sqrt_alpha.ndim < x_pos_t1.ndim:
        sqrt_alpha = sqrt_alpha.unsqueeze(-1)

    q = R_t2.mT @ R_t1
    q = _expand_rotation_for_action(x_pos_t1, q)

    # Translation offset in t2 local frame, raw meters.
    b_p_raw = (R_t2.mT @ (p_t1 - p_t2).unsqueeze(-1)).squeeze(-1)

    # Convert raw-meter offset to normalized action space.
    position_scale_t = torch.as_tensor(
        position_scale,
        device=b_p_raw.device,
        dtype=b_p_raw.dtype,
    )
    b_p = b_p_raw / position_scale_t
    b_p = _expand_pose_vector_for_action(x_pos_t1, b_p)

    # Translation: column-vector convention.
    x_pos_t2 = torch.einsum("...ij,...j->...i", q, x_pos_t1) + sqrt_alpha * b_p

    rot6_t1 = torch.cat([x_r1_t1, x_r2_t1], dim=-1)
    rot6_t2 = _transform_rot6d_t1_to_t2(
        rot6_t1,
        q,
        rotation_repr=rotation_repr,
    )
    x_r1_t2 = rot6_t2[..., 0:3]
    x_r2_t2 = rot6_t2[..., 3:6]

    return torch.cat(
        [x_pos_t2, x_r1_t2, x_r2_t2, x_gripper],
        dim=-1,
    )
