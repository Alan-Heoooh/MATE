"""Shared snippets for deploy/eval scripts (action geometry, safety)."""

import numpy as np

from utils.constants import SAFE_WORKSPACE_MIN, SAFE_WORKSPACE_MAX, SAFE_EPS


def project_policy_action_to_base(
    action, projector, *, camera_serial, action_coord, action_rotation_repr="rotation_6d"
):
    """Map a policy chunk from ``action_coord`` to base.

    ``action`` is (..., 3+6+1): translation + action rotation repr + width.
    """
    action_tcp = action[..., :-1]
    action_width = action[..., -1]
    tcp_base = projector.project_tcp(
        action_tcp,
        source_coord=action_coord,
        target_coord="base",
        cam=camera_serial,
        rotation_rep=action_rotation_repr,
    )
    return tcp_base, action_width


def apply_safe_workspace_clip_to_tcp(tcp, *, safe_min=None, safe_max=None, eps=None):
    """Clip only translation channels to the default safe workspace (copy)."""
    if safe_min is None:
        safe_min = SAFE_WORKSPACE_MIN
    if safe_max is None:
        safe_max = SAFE_WORKSPACE_MAX
    if eps is None:
        eps = SAFE_EPS
    out = np.asarray(tcp).copy()
    out[..., :3] = np.clip(out[..., :3], np.asarray(safe_min) + eps, np.asarray(safe_max) - eps)
    return out


def concat_tcp_width(tcp, width):
    return np.concatenate([tcp, np.asarray(width)[..., np.newaxis]], axis=-1)


def pack_action_trajectory_for_robot(
    action,
    projector,
    *,
    camera_serial,
    action_coord,
    action_rotation_repr="rotation_6d",
    clip_safe_workspace=True,
    safe_min=None,
    safe_max=None,
    safe_eps=None,
):
    """Project to base, optionally clip translation, return full (tcp_base, width) trajectory."""
    tcp_b, w = project_policy_action_to_base(
        action,
        projector,
        camera_serial=camera_serial,
        action_coord=action_coord,
        action_rotation_repr=action_rotation_repr,
    )
    if clip_safe_workspace:
        tcp_b = apply_safe_workspace_clip_to_tcp(tcp_b, safe_min=safe_min, safe_max=safe_max, eps=safe_eps)
    return concat_tcp_width(tcp_b, w)
