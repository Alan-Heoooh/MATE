import numpy as np

def normalize_tcp(tcp_list, config, gripper_width_included = True, trans_min = None, trans_max = None, max_gripper_width = None):
    ''' tcp_list: [T, 3(trans) + 6(rot) + 1(width)]'''
    tcp_list = np.asarray(tcp_list).copy()
    config_norm = config.data.normalization
    if trans_min is None:
        trans_min = np.array(config_norm.trans_min)
    else:
        trans_min = np.array(trans_min)
    if trans_max is None:
        trans_max = np.array(config_norm.trans_max)
    else:
        trans_max = np.array(trans_max)
    if max_gripper_width is None:
        max_gripper_width = config_norm.max_gripper_width
    tcp_list[:, :3] = (tcp_list[:, :3] - trans_min) / (trans_max - trans_min) * 2 - 1
    if gripper_width_included:
        tcp_list[:, -1] = tcp_list[:, -1] / max_gripper_width * 2 - 1
    return tcp_list

def normalize_obs_proprio(obs_proprio, config):
    '''
    Normalize obs_proprio using obs_proprio-specific normalization parameters.
    obs_proprio: [T, 9] = [T, 3(trans) + 6(rot_6d)]
    Automatically uses obs_proprio_trans_min/max if available, otherwise falls back to default trans_min/max.
    '''
    config_norm = config.data.normalization
    # Use obs_proprio-specific normalization if available, otherwise use default
    if hasattr(config_norm, 'obs_proprio_trans_min') and hasattr(config_norm, 'obs_proprio_trans_max'):
        trans_min = np.array(config_norm.obs_proprio_trans_min)
        trans_max = np.array(config_norm.obs_proprio_trans_max)
    else:
        trans_min = np.array(config_norm.trans_min)
        trans_max = np.array(config_norm.trans_max)
    return normalize_tcp(obs_proprio, config, gripper_width_included=False,
                        trans_min=trans_min, trans_max=trans_max)


def normalize_action(action, config):
    '''
    Normalize action using action-specific normalization parameters.
    action: [T, 10] = [T, 3(trans) + 6(rot_6d) + 1(gripper)]
    Automatically uses action_trans_min/max if available, otherwise falls back to default trans_min/max.
    '''
    config_norm = config.data.normalization
    # Use action-specific normalization if available, otherwise use default
    if hasattr(config_norm, 'action_trans_min') and hasattr(config_norm, 'action_trans_max'):
        trans_min = np.array(config_norm.action_trans_min)
        trans_max = np.array(config_norm.action_trans_max)
    else:
        trans_min = np.array(config_norm.trans_min)
        trans_max = np.array(config_norm.trans_max)
    return normalize_tcp(action, config, gripper_width_included=True,
                        trans_min=trans_min, trans_max=trans_max)

def normalize_global_action(action, config):
    """Normalize an absolute action in a fixed global coordinate frame.

    Global actions use ``trans_min/max``.  They must not silently inherit the
    usually much smaller ``action_trans_min/max`` range used for relative
    displacements.
    """
    config_norm = config.data.normalization
    return normalize_tcp(
        action,
        config,
        gripper_width_included=True,
        trans_min=config_norm.trans_min,
        trans_max=config_norm.trans_max,
    )

def action_translation_scale(config):
    """Per-axis scale that maps a raw action translation (meters) into the
    diffusion model's normalized [-1, 1] action space.

    For symmetric ``action_trans_min/max`` (e.g. ``[-0.2, 0.2]``) this is simply
    ``action_trans_max`` per axis. Falls back to ``trans_min/max`` for legacy
    configs, mirroring :func:`unnormalize_action`.

    The fusion-2.1 inter-TCP noisy-sample conversion in
    :mod:`policy.diffusion_eval` consumes this value to rescale the
    inter-frame displacement ``b_p`` from meters into the same units as the
    noisy sample. That conversion further requires the normalization to be
    **centred at zero** and **isotropic** across the three axes (otherwise the
    rotation ``Q`` would mix per-axis scales after the change of basis), so we
    enforce both conditions here and fail loudly with an actionable message.
    """
    config_norm = config.data.normalization
    if hasattr(config_norm, "action_trans_min") and hasattr(config_norm, "action_trans_max"):
        trans_min = np.asarray(config_norm.action_trans_min, dtype=np.float64)
        trans_max = np.asarray(config_norm.action_trans_max, dtype=np.float64)
    else:
        trans_min = np.asarray(config_norm.trans_min, dtype=np.float64)
        trans_max = np.asarray(config_norm.trans_max, dtype=np.float64)
    center = (trans_max + trans_min) / 2.0
    scale = (trans_max - trans_min) / 2.0
    if not np.allclose(center, 0.0, atol=1e-6):
        raise ValueError(
            "Fusion 2.1 noisy-sample conversion currently assumes a symmetric "
            "action_trans normalization (center == 0); got center={}.".format(center)
        )
    if not np.allclose(scale, scale[0], atol=1e-6):
        raise ValueError(
            "Fusion 2.1 noisy-sample conversion currently assumes isotropic "
            "action translation scale; got per-axis scale={}.".format(scale)
        )
    if np.any(scale <= 0):
        raise ValueError("Non-positive action translation scale: {}".format(scale))
    return scale


def unnormalize_action(action, config, trans_min = None, trans_max = None, max_gripper_width = None):
    action = np.asarray(action).copy()
    config_norm = config.data.normalization
    if trans_min is None:
        # Try to use action-specific normalization, fallback to default
        if hasattr(config_norm, 'action_trans_min'):
            trans_min = np.array(config_norm.action_trans_min)
        else:
            trans_min = np.array(config_norm.trans_min)
    else:
        trans_min = np.array(trans_min)
    if trans_max is None:
        if hasattr(config_norm, 'action_trans_max'):
            trans_max = np.array(config_norm.action_trans_max)
        else:
            trans_max = np.array(config_norm.trans_max)
    else:
        trans_max = np.array(trans_max)
    if max_gripper_width is None:
        max_gripper_width = config_norm.max_gripper_width
    action[..., :3] = (action[..., :3] + 1) / 2.0 * (trans_max - trans_min) + trans_min
    action[..., -1] = (action[..., -1] + 1) / 2.0 * max_gripper_width
    return action


def normalize_force_torque(force_torque_list, config):
    # float32: matches typical policy weights and torch.from_numpy without extra .float()
    force_torque_list = np.asarray(force_torque_list, dtype=np.float32).copy()
    config_norm = config.data.normalization
    force_min = np.array(config_norm.force_min)
    force_max = np.array(config_norm.force_max)
    torque_min = np.array(config_norm.torque_min)
    torque_max = np.array(config_norm.torque_max)
    force_torque_list[:, :3] = (force_torque_list[:, :3] - force_min) / (force_max - force_min) * 2 - 1
    force_torque_list[:, 3:] = (force_torque_list[:, 3:] - torque_min) / (torque_max - torque_min) * 2 - 1
    return force_torque_list
