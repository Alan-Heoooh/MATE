"""Validation and horizon-weight scheduling for MATE TCP-relative composition."""

from pathlib import Path

import numpy as np

from utils.config import check_config, get_action_rotation_repr, load_config


_RELATIVE_SHARED_ACTION_NORMALIZATION_KEYS = (
    "action_trans_min",
    "action_trans_max",
    "max_gripper_width",
)



def action_horizon(config):
    future = getattr(config.data, "action_horizon_future", None)
    history = getattr(config.data, "action_horizon_hist", 0)
    if future is not None:
        return int(future) + int(history)
    return int(config.data.num_action)


def _assert_equal(name, lhs, rhs, *, atol=1e-8):
    lhs_array = np.asarray(lhs)
    rhs_array = np.asarray(rhs)
    same = lhs_array.shape == rhs_array.shape
    if same:
        lhs_numeric = np.issubdtype(lhs_array.dtype, np.number)
        rhs_numeric = np.issubdtype(rhs_array.dtype, np.number)
        if lhs_numeric and rhs_numeric:
            same = np.allclose(lhs_array, rhs_array, rtol=0.0, atol=atol)
        else:
            same = np.array_equal(lhs_array, rhs_array)
    if not same:
        raise ValueError(
            "MATE requires identical {}; got lhs={} and rhs={}.".format(
                name, lhs, rhs
            )
        )


def validate_mate_configs(
    slow_config,
    fast_config,
    *,
    slow_infer_hz=2.0,
    fast_infer_hz=5.0,
):
    """Validate the mathematical and temporal assumptions of MATE."""
    check_config(slow_config, fast_config)

    relative_action = bool(slow_config.data.relative_action)
    slow_relative_proprio = bool(
        slow_config.data.relative_proprio_obs_for_relative_action
    )
    fast_relative_proprio = bool(
        fast_config.data.relative_proprio_obs_for_relative_action
    )
    if relative_action:
        if not slow_relative_proprio or not fast_relative_proprio:
            raise ValueError(
                "Relative MATE requires "
                "relative_proprio_obs_for_relative_action=true for both policies."
            )
        if get_action_rotation_repr(slow_config) != "rotation_col_6d":
            raise ValueError(
                "Relative MATE diffusion rebasing requires "
                "action_rotation_repr=rotation_col_6d for both policies. "
                "Row-based rotation_6d cannot be rebased by the paper's "
                "orthogonal block transform in noise space."
            )
    else:
        raise ValueError("MATE requires relative_action=true; absolute-action controls are not included.")

    slow_norm = slow_config.data.normalization
    fast_norm = fast_config.data.normalization
    normalization_keys = _RELATIVE_SHARED_ACTION_NORMALIZATION_KEYS
    for key in normalization_keys:
        if not hasattr(slow_norm, key) or not hasattr(fast_norm, key):
            raise ValueError(
                "Both MATE configs must define normalization.{}.".format(key)
            )
        _assert_equal(
            "normalization.{}".format(key),
            getattr(slow_norm, key),
            getattr(fast_norm, key),
        )

    if relative_action:
        action_min = np.asarray(slow_norm.action_trans_min, dtype=np.float64)
        action_max = np.asarray(slow_norm.action_trans_max, dtype=np.float64)
        action_center = (action_min + action_max) / 2.0
        action_scale = (action_max - action_min) / 2.0
        if not np.allclose(action_center, 0.0, rtol=0.0, atol=1e-8):
            raise ValueError(
                "Relative MATE noisy-sample rebasing requires "
                "zero-centered action translation normalization; got "
                "center={}.".format(action_center)
            )
        if np.any(action_scale <= 0) or not np.allclose(
            action_scale, action_scale[0], rtol=0.0, atol=1e-8
        ):
            raise ValueError(
                "Relative MATE noisy-sample rebasing requires positive "
                "isotropic action translation normalization; got "
                "scale={}.".format(action_scale)
            )

    slow_infer_hz = float(slow_infer_hz)
    fast_infer_hz = float(fast_infer_hz)
    if slow_infer_hz <= 0 or fast_infer_hz <= 0:
        raise ValueError("Inference frequencies must be positive.")
    if fast_infer_hz <= slow_infer_hz:
        raise ValueError(
            "The fast policy frequency must exceed the slow policy frequency; "
            "got slow={} Hz, fast={} Hz.".format(slow_infer_hz, fast_infer_hz)
        )

    action_freq = float(slow_config.data.action_freq)
    max_nominal_offset = int(np.ceil(action_freq / slow_infer_hz))
    slow_horizon = action_horizon(slow_config)
    fast_horizon = action_horizon(fast_config)
    if max_nominal_offset + fast_horizon > slow_horizon:
        raise ValueError(
            "Slow action horizon is too short for asynchronous overlap: "
            "ceil(action_freq/slow_hz) + fast_horizon = {} + {} > {}.".format(
                max_nominal_offset, fast_horizon, slow_horizon
            )
        )

    return {
        "action_frame": "tcp_relative",
        "slow_horizon": slow_horizon,
        "fast_horizon": fast_horizon,
        "action_freq": action_freq,
        "max_nominal_offset": max_nominal_offset,
    }


def latency_aware_horizon_weights(
    horizon,
    *,
    slow_weight_start=0.8,
    slow_weight_end=0.2,
    max_index=None,
):
    """Return paper-style per-action weights with shape ``(2, horizon)``.

    Row 0 is the delayed slow-policy weight and row 1 is its complementary
    fast-policy weight.  The schedule varies along the selected action horizon,
    not with the age of the cached slow observation.
    """
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if max_index is None:
        max_index = max(1, horizon - 1)
    max_index = int(max_index)
    if max_index <= 0:
        raise ValueError("max_index must be positive.")

    slow_weight_start = float(slow_weight_start)
    slow_weight_end = float(slow_weight_end)
    if not 0.0 <= slow_weight_start <= 1.0 or not 0.0 <= slow_weight_end <= 1.0:
        raise ValueError("Latency-aware weights must lie in [0, 1].")

    indices = np.arange(horizon, dtype=np.float32)
    ratio = np.clip(indices / float(max_index), 0.0, 1.0)
    slow = slow_weight_start + ratio * (slow_weight_end - slow_weight_start)
    fast = 1.0 - slow
    return np.stack([slow, fast], axis=0).astype(np.float32)


def observation_offset_steps(slow_time, fast_time, action_freq):
    delta = float(fast_time) - float(slow_time)
    if delta < -1e-6:
        raise ValueError(
            "Slow observation timestamp ({:.6f}) is newer than fast timestamp "
            "({:.6f}).".format(float(slow_time), float(fast_time))
        )
    return max(0, int(round(max(0.0, delta) * float(action_freq))))


def validate_checkpoint_training_config(checkpoint_path, runtime_config, label):
    """Verify a checkpoint's saved ``hpara.yaml`` against its eval config.

    Returns the saved path when present.  Older checkpoints without a saved
    config are allowed but must be called out by the deployment script.
    """
    saved_path = Path(checkpoint_path).expanduser().resolve().parent / "hpara.yaml"
    if not saved_path.is_file():
        return None

    saved = load_config(str(saved_path), resolve_local=False)
    scalar_fields = (
        "action_dim",
        "action_freq",
        "relative_action",
        "relative_proprio_obs_for_relative_action",
        "action_rotation_repr",
    )
    for field in scalar_fields:
        _assert_equal(
            "{} checkpoint data.{}".format(label, field),
            getattr(saved.data, field),
            getattr(runtime_config.data, field),
        )
    _assert_equal(
        "{} checkpoint action horizon".format(label),
        action_horizon(saved),
        action_horizon(runtime_config),
    )
    normalization_keys = _RELATIVE_SHARED_ACTION_NORMALIZATION_KEYS
    for key in normalization_keys:
        _assert_equal(
            "{} checkpoint normalization.{}".format(label, key),
            getattr(saved.data.normalization, key),
            getattr(runtime_config.data.normalization, key),
        )
    return saved_path
