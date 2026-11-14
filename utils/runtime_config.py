"""Local machine settings without shell execution or model-parameter overrides."""
import copy
import os
from pathlib import Path
import re

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_VARIABLE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)(?::-([^}]*))?\}")


def local_variables(local_path=None):
    """Read only the explicit variables mapping; never export values to os.environ.

    An empty MATE_LOCAL_CONFIG disables automatic local configuration, useful for
    inspecting the exact settings included in a release.
    """
    if local_path is None:
        local_path = os.environ.get("MATE_LOCAL_CONFIG", "configs/local.yaml")
    if not local_path:
        return {}
    path = Path(local_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        if "MATE_LOCAL_CONFIG" in os.environ or str(local_path) != "configs/local.yaml":
            raise FileNotFoundError("Local configuration does not exist: {}".format(path))
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict) or set(raw) - {"variables"}:
        raise ValueError("Local configuration must contain only a 'variables' mapping.")
    values = raw.get("variables", {})
    if not isinstance(values, dict):
        raise ValueError("Local configuration 'variables' must be a mapping.")
    return values


def variable(name, default=None, values=None):
    values = local_variables() if values is None else values
    return os.environ[name] if name in os.environ else values.get(name, default)


def expand(value, values=None):
    """Expand ${NAME} and ${NAME:-default}; preserve identifier strings verbatim."""
    values = local_variables() if values is None else values
    if isinstance(value, dict):
        return {key: expand(item, values) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item, values) for item in value]
    if not isinstance(value, str):
        return value

    def resolve(match):
        name, fallback = match.groups()
        result = variable(name, None, values)
        if result is None or result == "":
            if fallback is None:
                raise ValueError("Set {} in the environment or configs/local.yaml.".format(name))
            result = fallback
        # Lists/mappings from YAML stay typed; environment vectors may use JSON/YAML.
        if isinstance(result, str) and result.strip().startswith(("[", "{")):
            result = yaml.safe_load(result)
        return copy.deepcopy(result)

    full = _VARIABLE.fullmatch(value)
    if full:
        return resolve(full)
    return _VARIABLE.sub(lambda match: str(resolve(match)), value)


def resolve_path(value, values=None):
    path = Path(str(expand(str(value), values))).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def apply_io_overrides(raw, values):
    """Apply explicit local I/O settings to legacy checkpoint YAML as well.

    Model structure, normalization, action representation, and horizon are never
    overridden here: checkpoint compatibility checks must see their saved values.
    """
    data = raw.get("data", {})
    wrist = bool(data.get("inhand_camera", False))
    agent = raw.get("deploy", {}).get("agent")
    mappings = {
        "robot_serial": "MATE_ROBOT_SERIAL",
        "camera_serial": "MATE_WRIST_CAMERA_SERIAL" if wrist else "MATE_CAMERA_SERIAL",
        "ready_pose": "MATE_READY_POSE", "ready_pose_joints": "MATE_READY_JOINTS",
        "camera_intrinsics": "MATE_CAMERA_INTRINSICS",
        "robot_interface_module": "MATE_ROBOT_INTERFACE_MODULE",
    }
    if isinstance(agent, dict):
        for key, name in mappings.items():
            if name in os.environ or name in values:
                agent[key] = expand("${" + name + ":-}", values)
    for key, name in {
        "cam_id": "MATE_DATA_WRIST_CAMERA_ID" if wrist else "MATE_DATA_CAMERA_ID",
    }.items():
        if name in os.environ or name in values:
            data[key] = expand("${" + name + ":-}", values)
    for field, name in {
        "tcp_pose": "MATE_TCP_POSE_KEY", "ee_command": "MATE_EE_COMMAND_KEY",
        "force_torque": "MATE_FORCE_TORQUE_KEY", "timestamp": "MATE_TIMESTAMP_KEY",
    }.items():
        if name in os.environ or name in values:
            data.setdefault("hdf5_keys", {})[field] = expand("${" + name + "}", values)
    return raw


def validate_device_settings(settings):
    """Validate required machine-specific values before any device is enabled."""
    import math
    for key in ("robot_serial", "camera_serial"):
        value = settings.get(key)
        if not isinstance(value, str) or not value.strip() or any(
            marker in value.upper() for marker in ("YOUR_", "REPLACE_", "<", ">")
        ):
            raise ValueError("Set deploy.agent.{} in your local configuration.".format(key))
    for key in ("ready_pose", "ready_pose_joints"):
        values = settings.get(key)
        if not isinstance(values, (list, tuple)) or len(values) != 7 or not all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in values
        ):
            raise ValueError("deploy.agent.{} must contain seven finite values.".format(key))
    quat = settings["ready_pose"][3:]
    if not math.isclose(sum(float(x) ** 2 for x in quat), 1.0, abs_tol=1e-3):
        raise ValueError("ready_pose must use a unit wxyz quaternion.")
    intrinsics = settings.get("camera_intrinsics")
    if intrinsics not in (None, ""):
        if not isinstance(intrinsics, (list, tuple)) or len(intrinsics) != 3 or any(
            not isinstance(row, (list, tuple)) or len(row) != 3 for row in intrinsics
        ):
            raise ValueError("camera_intrinsics must be a 3x3 matrix or null.")
        if not all(isinstance(x, (int, float)) and math.isfinite(x) for row in intrinsics for x in row):
            raise ValueError("camera_intrinsics must be finite.")
        if intrinsics[0][0] <= 0 or intrinsics[1][1] <= 0:
            raise ValueError("camera_intrinsics focal lengths must be positive.")
