"""Dataset field names are independent of the live robot's serial number."""
import numpy as np


def field_keys(config):
    configured = getattr(config.data, "hdf5_keys", {})
    keys = {name: configured.get(name, name) for name in (
        "timestamp", "tcp_pose", "ee_command", "force_torque",
    )}
    if not all(isinstance(value, str) and value for value in keys.values()):
        raise ValueError("data.hdf5_keys values must be non-empty strings.")
    return keys


def read_lowdim(group, config, require_force=True):
    keys = field_keys(config)
    fields = ["timestamp", "tcp_pose", "ee_command"]
    if require_force:
        fields.append("force_torque")
    missing = [keys[name] for name in fields if keys[name] not in group]
    if missing:
        raise KeyError("Missing HDF5 fields {}. Configure data.hdf5_keys; available fields: {}".format(
            missing, list(group.keys()),
        ))
    return {name: np.asarray(group[keys[name]]) for name in fields}
