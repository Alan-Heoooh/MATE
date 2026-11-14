import yaml
from utils.runtime_config import expand, local_variables, resolve_path, apply_io_overrides
try:
    from easydict import EasyDict as edict  # type: ignore
except Exception:
    class edict(dict):
        """
        Minimal EasyDict-compatible fallback.
        Provides attribute access (cfg.a.b) for nested dicts and lists of dicts.
        """
        def __init__(self, mapping=None, **kwargs):
            super().__init__()
            mapping = {} if mapping is None else mapping
            if isinstance(mapping, dict):
                items = mapping.items()
            else:
                # best-effort: try to cast to dict
                items = dict(mapping).items()
            for k, v in items:
                self[k] = self._wrap(v)
            for k, v in kwargs.items():
                self[k] = self._wrap(v)

        @staticmethod
        def _wrap(v):
            if isinstance(v, dict):
                return edict(v)
            if isinstance(v, list):
                return [edict._wrap(x) for x in v]
            if isinstance(v, tuple):
                return tuple(edict._wrap(x) for x in v)
            return v

        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError as e:
                raise AttributeError(item) from e

        def __setattr__(self, key, value):
            self[key] = self._wrap(value)

        def __delattr__(self, item):
            try:
                del self[item]
            except KeyError as e:
                raise AttributeError(item) from e

def to_plain(obj):
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(x) for x in obj]
    return obj

def save_config(config, path):
    with open(path, "w") as f:
        yaml.safe_dump(to_plain(config), f, default_flow_style=False, sort_keys=False)

def load_config(path, resolve_local=True):
    values = local_variables() if resolve_local else {}
    with open(resolve_path(path, values), "r") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping: {}".format(path))
    raw = expand(raw, values)
    cfg = edict(apply_io_overrides(raw, values) if resolve_local else raw)
    if hasattr(cfg, "data"):
        if not hasattr(cfg.data, "action_rotation_repr"):
            cfg.data.action_rotation_repr = "rotation_6d"
        valid_reprs = {"rotation_6d", "rotation_col_6d"}
        if cfg.data.action_rotation_repr not in valid_reprs:
            raise ValueError(
                "Unsupported data.action_rotation_repr: {} (expected one of {}).".format(
                    cfg.data.action_rotation_repr, sorted(valid_reprs)
                )
            )
    return cfg


def get_action_rotation_repr(config):
    """Get action rotation representation with backward-compatible default."""
    action_rotation_repr = getattr(config.data, "action_rotation_repr", "rotation_6d")
    if action_rotation_repr not in {"rotation_6d", "rotation_col_6d"}:
        raise ValueError(
            "Unsupported data.action_rotation_repr: {} (expected 'rotation_6d' or 'rotation_col_6d').".format(
                action_rotation_repr
            )
        )
    return action_rotation_repr

def check_config(vision_config, ft_config):
    if vision_config.data.action_dim != ft_config.data.action_dim:
        raise ValueError("Action dims must be consistent between vision and ft")
    # if vision_config.data.num_action != ft_config.data.num_action:
    #     raise ValueError("Action horizon must be consistent between vision and ft")
    if vision_config.data.action_freq != ft_config.data.action_freq:
        raise ValueError("Action frequency must be consistent between vision and ft")
    if vision_config.data.relative_action != ft_config.data.relative_action:
        raise ValueError("Relative action must be consistent between vision and ft")
    if get_action_rotation_repr(vision_config) != get_action_rotation_repr(ft_config):
        raise ValueError(
            "action_rotation_repr must be consistent between vision and ft "
            "(got vision={}, ft={})".format(
                get_action_rotation_repr(vision_config),
                get_action_rotation_repr(ft_config),
            )
        )

def resolve_agent_config(vision_config, ft_config):
    agent_config = getattr(ft_config, "deploy", edict())
    if not hasattr(agent_config, "agent"):
        agent_config = getattr(vision_config, "deploy", edict())
    return agent_config
