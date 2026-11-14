from policy import RISE, FT_Proprio, RISE_FT_Proprio

POLICY_REGISTRY = {
    "rise": RISE,
    "force_proprio": FT_Proprio,
    "rise_ft_proprio_2d": RISE_FT_Proprio,
}

def make_policy(config, device, policy_type=None):
    policy_type = (config.model.type if policy_type is None else policy_type)
    try:
        PolicyCls = POLICY_REGISTRY[policy_type]
    except KeyError as e:
        valid = ", ".join(sorted(POLICY_REGISTRY.keys()))
        raise ValueError(f"Invalid policy type: {policy_type}. Valid: {valid}") from e

    return PolicyCls(config=config).to(device)