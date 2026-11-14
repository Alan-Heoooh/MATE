from policy.diffusion_eval import DiffusionUNetPolicy_Eval

def infer_obs_feature_dim(config):
    obs_feature_dim = getattr(config.model, "obs_feature_dim", None)
    if obs_feature_dim is not None:
        return obs_feature_dim

    obs_feature_dims = getattr(config.model, "obs_feature_dims", None)
    if obs_feature_dims is not None:
        return sum(obs_feature_dims.values())

    return getattr(config.model, "ft_feature_dim", None)    


def build_eval_action_head(
    vision_policy,
    ft_policy,
    vision_config,
    ft_config,
    modal_weights,
    device,
    num_inference_steps=10,
):
    vision_obs_feature_dim = infer_obs_feature_dim(vision_config)
    ft_obs_feature_dim = infer_obs_feature_dim(ft_config)
    if vision_obs_feature_dim is None or ft_obs_feature_dim is None:
        raise ValueError("Failed to infer obs_feature_dim for vision or ft from configs.")

    action_dim = ft_config.data.action_dim
    # Support both new and legacy action horizon config names.
    # New: predict full horizon = history + future.
    # Legacy: predict horizon = num_action.
    def _get_horizon(cfg):
        future = getattr(cfg.data, "action_horizon_future", None)
        hist = getattr(cfg.data, "action_horizon_hist", None)
        if future is not None:
            if hist is None:
                hist = 0
            return int(future) + int(hist)
        legacy = getattr(cfg.data, "num_action", None)
        if legacy is None:
            raise ValueError(
                "Missing action horizon in config.data: expected "
                "`action_horizon_future` (+ optional `action_horizon_hist`) or `num_action`."
            )
        return int(legacy)

    horizon_vision = _get_horizon(vision_config)
    horizon_ft = _get_horizon(ft_config)
    # if horizon_vision != horizon_ft:
    #     raise ValueError(f"Action horizon mismatch between vision and ft: {horizon_vision} vs {horizon_ft}")
    horizon = horizon_vision

    vd = vision_config.model.action_decoder
    fd = ft_config.model.action_decoder
    if vd.type != "diffusion_unet" or fd.type != "diffusion_unet":
        raise ValueError(
            "MATE currently requires diffusion_unet action decoders; "
            "got vision={} and second={}.".format(vd.type, fd.type)
        )
    num_inference_steps = int(num_inference_steps)
    if num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive.")

    eval_action_head = DiffusionUNetPolicy_Eval(
        action_dim=action_dim,
        num_modules=2,
        horizon=horizon,
        n_obs_steps=1,
        obs_feature_dim=[vision_obs_feature_dim, ft_obs_feature_dim],
        num_inference_steps=num_inference_steps,
        diffusion_step_embed_dim=[vd.diffusion_step_embed_dim, fd.diffusion_step_embed_dim],
        down_dims=[vd.down_dims, fd.down_dims],
        modal_weights=modal_weights,  # (vision, ft)
    ).to(device)

    # vision policy's diffusion head -> eval.models[0]
    eval_action_head.models[0].load_state_dict(
        vision_policy.action_decoder.model.state_dict()
    )
    # ft policy's diffusion head -> eval.models[1]
    eval_action_head.models[1].load_state_dict(
        ft_policy.action_decoder.model.state_dict()
    )
    return eval_action_head


def build_mate_action_head(
    slow_policy,
    fast_policy,
    slow_config,
    fast_config,
    modal_weights,
    device,
    num_inference_steps=10,
):
    """Build the shared sampler for any slow/fast pair of diffusion policies."""
    return build_eval_action_head(
        vision_policy=slow_policy,
        ft_policy=fast_policy,
        vision_config=slow_config,
        ft_config=fast_config,
        modal_weights=modal_weights,
        device=device,
        num_inference_steps=num_inference_steps,
    )
