import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from policy.diffusion_modules.conditional_unet1d import ConditionalUnet1D
from policy.diffusion_modules.mask_generator import LowdimMaskGenerator
from policy.noise_transform import (
    convert_noise_10d_t1_to_t2,
    convert_noise_10d_t2_to_t1,
    convert_noisy_sample_10d_t1_to_t2,
)

class DiffusionUNetPolicy_Eval(nn.Module):
    def __init__(self,
            action_dim,
            num_modules,
            horizon, 
            n_obs_steps,
            obs_feature_dim:list,
            num_inference_steps=20,
            diffusion_step_embed_dim=[256, 128],
            down_dims=[(256,512), (32,64)],
            modal_weights=None,
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # create diffusion model
        input_dim = action_dim
        models = []
        for i in range(num_modules):
            global_cond_dim = obs_feature_dim[i] * n_obs_steps
            model = ConditionalUnet1D(
                input_dim=input_dim,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim[i],
                down_dims=down_dims[i],
                kernel_size=kernel_size,
                n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            )
            models.append(model)
        self.models = nn.ModuleList(models)

        # create noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon"
        )

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

        self.vision_index = 0
        self.ft_index = 1

        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        if modal_weights is None:
            modal_weights = [1.0] * num_modules
        self.modal_weights = nn.Parameter(torch.tensor(modal_weights), requires_grad=False)

    # ========= inference  ============
        # return trajectory_ft[:, overlap_start:overlap_end]
        # return trajectory_ft

    def conditional_sample_with_ft_and_current_vision_noise(
            self,
            condition_data,
            condition_mask,
            vision_local_cond=None,
            vision_global_cond=None,
            ft_local_cond=None,
            ft_global_cond=None,
            noise_rotation_t1=None,
            noise_rotation_t2=None,
            noise_position_t1=None,
            noise_position_t2=None,
            vision_ft_offset_steps: int = 0,
            ft_horizon: int = 50,
            position_scale=None,
            rotation_repr: str = "rotation_col_6d",
            fusion_weights=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs,
        ):
        models = self.models
        scheduler = self.noise_scheduler
        if len(models) < 2:
            raise ValueError(
                "Need at least two modalities (vision + ft) to use "
                "`conditional_sample_with_ft_and_current_vision_noise`."
            )
        if (
            noise_rotation_t1 is None
            or noise_rotation_t2 is None
            or noise_position_t1 is None
            or noise_position_t2 is None
        ):
            raise ValueError(
                "Current vision fusion requires t1/t2 rotations and positions "
                "for noisy sample coordinate conversion."
            )
        if position_scale is None:
            raise ValueError(
                "Current vision fusion requires `position_scale` (action "
                "translation normalization scale, typically "
                "`(action_trans_max - action_trans_min) / 2`) so that the "
                "displacement term b_p is converted to normalized action "
                "units. Passing raw meters here under-corrects the inter-TCP "
                "displacement and degrades both position and rotation outputs."
            )

        device = condition_data.device
        dtype = condition_data.dtype
        B = condition_data.shape[0]
        T = condition_data.shape[1]
        Da = condition_data.shape[2]
        vision_ft_offset_steps = int(vision_ft_offset_steps)
        ft_horizon = int(ft_horizon)

        if vision_ft_offset_steps < 0:
            raise ValueError("vision_ft_offset_steps must be non-negative.")
        if ft_horizon <= 0:
            raise ValueError("ft_horizon must be positive.")
        if vision_ft_offset_steps + ft_horizon > T:
            raise ValueError(
                "Vision/FT overlap exceeds vision horizon: "
                f"offset={vision_ft_offset_steps}, ft_horizon={ft_horizon}, T={T}."
            )

        noise_rotation_t1 = torch.as_tensor(noise_rotation_t1, device=device, dtype=dtype)
        noise_rotation_t2 = torch.as_tensor(noise_rotation_t2, device=device, dtype=dtype)
        noise_position_t1 = torch.as_tensor(noise_position_t1, device=device, dtype=dtype)
        noise_position_t2 = torch.as_tensor(noise_position_t2, device=device, dtype=dtype)

        vision_latent = torch.randn(
            size=(B, T, Da),
            device=device,
            dtype=dtype,
            generator=generator,
        )

        scheduler.set_timesteps(self.num_inference_steps)
        first_t = scheduler.timesteps[0]
        alpha_bar_t = scheduler.alphas_cumprod[first_t].to(device=device, dtype=dtype)
        overlap_slice = slice(vision_ft_offset_steps, vision_ft_offset_steps + ft_horizon)

        if fusion_weights is None:
            active_weights_tensor = torch.stack([
                self.modal_weights[self.vision_index],
                self.modal_weights[self.ft_index],
            ])
        else:
            active_weights_tensor = torch.as_tensor(
                fusion_weights, device=device, dtype=dtype
            )
        if active_weights_tensor.ndim == 1:
            if active_weights_tensor.shape != (2,):
                raise ValueError(
                    "fusion_weights must have shape (2,) or (2, horizon); got {}.".format(
                        tuple(active_weights_tensor.shape)
                    )
                )
            active_weights_tensor = active_weights_tensor[:, None]
        elif active_weights_tensor.ndim == 2:
            if active_weights_tensor.shape != (2, ft_horizon):
                raise ValueError(
                    "Per-action fusion_weights must have shape (2, {}); got {}.".format(
                        ft_horizon, tuple(active_weights_tensor.shape)
                    )
                )
        else:
            raise ValueError(
                "fusion_weights must have shape (2,) or (2, horizon); got {}.".format(
                    tuple(active_weights_tensor.shape)
                )
            )
        if torch.any(active_weights_tensor < 0):
            raise ValueError("fusion_weights must be non-negative.")
        weight_sum = active_weights_tensor.sum(dim=0, keepdim=True)
        if torch.any(weight_sum <= 0):
            raise ValueError("At least one modality weight must be positive at every action index.")
        # (2, H) -> (2, 1, H, 1), which broadcasts over batch and action dims.
        normalized_weights = (active_weights_tensor / weight_sum)[:, None, :, None]

        for t in scheduler.timesteps:
            alpha_bar_t = scheduler.alphas_cumprod[t].to(device=device, dtype=dtype)

            # convert noisy sample from t1 to t2
            trajectory = convert_noisy_sample_10d_t1_to_t2(
                vision_latent[:, overlap_slice],
                noise_rotation_t1,
                noise_rotation_t2,
                noise_position_t1,
                noise_position_t2,
                alpha_bar_t,
                position_scale=position_scale,
                rotation_repr=rotation_repr,
            )

            # predict ft noise
            ft_noise = models[self.ft_index](
                trajectory,
                t,
                local_cond=ft_local_cond,
                global_cond=ft_global_cond,
            )
            
            # predict vision noise
            vision_query = vision_latent.clone()
            vision_noise_full_t1 = models[self.vision_index](
                vision_query,
                t,
                local_cond=vision_local_cond,
                global_cond=vision_global_cond,
            )
            # convert vision noise from t1 to t2
            vision_noise_t1 = vision_noise_full_t1[:, overlap_slice]
            vision_noise_t2 = convert_noise_10d_t1_to_t2(
                vision_noise_t1,
                noise_rotation_t1,
                noise_rotation_t2,
                rotation_repr=rotation_repr
            )

            # fuse vision and ft noise
            pred_t2 = (
                normalized_weights[0] * vision_noise_t2
                + normalized_weights[1] * ft_noise
            )

            # convert fused noise from t2 to t1
            pred_t1_overlap = convert_noise_10d_t2_to_t1(
                pred_t2,
                noise_rotation_t1,
                noise_rotation_t2,
                rotation_repr=rotation_repr,
            )
            pred_t1_full = vision_noise_full_t1.clone()
            pred_t1_full[:, overlap_slice] = pred_t1_overlap

            # update vision latent
            vision_latent = scheduler.step(
                pred_t1_full, t, vision_query, generator=generator, **kwargs
            ).prev_sample

        # return overlap window
        trajectory = vision_latent[:, overlap_slice]
        return trajectory


    def predict_action_with_ft_and_current_vision_noise(
            self,
            vision_readout,
            ft_readout,
            noise_rotation_t1=None,
            noise_rotation_t2=None,
            noise_position_t1=None,
            noise_position_t2=None,
            vision_ft_offset_steps: int = 0,
            ft_horizon: int = 50,
            position_scale=None,
            rotation_repr: str = "rotation_col_6d",
            fusion_weights=None,
        ):
        """
        Re-query the vision model at every denoising step using the current F/T
        sample transformed back into the vision coordinate frame.
        """
        B = ft_readout.shape[0]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        device = ft_readout.device
        dtype = ft_readout.dtype
        assert vision_readout.shape[0] == B * To
        assert ft_readout.shape[0] == B * To

        vision_global_cond = vision_readout.reshape(B, -1)
        ft_global_cond = ft_readout.reshape(B, -1)
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        sample = self.conditional_sample_with_ft_and_current_vision_noise(
            cond_data,
            cond_mask,
            vision_local_cond=None,
            vision_global_cond=vision_global_cond,
            ft_local_cond=None,
            ft_global_cond=ft_global_cond,
            noise_rotation_t1=noise_rotation_t1,
            noise_rotation_t2=noise_rotation_t2,
            noise_position_t1=noise_position_t1,
            noise_position_t2=noise_position_t2,
            vision_ft_offset_steps=vision_ft_offset_steps,
            ft_horizon=ft_horizon,
            position_scale=position_scale,
            rotation_repr=rotation_repr,
            fusion_weights=fusion_weights,
            **self.kwargs,
        )

        action_pred = sample[..., :Da]
        return action_pred

    def predict_action_with_current_fast_guidance(
        self, slow_readout, fast_readout, *, slow_rotation, fast_rotation,
        slow_position, fast_position, offset_steps, fast_horizon,
        position_scale, rotation_repr, fusion_weights=None,
    ):
        """Compose current fast guidance with delayed guidance using TCP rebasing."""
        return self.predict_action_with_ft_and_current_vision_noise(
            vision_readout=slow_readout, ft_readout=fast_readout,
            noise_rotation_t1=slow_rotation, noise_rotation_t2=fast_rotation,
            noise_position_t1=slow_position, noise_position_t2=fast_position,
            vision_ft_offset_steps=offset_steps, ft_horizon=fast_horizon,
            position_scale=position_scale, rotation_repr=rotation_repr,
            fusion_weights=fusion_weights,
        )
