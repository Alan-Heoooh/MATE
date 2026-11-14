from typing import Optional
import torch
from policy.diffusion_modules.module_attr_mixin import ModuleAttrMixin


class LowdimMaskGenerator(ModuleAttrMixin):
    def __init__(self,
        action_dim, obs_dim,
        # obs mask setup
        max_n_obs_steps=2, 
        fix_obs_steps=True, 
        # action mask
        action_visible=False,
        # optional: do not supervise first hist_horizon action steps
        hist_horizon: int = 0,
        supervise_hist: bool = True,
        ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible
        self.hist_horizon = int(hist_horizon) if hist_horizon is not None else 0
        self.supervise_hist = supervise_hist

    @torch.no_grad()
    def forward(self, shape, seed=None, hist_clip_steps: Optional[torch.Tensor] = None):
        device = self.device
        B, T, D = shape
        assert D == (self.action_dim + self.obs_dim)

        # create all tensors on this device
        rng = torch.Generator(device=device)
        if seed is not None:
            rng = rng.manual_seed(seed)

        # generate dim mask
        dim_mask = torch.zeros(size=shape, 
            dtype=torch.bool, device=device)
        is_action_dim = dim_mask.clone()
        is_action_dim[...,:self.action_dim] = True
        is_obs_dim = ~is_action_dim

        # generate obs mask
        if self.fix_obs_steps:
            obs_steps = torch.full((B,), 
            fill_value=self.max_n_obs_steps, device=device)
        else:
            obs_steps = torch.randint(
                low=1, high=self.max_n_obs_steps+1, 
                size=(B,), generator=rng, device=device)
            
        steps = torch.arange(0, T, device=device).reshape(1,T).expand(B,T)
        obs_mask = (steps.T < obs_steps).T.reshape(B,T,1).expand(B,T,D)
        obs_mask = obs_mask & is_obs_dim

        # generate action mask
        if self.action_visible:
            action_steps = torch.maximum(
                obs_steps - 1, 
                torch.tensor(0,
                    dtype=obs_steps.dtype, 
                    device=obs_steps.device))
            action_mask = (steps.T < action_steps).T.reshape(B,T,1).expand(B,T,D)
            action_mask = action_mask & is_action_dim
        mask = obs_mask
        if self.action_visible:
            mask = mask | action_mask

        # Optionally mark the first `hist_horizon` action steps as "conditioned"
        # so they are excluded from diffusion loss (useful when those actions
        # have been clipped to zeros, e.g. before base_action).
        if self.hist_horizon > 0 and not self.supervise_hist:
            if hist_clip_steps is None:
                hist_len = min(self.hist_horizon, T)
                hist_steps = (steps < hist_len).reshape(B, T, 1).expand(B, T, D)
            else:
                if not torch.is_tensor(hist_clip_steps):
                    hist_clip_steps = torch.tensor(hist_clip_steps, device=device, dtype=torch.long)
                else:
                    hist_clip_steps = hist_clip_steps.to(device=device, dtype=torch.long)
                if hist_clip_steps.dim() == 0:
                    hist_clip_steps = hist_clip_steps.unsqueeze(0).expand(B)
                assert hist_clip_steps.shape[0] == B, "hist_clip_steps batch size mismatch"
                hist_clip_steps = torch.clamp(hist_clip_steps, min=0, max=T)
                hist_steps = (steps < hist_clip_steps.view(B, 1)).reshape(B, T, 1).expand(B, T, D)

            hist_action_mask = hist_steps & is_action_dim
            mask = mask | hist_action_mask
        
        return mask
