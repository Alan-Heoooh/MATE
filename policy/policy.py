import torch
import torch.nn as nn
import inspect

from policy.tokenizer import Sparse3DEncoder
from policy.transformer import Transformer

from policy.encoder import get_encoder
from policy.action_decoder import get_action_decoder

def get_action_horizon_from_config(config):
    """Support both new and legacy action horizon config names."""
    future = getattr(config.data, "action_horizon_future", None)
    hist = getattr(config.data, "action_horizon_hist", None)

    # New config: predict full horizon = history + future.
    if future is not None:
        if hist is None:
            hist = 0
        return int(future) + int(hist)

    # Legacy config fallback.
    horizon = getattr(config.data, "num_action", None)
    if horizon is None:
        raise ValueError(
            "Missing action horizon in config.data: expected "
            "action_horizon_future (+ optional action_horizon_hist) or num_action."
        )
    return int(horizon)

def get_hist_horizon_from_config(config):
    hist = getattr(config.data, "action_horizon_hist", None)
    if hist is None:
        hist = getattr(config.data, "num_action_hist", 0)
    return int(hist)

class RISE(nn.Module):
    def __init__(
        self, 
        config,
    ):
        super().__init__()
        num_obs = 1
        self._parse_config(config)
        self.sparse_encoder = Sparse3DEncoder(self.input_dim, self.obs_feature_dim)
        self.transformer = Transformer(
            self.hidden_dim,
            self.nheads,
            self.num_encoder_layers,
            self.num_decoder_layers,
            self.dim_feedforward,
            self.dropout,
        )
        decoder_kwargs = {}
        self.action_decoder = get_action_decoder(self.action_decoder_type)(
            self.action_dim,
            self.num_action,
            n_obs_steps=num_obs,
            obs_feature_dim=self.obs_feature_dim,
            num_inference_steps=self.num_inference_steps,
            diffusion_step_embed_dim=self.diffusion_step_embed_dim,
            down_dims=self.down_dims,
            kernel_size=self.kernel_size,
            n_groups=self.n_groups,
            **decoder_kwargs,
        )
        self.readout_embed = nn.Embedding(1, self.hidden_dim)

    def _parse_config(self, config):
        self.input_dim = config.model.input_dim
        self.obs_feature_dim = config.model.obs_feature_dim
        self.num_action = get_action_horizon_from_config(config)
        self.action_dim = config.data.action_dim

        # transformer
        self.hidden_dim = config.model.hidden_dim
        self.nheads = config.model.nheads
        self.num_encoder_layers = config.model.num_encoder_layers
        self.num_decoder_layers = config.model.num_decoder_layers
        self.dim_feedforward = config.model.dim_feedforward
        self.dropout = config.model.dropout

        # action decoder
        self.action_decoder_config = config.model.action_decoder
        self.action_decoder_type = self.action_decoder_config.type
        self.num_inference_steps = self.action_decoder_config.num_inference_steps
        self.diffusion_step_embed_dim = self.action_decoder_config.diffusion_step_embed_dim
        self.down_dims = self.action_decoder_config.down_dims
        self.kernel_size = self.action_decoder_config.kernel_size
        self.n_groups = self.action_decoder_config.n_groups

    def encode(self, cloud, batch_size=24):
        src, pos, src_padding_mask = self.sparse_encoder(cloud, batch_size=batch_size)
        readout_tokens = self.transformer(
            src, src_padding_mask,
            self.readout_embed.weight, pos
        )[-1]                       # (B, 1+N, hidden_dim)
        readout = readout_tokens[:, 0]  # (B, hidden_dim) or obs_feature_dim
        return readout

    def forward(self, cloud, actions = None, batch_size = 24):
        readout = self.encode(cloud, batch_size=batch_size)
        if actions is not None:
            loss = self.action_decoder.compute_loss(readout, actions)
            return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(readout)
            return action_pred

class FT_Proprio(nn.Module):
    def __init__(
        self, 
        config,
    ):
        super().__init__()
        
        self._parse_config(config)

        # dropout applied on the fused feature representation
        self.fusion_dropout = nn.Dropout(self.fusion_dropout_p)

        if self.fusion_type == "feature_concat":
            # Support force-torque only mode when proprio_feature_dim is 0
            if self.proprio_encoder.output_dim == 0:
                assert self.ft_encoder.output_dim == self.obs_feature_dim, \
                    f"When proprio_feature_dim=0, obs_feature_dim must equal ft_feature_dim. " \
                    f"Got ft_feature_dim={self.ft_encoder.output_dim}, obs_feature_dim={self.obs_feature_dim}"
                self.ft_encoder = get_encoder(self.ft_encoder)
                self.proprio_encoder = None  # Not used in force-torque only mode
            else:
                assert self.ft_encoder.output_dim + self.proprio_encoder.output_dim == self.obs_feature_dim
                self.ft_encoder = get_encoder(self.ft_encoder)
                self.proprio_encoder = get_encoder(self.proprio_encoder)
        else:
            raise ValueError(f"Invalid fusion_type: {self.fusion_type}")

        decoder_kwargs = {}
        self.action_decoder = get_action_decoder(self.action_decoder_config.type)(
            self.action_dim, 
            self.num_action, 
            n_obs_steps=1, 
            obs_feature_dim=self.obs_feature_dim, 
            num_inference_steps=self.action_decoder_config.num_inference_steps, 
            diffusion_step_embed_dim=self.action_decoder_config.diffusion_step_embed_dim, 
            down_dims=self.action_decoder_config.down_dims, 
            kernel_size=self.action_decoder_config.kernel_size, 
            n_groups=self.action_decoder_config.n_groups, 
            hist_horizon_for_loss=self.action_horizon_hist,
            supervise_hist_actions=self.supervise_hist_actions,
            **decoder_kwargs,
        )


    def _parse_config(self, config):
        self.ft_encoder = config.model.ft_encoder
        self.proprio_encoder = config.model.proprio_encoder
        self.fusion_type = config.model.fusion_type
        # probability for dropout on fused feature
        self.fusion_dropout_p = getattr(config.model, "fusion_dropout", 0.0)
        self.obs_feature_dim = config.model.obs_feature_dim
        self.num_action = get_action_horizon_from_config(config)
        self.action_horizon_hist = get_hist_horizon_from_config(config)
        self.action_dim = config.data.action_dim

        self.action_decoder_config = config.model.action_decoder
        # supervision strategy for base-action history segment
        self.supervise_hist_actions = getattr(
            self.action_decoder_config,
            "supervise_hist_actions",
            getattr(config.data, "supervise_hist_actions", True),
        )

        if self.proprio_encoder is not None:
            print(f"[FT_Proprio] fusion_type = {self.fusion_type}, "
                  f"ft_encoder = {self.ft_encoder.type}, proprio_encoder = {self.proprio_encoder.type}")
        else:
            print(f"[FT_Proprio] fusion_type = {self.fusion_type}, "
                  f"ft_encoder = {self.ft_encoder.type}, force-torque only mode (no proprio_encoder)")


    def encode(self, force_torque, proprio=None):
        """
        force_torque: (B, num_obs_ft, input_dim_ft)
        proprio:          (B, num_obs_proprio, input_dim_proprio) or None
        return:       (B, obs_feature_dim)
        """
        B = force_torque.size(0)

        if self.fusion_type == "feature_concat":
            # --- ft feature ---
            ft_feature = self.ft_encoder(force_torque)     # (B, ft_feature_dim)

            # --- proprio feature ---
            if self.proprio_encoder is None:
                # Force-torque only mode
                feature = ft_feature                       # (B, ft_feature_dim = obs_feature_dim)
            elif proprio is not None:
                proprio_feature = self.proprio_encoder(proprio)    # (B, proprio_feature_dim)
                feature = torch.cat(
                    [ft_feature, proprio_feature], dim=-1
                )                                           # (B, ft+proprio = obs_feature_dim)
            else:
                # Proprio encoder exists but proprio is None - pad with zeros to match obs_feature_dim
                proprio_feature_dim = self.proprio_encoder.output_dim
                device = ft_feature.device
                proprio_feature = torch.zeros(B, proprio_feature_dim, device=device)
                feature = torch.cat([ft_feature, proprio_feature], dim=-1)  # (B, obs_feature_dim)
        else:
            raise ValueError(f"Invalid fusion_type: {self.fusion_type}")

        # apply dropout on the fused feature representation
        feature = self.fusion_dropout(feature)

        return feature   # (B, obs_feature_dim)

    def forward(self, force_torque, proprio=None, actions = None, batch_size = 24, base_action_hist_clip_steps=None):
        """
        force_torque: (B, num_obs_ft, input_dim_ft)
        proprio: (B, num_obs_proprio, input_dim_proprio) or None
        """
        feature = self.encode(force_torque, proprio) # (B, feature_dim)

        if actions is not None:
            loss = self.action_decoder.compute_loss(feature, actions, hist_clip_steps=base_action_hist_clip_steps)
            return loss
        else:
            with torch.no_grad():
               action_pred = self.action_decoder.predict_action(feature, hist_clip_steps=base_action_hist_clip_steps)
            return action_pred

class RISE_FT_Proprio(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._parse_config(config)
        if self.ft_feature_dim != 0 or self.proprio_feature_dim != 0:
            raise ValueError("The MATE RGB component accepts vision only.")
        self.fusion_dropout = nn.Dropout(self.fusion_dropout_p)
        self.vision_encoder_cfg.output_dim = self.vision_feature_dim
        self.vision_encoder = get_encoder(self.vision_encoder_cfg)
        self.ft_encoder = None
        self.proprio_encoder = None
        self.obs_feature_dim = int(self.vision_feature_dim)
        self.action_decoder = get_action_decoder(self.action_decoder_config.type)(
            self.action_dim, self.num_action, n_obs_steps=1,
            obs_feature_dim=self.obs_feature_dim,
            num_inference_steps=self.action_decoder_config.num_inference_steps,
            diffusion_step_embed_dim=self.action_decoder_config.diffusion_step_embed_dim,
            down_dims=self.action_decoder_config.down_dims,
            kernel_size=self.action_decoder_config.kernel_size,
            n_groups=self.action_decoder_config.n_groups,
            hist_horizon_for_loss=self.action_horizon_hist,
            supervise_hist_actions=self.supervise_hist_actions,
        )


    def _parse_config(self, config):
        self.fusion_type = config.model.fusion_type
        # probability for dropout on fused feature
        self.fusion_dropout_p = getattr(config.model, "fusion_dropout", 0.0)
        self.obs_feature_dims = config.model.obs_feature_dims
        self.num_action = get_action_horizon_from_config(config)
        self.action_horizon_hist = get_hist_horizon_from_config(config)
        self.action_dim = config.data.action_dim

        self.action_decoder_config = config.model.action_decoder
        # supervision strategy for base-action history segment
        self.supervise_hist_actions = getattr(
            self.action_decoder_config,
            "supervise_hist_actions",
            getattr(config.data, "supervise_hist_actions", True),
        )

        encoders = getattr(config.model, "encoders", None)
        self.vision_encoder_cfg = getattr(encoders, "vision_encoder", None)
        self.ft_encoder_cfg = getattr(encoders, "ft_encoder", None)
        self.proprio_encoder_cfg = getattr(encoders, "proprio_encoder", None)

        self.vision_feature_dim = getattr(self.obs_feature_dims, "vision_feature_dim", 0)
        self.ft_feature_dim = getattr(self.obs_feature_dims, "ft_feature_dim", 0)
        self.proprio_feature_dim = getattr(self.obs_feature_dims, "proprio_feature_dim", 0)

        v_type = getattr(self.vision_encoder_cfg, "type", None)
        f_type = getattr(self.ft_encoder_cfg, "type", None)
        p_type = getattr(self.proprio_encoder_cfg, "type", None)
        v_dim = self.vision_feature_dim
        f_dim = self.ft_feature_dim
        p_dim = self.proprio_feature_dim

        print(f"[RISE_FT_Proprio] fusion_type={self.fusion_type} | "
              f"vision_encoder={v_type}(dim={v_dim}) | "
              f"ft_encoder={f_type}(dim={f_dim}) | "
              f"proprio_encoder={p_type}(dim={p_dim})")


    @staticmethod
    def _call_encoder_maybe_with_batch_size(encoder: nn.Module, x, batch_size: int):
        """
        Some encoders (e.g. `rise_encoder`) require a `batch_size` kwarg, while others
        (e.g. ResNet encoders) do not. Call robustly without breaking either type.
        """
        # Prefer signature inspection (cheap + avoids try/except in the hot path when possible)
        try:
            sig = inspect.signature(encoder.forward)
            if "batch_size" in sig.parameters:
                return encoder(x, batch_size=batch_size)
            return encoder(x)
        except (TypeError, ValueError):
            # Fallback for encoders without an inspectable signature (C++/wrapped/JIT/etc.)
            try:
                return encoder(x, batch_size=batch_size)
            except TypeError:
                return encoder(x)

    def encode(self, vision=None, force_torque=None, proprio=None, batch_size=24):
        if vision is None:
            raise ValueError("The MATE RGB component requires an image.")
        if force_torque is not None or proprio is not None:
            raise ValueError("The MATE RGB component accepts vision only.")
        feature = self._call_encoder_maybe_with_batch_size(self.vision_encoder, vision, batch_size)
        return self.fusion_dropout(feature)

    def forward(self, vision=None, force_torque=None, proprio=None, actions=None, batch_size: int = 24, base_action_hist_clip_steps=None):
        feature = self.encode(vision=vision, force_torque=force_torque, proprio=proprio, batch_size=batch_size)

        if actions is not None:
            loss = self.action_decoder.compute_loss(feature, actions, hist_clip_steps=base_action_hist_clip_steps)
            return loss
        else:
            with torch.no_grad():
               action_pred = self.action_decoder.predict_action(feature)
            return action_pred
