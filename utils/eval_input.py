import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from utils.ME_utils import create_batch
from utils.pcd_utils import create_point_cloud
from utils.transformation import xyz_rot_transform
from utils.normalization import normalize_obs_proprio, normalize_force_torque
from utils.action_utils import absolute_actions_to_relative_actions


def create_2d_vision_input(colors, config, device):
    """Apply the same resize/crop/normalize pipeline as RealWorld2DDataset."""
    height, width = map(int, getattr(config.data, "img_size", [720, 1280]))
    crop_ratio = float(getattr(config.data, "crop_ratio", 0.9))
    crop_height = int(height * crop_ratio)
    crop_width = int(width * crop_ratio)

    image = Image.fromarray(np.asarray(colors, dtype=np.uint8), mode="RGB")
    resize = T.Resize([height, width])
    image = resize(image)
    image = T.CenterCrop([crop_height, crop_width])(image)
    image = resize(image)

    image = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray(config.data.normalization.img_mean, dtype=np.float32)
    std = np.asarray(config.data.normalization.img_std, dtype=np.float32)
    image = ((image - mean) / std).transpose(2, 0, 1).copy()
    return torch.from_numpy(image).unsqueeze(0).to(device)


def create_vision_input(
    colors,
    depths,
    cam_intrinsics,
    config,
    depth_scale=1000.0,
    voxel_size=0.005,
    projector=None,
):
    """RGB-D + training config → Minkowski batch coords/feats and (N, 6) point cloud array."""
    cloud = create_point_cloud(
        colors,
        depths,
        cam_intrinsics,
        config,
        depth_scale=depth_scale,
        voxel_size=voxel_size,
    )
    obs_coord = str(getattr(config.data, "obs_coord", "camera"))
    if obs_coord != "camera":
        if projector is None:
            raise ValueError(
                "create_vision_input requires a projector when obs_coord={}."
                .format(obs_coord)
            )
        cloud = cloud.copy()
        cloud[:, :3] = projector.project_pcd(
            cloud[:, :3], source_coord="camera", target_coord=obs_coord
        )
    coords = np.ascontiguousarray(cloud[:, :3] / voxel_size, dtype=np.int32)
    coords_batch, feats_batch = create_batch(coords, cloud)
    return coords_batch, feats_batch, cloud


def create_ft_input(agent, config, projector, device, force_torque=None):
    """Normalized FT tensor plus raw window. Pass ``force_torque`` to avoid a second ``get_ft_window`` call."""
    force_torque_coord = config.data.obs_coord if config.data.force_torque_coord is None else config.data.force_torque_coord
    if force_torque is None:
        force_torque = agent.get_ft_window(
            freq=config.data.obs_ft_freq,
            length=config.data.num_obs_ft,
            coordinate=force_torque_coord,
            projector=projector,
        )
    force_torque_norm = normalize_force_torque(force_torque, config)
    ft_data = torch.from_numpy(force_torque_norm).unsqueeze(0).to(device)
    return ft_data, force_torque, force_torque_coord


def create_proprio_input(agent, config, projector, device, base_absolute_action=None, tcp_raw=None):
    """Build normalized proprio input for the policy.

    The dataset training contract is:
        - ``obs_proprio`` is relativized by ``base_absolute_action``.
        - The de-relativization base used to convert predicted relative actions
          back to absolute actions MUST be the same pose.
    In async eval this means both workers must agree on ``base_absolute_action``
    (= vision-time tcp projected into ``obs_coord``); otherwise the predicted
    trajectory is shifted by the inter-observation tcp displacement.

    Parameters
    ----------
    base_absolute_action : np.ndarray or None
        Optional ``(7,)`` pose (xyz + quaternion) expressed in
        ``config.data.obs_coord``. If None and relative mode is on, falls back
        to ``tcp_raw[-1]`` (the legacy sync-mode behavior).
    tcp_raw : np.ndarray or None
        Optional pre-fetched TCP window; if None, ``get_tcp_window`` is called.
    """
    if tcp_raw is None:
        tcp_raw = agent.get_tcp_window(
            freq=config.data.obs_proprio_freq,
            length=config.data.num_obs_proprio,
            coordinate=config.data.obs_coord,
            projector=projector,
        )
    if config.data.relative_proprio_obs_for_relative_action and config.data.relative_action:
        if base_absolute_action is None:
            base_absolute_action = tcp_raw[-1]
        tcp = absolute_actions_to_relative_actions(tcp_raw, base_absolute_action, rotation_rep="quaternion")
    else:
        tcp = tcp_raw

    tcp_rot6d = xyz_rot_transform(
        tcp, from_rep="quaternion", to_rep="rotation_6d"
    )
    tcp_norm = normalize_obs_proprio(tcp_rot6d, config)
    tcp_data = torch.from_numpy(tcp_norm).unsqueeze(0).to(device)
    return tcp_data, tcp_raw
