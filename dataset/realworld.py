import os
import sys
import torch
import numpy as np
import h5py
import open3d as o3d
import MinkowskiEngine as ME
import torchvision.transforms as T
import collections.abc as container_abcs

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

from dataset.constants import TO_TENSOR_KEYS
from utils.dataset_schema import read_lowdim
from utils.runtime_config import resolve_path
from dataset.projector import Projector
from utils.config import get_action_rotation_repr
from utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_point, xyz_rot_transform, xyz_rot_to_mat
from utils.normalization import (
    normalize_tcp,
    normalize_force_torque,
    normalize_obs_proprio,
    normalize_action,
    normalize_global_action,
)
from utils.visualize import create_force_torque_arrow
from utils.action_utils import absolute_actions_to_relative_actions, compute_desired_action_tcps

class RealWorldVisionDataset(Dataset):
    """
    Real-world Dataset.
    """
    def __init__(
        self, 
        path, 
        config,
        split = 'train',
    ):
        assert split in ['train', 'val', 'all']

        # Support both single path (str) and multiple paths (list)
        if isinstance(path, str):
            self.paths = [path]
        elif isinstance(path, (list, tuple)):
            self.paths = list(path)
        else:
            raise TypeError(f"path must be str or list/tuple, got {type(path)}")
        
        self.path = self.paths[0]  # Keep for backward compatibility
        self.split = split
        self.config = config
        self._parse_config(config)
        cam_ids = [str(config.data.cam_id)]

        # Use first path's calib (assuming all paths use same calibration)
        self.calib_path = str(resolve_path(config.data.calib_path)) if getattr(config.data, "calib_path", "") else os.path.join(self.paths[0], "calib")

        # Collect all demos from all paths
        # Use unique identifier to handle duplicate demo names across different paths
        # Format: "{base_path}::{demo_name}" to ensure uniqueness
        self.all_demos = []  # List of unique identifiers
        self.demo_id_to_info = {}  # Map unique demo_id to (base_path, demo_name)
        
        for base_path in self.paths:
            data_path = os.path.join(base_path, split)
            if not os.path.exists(data_path):
                print(f"Warning: data path {data_path} does not exist, skipping...")
                continue
            demos_in_path = sorted(os.listdir(data_path))
            for demo_name in demos_in_path:
                # Create unique identifier using base_path and demo_name
                demo_id = f"{base_path}::{demo_name}"
                if demo_id not in self.all_demos:
                    self.all_demos.append(demo_id)
                    self.demo_id_to_info[demo_id] = (base_path, demo_name)
        
        self.num_demos = len(self.all_demos)

        self.data_paths = []
        self.cam_ids = []
        self.obs_frame_ids = []
        self.image_frame_ids = []
        self.sample_timestamps = []
        self.action_frame_ids = []
        self.projectors = {}

        self.obs_force_torque_raw = []
        self.obs_tcp_pose_raw = []
        self.obs_proprio_raw = []
        self.action_future_tcp_pose = []
        self.ee_command_future = []
        self.action_force_torque = []
        self.skipped_image_alignment_samples = 0

        for i in range(self.num_demos):
            demo_id = self.all_demos[i]
            base_path, demo_name = self.demo_id_to_info[demo_id]
            demo_path = os.path.join(base_path, split, demo_name)
            lowdim_path = os.path.join(demo_path, "lowdim")
            with h5py.File(os.path.join(lowdim_path, getattr(config.data, "lowdim_file", "lowdim_updated.h5")), "r") as f:
                lowdim = read_lowdim(f, config, require_force=True)
                timestamp = lowdim["timestamp"]
                tcp_pose = lowdim["tcp_pose"]
                ee_command = lowdim["ee_command"]
                force_torque = lowdim["force_torque"]

            if self.temporal_smoothing:
                force_torque = self._moving_average_force_torque(force_torque, self.temporal_smoothing_window)

            for cam_id in cam_ids:
                # path
                cam_path = os.path.join(demo_path, "cam_{}".format(cam_id))
                if not os.path.exists(cam_path):
                    continue
                # get frame ids
                frame_ids = [
                    int(os.path.splitext(x)[0]) 
                    for x in sorted(os.listdir(os.path.join(cam_path, "color"))) 
                ]
                if len(frame_ids) == 0:
                    continue
                image_frame_ids = None
                if self.return_images:
                    image_color_dir = os.path.join(
                        demo_path, f"cam_{self.image_cam_id}", "color"
                    )
                    if not os.path.isdir(image_color_dir):
                        raise FileNotFoundError(
                            f"Missing 2D image camera directory: {image_color_dir}"
                        )
                    image_frame_ids = sorted(
                        int(os.path.splitext(x)[0])
                        for x in os.listdir(image_color_dir)
                    )
                    if len(image_frame_ids) == 0:
                        raise ValueError(
                            f"No 2D image frames found in {image_color_dir}."
                        )
                # get samples according to num_obs and num_action
                obs_frame_ids_list = []
                image_obs_frame_ids_list = []
                action_frame_ids_list = []
                padding_mask_list = []

                action_future_tcp_pose_list = []
                ee_command_future_list = []
                obs_tcp_pose_list = []
                obs_force_torque_list = []
                obs_proprio_list = []
                action_force_torque_list = []
                sample_timestamp_list = []

                for cur_idx in range(len(frame_ids) - 1):
                    cur_frame_id = frame_ids[cur_idx]
                    obs_pad_before = max(0, self.num_obs - cur_idx - 1)
                    frame_begin = max(0, cur_idx - self.num_obs + 1)
                    obs_frame_ids = frame_ids[:1] * obs_pad_before + frame_ids[frame_begin: cur_idx + 1]
                    if image_frame_ids is not None:
                        aligned_image_frames = []
                        alignment_valid = True
                        for obs_frame_id in obs_frame_ids:
                            insert_at = np.searchsorted(
                                image_frame_ids, obs_frame_id
                            )
                            candidate_indices = {
                                max(0, insert_at - 1),
                                min(len(image_frame_ids) - 1, insert_at),
                            }
                            image_frame_id = min(
                                (image_frame_ids[j] for j in candidate_indices),
                                key=lambda x: abs(x - obs_frame_id),
                            )
                            time_delta = abs(image_frame_id - obs_frame_id)
                            if (
                                self.image_max_time_delta is not None
                                and time_delta > self.image_max_time_delta
                            ):
                                alignment_valid = False
                                break
                            aligned_image_frames.append(image_frame_id)
                        if not alignment_valid:
                            self.skipped_image_alignment_samples += 1
                            continue
                        image_obs_frame_ids_list.append(aligned_image_frames)
                    obs_frame_ids_list.append(obs_frame_ids)

                    # idx of cur_frame_id in timestamp
                    cur_low_dim_idx = np.argmin(np.abs(timestamp - cur_frame_id))
                    # obs_force_torque_ids = np.arange(cur_low_dim_idx - num_obs_ft * obs_ft_step, cur_low_dim_idx, obs_ft_step)
                    obs_force_torque_ids = np.arange(cur_low_dim_idx - self.num_obs_ft * self.obs_ft_step + self.obs_ft_step // 2, cur_low_dim_idx + self.obs_ft_step // 2)
                    obs_force_torque_ids = np.clip(obs_force_torque_ids, 0, len(timestamp) - 1)
                    obs_proprio_ids = np.arange(cur_low_dim_idx - self.num_obs_proprio * self.obs_proprio_step + self.obs_proprio_step // 2, cur_low_dim_idx + self.obs_proprio_step // 2)
                    obs_proprio_ids = np.clip(obs_proprio_ids, 0, len(timestamp) - 1)
                    action_future_ids = np.arange(cur_low_dim_idx + 1, cur_low_dim_idx + self.action_horizon_future * self.action_step + 1, self.action_step)
                    action_future_ids = np.clip(action_future_ids, 0, len(timestamp) - 1)

                    obs_tcp_pose_list.append(tcp_pose[obs_force_torque_ids])
                    obs_force_torque_list.append(force_torque[obs_force_torque_ids])
                    obs_proprio_list.append(tcp_pose[obs_proprio_ids])
                    action_future_tcp_pose_list.append(tcp_pose[action_future_ids])
                    ee_command_future_list.append(ee_command[action_future_ids])
                    action_force_torque_list.append(force_torque[action_future_ids])
                    sample_timestamp_list.append(cur_frame_id)
                
                self.data_paths += [demo_path] * len(obs_frame_ids_list)
                self.cam_ids += [cam_id] * len(obs_frame_ids_list)
                self.obs_frame_ids += obs_frame_ids_list
                self.image_frame_ids += image_obs_frame_ids_list
                self.sample_timestamps += sample_timestamp_list
                self.obs_tcp_pose_raw += obs_tcp_pose_list
                self.obs_force_torque_raw += obs_force_torque_list
                self.obs_proprio_raw += obs_proprio_list
                self.action_future_tcp_pose += action_future_tcp_pose_list
                self.ee_command_future += ee_command_future_list
                self.action_force_torque += action_force_torque_list
        if self.skipped_image_alignment_samples:
            print(
                "Skipped {} samples whose 3D/2D camera timestamps exceeded "
                "image_max_time_delta={}.".format(
                    self.skipped_image_alignment_samples,
                    self.image_max_time_delta,
                )
            )
        
    def _parse_config(self, config):
        self.num_obs = config.data.num_obs
        self.num_obs_ft = config.data.num_obs_ft
        self.num_obs_proprio = config.data.num_obs_proprio
        self.obs_proprio_freq = config.data.obs_proprio_freq
        self.obs_ft_freq = config.data.obs_ft_freq
        self.obs_ft_step = int(1000.0 / self.obs_ft_freq)
        self.obs_proprio_step = int(1000.0 / self.obs_proprio_freq)
        self.action_horizon_future = getattr(config.data, "action_horizon_future", getattr(config.data, "num_action", None))
        if self.action_horizon_future is None:
            raise ValueError("config.data.action_horizon_future is required (or legacy config.data.num_action).")
        self.action_freq = config.data.action_freq
        self.action_step = int(1000.0 / self.action_freq)
        self.relative_action = config.data.relative_action
        self.relative_proprio_obs_for_relative_action = config.data.relative_proprio_obs_for_relative_action
        # force_torque_coord: "tcp" to keep force_torque in tcp coordinate, None to use obs_coord/action_coord
        self.force_torque_coord = getattr(config.data, "force_torque_coord", None)
        self.obs_coord = config.data.obs_coord
        self.action_coord = config.data.action_coord
        self.action_rotation_repr = get_action_rotation_repr(config)
        # Validate force_torque_coord
        if self.force_torque_coord is not None:
            assert self.force_torque_coord == "tcp", f"force_torque_coord must be 'tcp' or None, got {self.force_torque_coord}"
        self.voxel_size = config.data.voxel_size
        self.workspace_min = np.array(config.train.workspace.min)
        self.workspace_max = np.array(config.train.workspace.max)
        self.translation_min = np.asarray(config.data.normalization.trans_min)
        self.translation_max = np.asarray(config.data.normalization.trans_max)
        self.aug_point = config.train.augmentation.point
        self.aug_trans_min = np.array(config.train.augmentation.aug_trans_min)
        self.aug_trans_max = np.array(config.train.augmentation.aug_trans_max)
        self.aug_rot_min = np.array(config.train.augmentation.aug_rot_min)
        self.aug_rot_max = np.array(config.train.augmentation.aug_rot_max)
        self.aug_color = config.train.augmentation.color
        self.aug_color_params = np.array(config.train.augmentation.aug_color_params)
        self.aug_color_prob = config.train.augmentation.aug_color_prob
        self.aug_ft = config.train.augmentation.ft_sample
        self.temporal_smoothing = config.train.ft_temporal_smoothing
        self.temporal_smoothing_window = config.train.ft_temporal_smoothing_window
        self.with_cloud = config.data.with_cloud
        self.vis = config.data.vis
        self.stiffness = np.array(config.data.stiffness)
        # self.stiffness = np.diag(config.data.stiffness)

        self.img_mean = np.array(config.data.normalization.img_mean)
        self.img_std = np.array(config.data.normalization.img_std)
        self.return_images = bool(getattr(config.data, "return_images", False))
        self.image_cam_id = str(
            getattr(config.data, "image_cam_id", config.data.cam_id)
        )
        image_max_time_delta = getattr(
            config.data, "image_max_time_delta", None
        )
        self.image_max_time_delta = (
            None
            if image_max_time_delta is None
            else int(image_max_time_delta)
        )
        self.img_size = tuple(getattr(config.data, "img_size", (720, 1280)))
        self.crop_ratio = float(getattr(config.data, "crop_ratio", 0.9))
        if self.return_images:
            if len(self.img_size) != 2:
                raise ValueError(
                    "config.data.img_size must be [height, width] when return_images=True."
                )
            if not 0.0 < self.crop_ratio <= 1.0:
                raise ValueError(
                    "config.data.crop_ratio must be in (0, 1] when return_images=True."
                )
            if self.aug_point:
                raise ValueError(
                    "Point augmentation must be disabled when return_images=True: "
                    "the current geometric augmentation transforms point clouds and "
                    "actions, but not the 2D image."
                )
        if self.relative_action:
            assert self.obs_coord == self.action_coord, (
                "relative_action=True requires obs_coord == action_coord, "
                f"got obs_coord={self.obs_coord}, action_coord={self.action_coord}. "
                "base_absolute_action is taken from obs_proprio[-1] (in obs_coord) and "
                "applied to actions (in action_coord); mismatched frames would invalidate the relative transform."
            )

    def __len__(self):
        return len(self.obs_frame_ids)

    def _augmentation(self, clouds, tcps):
        translation_offsets = np.random.rand(3) * (self.aug_trans_max - self.aug_trans_min) + self.aug_trans_min
        rotation_angles = np.random.rand(3) * (self.aug_rot_max - self.aug_rot_min) + self.aug_rot_min
        rotation_angles = rotation_angles / 180 * np.pi  # tranform from degree to radius
        aug_mat = rot_trans_mat(translation_offsets, rotation_angles)
        aug_rot = aug_mat[:3, :3]  # rotation matrix
        center = clouds[-1][..., :3].mean(axis = 0)

        for i in range(len(clouds)):
            clouds[i][..., :3] -= center
            clouds[i] = apply_mat_to_point(clouds[i], aug_mat)
            clouds[i][..., :3] += center

        tcps[..., :3] -= center
        tcps = apply_mat_to_pose(tcps, aug_mat, rotation_rep = "quaternion")
        tcps[..., :3] += center

        return clouds, tcps

    def _moving_average_force_torque(self, force_torque_list, window_size):
        if window_size <= 1:
            return force_torque_list
        kernel = np.ones(window_size, dtype=np.float32) / window_size
        # If 1D: convolve along the first axis
        if force_torque_list.ndim == 1:
            return np.convolve(force_torque_list, kernel, mode='same')
        # If 2D: convolve each column independently
        out = np.empty_like(force_torque_list, dtype=np.float32)
        for c in range(force_torque_list.shape[1]):
            out[:, c] = np.convolve(force_torque_list[:, c], kernel, mode='same')
        return out

    def load_point_cloud(self, colors, depths, intrinsics, depth_scale=1000.):
        h, w = depths.shape
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        colors = o3d.geometry.Image(colors.astype(np.uint8))
        depths = o3d.geometry.Image(depths.astype(np.float32))
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            colors, depths, depth_scale, convert_rgb_to_intensity = False
        )
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
        bbox3d = o3d.geometry.AxisAlignedBoundingBox(self.workspace_min, self.workspace_max)
        cloud = cloud.crop(bbox3d)
        # downsample
        cloud = cloud.voxel_down_sample(self.voxel_size)
        return cloud

    def _preprocess_images_for_2d(self, colors_list):
        """Apply the same crop/resize and ImageNet normalization as RealWorld2DDataset."""
        height, width = self.img_size
        crop_height = int(height * self.crop_ratio)
        crop_width = int(width * self.crop_ratio)
        resize = T.Resize([height, width])
        crop = (
            T.RandomCrop([crop_height, crop_width])
            if self.split == "train"
            else T.CenterCrop([crop_height, crop_width])
        )

        images = []
        for colors in colors_list:
            image = Image.fromarray(colors.astype(np.uint8))
            image = resize(image)
            image = crop(image)
            image = resize(image)
            image = np.asarray(image, dtype=np.float32) / 255.0
            image = (image - self.img_mean) / self.img_std
            images.append(image.transpose(2, 0, 1))

        images = torch.from_numpy(np.stack(images, axis=0)).float()
        if self.num_obs == 1:
            images = images[0]
        return images

    def __getitem__(self, index):
        data_path = self.data_paths[index]
        cam_id = self.cam_ids[index]
        obs_frame_ids = self.obs_frame_ids[index]
        image_frame_ids = (
            self.image_frame_ids[index] if self.return_images else None
        )
        timestamp = self.sample_timestamps[index]

        # directories
        color_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'color')
        depth_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'depth')

        # load camera projector
        # projector = Projector(os.path.join(self.calib_path, "new_right.npy"))
        projector = Projector(os.path.join(self.calib_path, getattr(self.config.data, "calibration_file", "result.json")), global_cam_serial=str(self.config.data.cam_id))
        intrinsics = projector.cam_intrinsics[cam_id]
        depth_scale = projector.depth_scales[cam_id]

        # create color jitter
        if self.split == 'train' and self.aug_color:
            jitter = T.ColorJitter(
                brightness = self.aug_color_params[0],
                contrast = self.aug_color_params[1],
                saturation = self.aug_color_params[2],
                hue = self.aug_color_params[3]
            )
            jitter = T.RandomApply([jitter], p = self.aug_color_prob)

        # load colors and depths
        colors_list = []
        depths_list = []
        for frame_id in obs_frame_ids:
            colors = Image.open(os.path.join(color_dir, "{}.png".format(frame_id)))
            if self.split == 'train' and self.aug_color:
                colors = jitter(colors)
            colors_list.append(colors)
            depths_list.append(
                np.array(Image.open(os.path.join(depth_dir, "{}.png".format(frame_id))), dtype = np.float32)
            )
        colors_list = np.stack(colors_list, axis = 0)
        depths_list = np.stack(depths_list, axis = 0)
        images = None
        if self.return_images:
            if self.image_cam_id == cam_id and image_frame_ids == obs_frame_ids:
                image_colors_list = colors_list
            else:
                image_color_dir = os.path.join(
                    data_path, f"cam_{self.image_cam_id}", "color"
                )
                image_colors_list = np.stack(
                    [
                        np.asarray(
                            Image.open(
                                os.path.join(
                                    image_color_dir, f"{frame_id}.png"
                                )
                            )
                        )
                        for frame_id in image_frame_ids
                    ],
                    axis=0,
                )
            images = self._preprocess_images_for_2d(image_colors_list)

        # point clouds
        clouds = []
        for i, frame_id in enumerate(obs_frame_ids):
            cloud = self.load_point_cloud(colors_list[i], depths_list[i], intrinsics, depth_scale)
            points = np.array(cloud.points)
            colors = np.array(cloud.colors)
            # apply imagenet normalization
            colors = (colors - self.img_mean) / self.img_std
            clouds.append(np.concatenate([points, colors], axis = -1))

        # obs force-torque
        obs_force_torque_raw = self.obs_force_torque_raw[index]
        obs_tcp_pose_raw = self.obs_tcp_pose_raw[index]
        obs_proprio_raw = self.obs_proprio_raw[index]
        action_force_torque_raw = self.action_force_torque[index]
        # obs force-torque augmentations
        if self.split == 'train' and self.aug_ft:
            start_idx = np.random.randint(0, self.obs_ft_step)
        else:
            start_idx = self.obs_ft_step // 2
        idx = start_idx + self.obs_ft_step * np.arange(self.num_obs_ft)
        idx_proprio = start_idx + self.obs_proprio_step * np.arange(self.num_obs_proprio)
        obs_force_torque = obs_force_torque_raw[idx]
        obs_tcp_pose = obs_tcp_pose_raw[idx] # (num_obs_ft, 7) in base coordinate
        obs_proprio = obs_proprio_raw[idx_proprio] # (num_obs_proprio, 7) in base coordinate
        obs_tcps = projector.project_tcp(obs_tcp_pose, source_coord = "base", target_coord = "camera") # (num_obs_ft, 7) in camera coordinate
        obs_proprio = projector.project_tcp(obs_proprio, source_coord = "base", target_coord = "camera") # (num_obs_proprio, 7) in camera coordinate
        # actions
        action_future_tcps = self.action_future_tcp_pose[index]
        action_future_grippers = self.ee_command_future[index]
        action_future_tcps = projector.project_tcp(action_future_tcps, source_coord = "base", target_coord = "camera")

        # point & tcp (and optionally force/torque) augmentations
        if self.split == 'train' and self.aug_point:
            tcps = np.concatenate((obs_tcps, obs_proprio, action_future_tcps), axis = 0)
            clouds, tcps = self._augmentation(clouds, tcps)
            obs_tcps = tcps[:self.num_obs_ft] # (num_obs_ft, 7) in camera coordinate
            obs_proprio = tcps[self.num_obs_ft:self.num_obs_ft + self.num_obs_proprio] # (num_obs_proprio, 7) in camera coordinate
            action_future_tcps = tcps[self.num_obs_ft + self.num_obs_proprio:]
        
        for i in range(len(obs_frame_ids)):
            clouds[i][..., :3] = projector.project_pcd(clouds[i][..., :3], source_coord = "camera", target_coord = self.obs_coord)
        obs_tcps = projector.project_tcp(obs_tcps, source_coord = "camera", target_coord = self.obs_coord)
        action_future_tcps = projector.project_tcp(action_future_tcps, source_coord = "camera", target_coord = self.action_coord)
        obs_proprio = projector.project_tcp(obs_proprio, source_coord = "camera", target_coord = self.obs_coord)
        obs_tcps_base = projector.project_tcp(obs_tcps, source_coord = self.obs_coord, target_coord = "base")
        action_future_tcps_base = projector.project_tcp(action_future_tcps, source_coord = self.action_coord, target_coord = "base")

        # force_torque coordinate transformation
        if self.force_torque_coord == "tcp":
            # Keep force_torque in tcp coordinate (independent of obs_coord/action_coord)
            obs_force_torque = obs_force_torque.copy()  # already in tcp coordinate
            action_force_torque = action_force_torque_raw.copy()  # already in tcp coordinate
        else:
            # Transform force_torque to obs_coord and action_coord (default behavior)
            obs_force_torque = projector.project_wrench(
                obs_force_torque,
                source_coord="tcp",
                target_coord=self.obs_coord,
                tcp=xyz_rot_to_mat(obs_tcps_base, rotation_rep="quaternion"),
            )
            action_force_torque = projector.project_wrench(
                action_force_torque_raw,
                source_coord="tcp",
                target_coord=self.action_coord,
                tcp=xyz_rot_to_mat(action_future_tcps_base, rotation_rep="quaternion"),
            )

        # compute desired action tcps
        desired_action_tcps = compute_desired_action_tcps(
            projector,
            action_future_tcps,
            action_force_torque,
            action_future_tcps_base,
            self.force_torque_coord,
            self.action_coord,
            self.stiffness,
        )

        # relative action
        if self.relative_action:
            base_absolute_action = obs_proprio[-1].copy()
            action_future_tcps = absolute_actions_to_relative_actions(action_future_tcps, base_absolute_action, rotation_rep = "quaternion")

            if self.relative_proprio_obs_for_relative_action:
                obs_proprio = absolute_actions_to_relative_actions(obs_proprio, base_absolute_action, rotation_rep = "quaternion")

        
        # visualization
        if self.vis:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            points = clouds[-1][..., :3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors * self.img_std + self.img_mean)
            traj = []
            # red box stands for the workspace range
            bbox3d_1 = o3d.geometry.AxisAlignedBoundingBox(self.workspace_min, self.workspace_max)
            bbox3d_1.color = [1, 0, 0]
            # green box stands for the translation normalization range
            bbox3d_2 = o3d.geometry.AxisAlignedBoundingBox(self.translation_min, self.translation_max)
            bbox3d_2.color = [0, 1, 0]
            action_tcps_vis = xyz_rot_transform(action_future_tcps, from_rep = "quaternion", to_rep = "matrix")
            # action visualization
            for i in range(len(action_tcps_vis)):
                # force_arrow = create_force_torque_arrow(action_force_torque[i], action_future_tcps[i])
                # traj.append(force_arrow)
                traj.append(o3d.geometry.TriangleMesh.create_sphere(0.001).translate(action_future_tcps[i][:3]).paint_uniform_color([1, 0, 0]))
            # desired action visualization
            for i in range(len(desired_action_tcps)):
                # visualize as points with color blue
                traj.append(o3d.geometry.TriangleMesh.create_sphere(0.001).translate(desired_action_tcps[i][:3]).paint_uniform_color([0, 0, 1]))
            obs_tcps_vis = xyz_rot_transform(obs_tcps, from_rep = "quaternion", to_rep = "matrix")
            # obs force-torque visualization
            if self.force_torque_coord == "tcp":
                obs_force_torque_vis = projector.project_wrench(
                    obs_force_torque,
                    source_coord="tcp",
                    target_coord=self.obs_coord,
                    tcp=xyz_rot_to_mat(obs_tcp_pose, rotation_rep="quaternion"),
                )
            else:
                obs_force_torque_vis = obs_force_torque
            for i in range(len(obs_tcps_vis)):
                force_arrow = create_force_torque_arrow(obs_force_torque_vis[i], obs_tcps[i], color = [1, 0, 0])
                traj.append(force_arrow)
            # obs proprio visualization
            obs_proprio_vis = xyz_rot_transform(obs_proprio, from_rep = "quaternion", to_rep = "matrix")
            for i in range(len(obs_proprio_vis)):
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03).transform(obs_proprio_vis[i])
                traj.append(frame)
            # if self.relative_action:
            #     relative_action_tcps_vis = xyz_rot_transform(relative_action_tcps, from_rep = "quaternion", to_rep = "matrix")
            #     for i in range(len(relative_action_tcps_vis)):
            #         frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03).transform(relative_action_tcps_vis[i])
            #         traj.append(frame)
            o3d.visualization.draw_geometries([pcd.voxel_down_sample(self.voxel_size), bbox3d_1, bbox3d_2, *traj, axis])
        
        # rotation transformation (to 6d)
        obs_tcps = xyz_rot_transform(obs_tcps, from_rep = "quaternion", to_rep = "rotation_6d")
        obs_proprio = xyz_rot_transform(obs_proprio, from_rep = "quaternion", to_rep = "rotation_6d")
        action_future_tcps = xyz_rot_transform(
            action_future_tcps, from_rep="quaternion", to_rep=self.action_rotation_repr
        )

        action_future = np.concatenate((action_future_tcps, action_future_grippers), axis = -1)

        # normalization
        # Use separate normalization functions for obs_proprio and action
        obs_force_torque_normalized = normalize_force_torque(obs_force_torque.copy(), config = self.config)
        obs_tcps_normalized = normalize_tcp(obs_tcps.copy(), config = self.config, gripper_width_included = False)
        obs_proprio_normalized = normalize_obs_proprio(obs_proprio.copy(), config = self.config)
        if self.relative_action:
            action_future_normalized = normalize_action(
                action_future.copy(), config=self.config
            )
        else:
            action_future_normalized = normalize_global_action(
                action_future.copy(), config=self.config
            )


        # make voxel input
        input_coords_list = []
        input_feats_list = []
        for cloud in clouds:
            # Upd Note. Make coords contiguous.
            coords = np.ascontiguousarray(cloud[:, :3] / self.voxel_size, dtype = np.int32)
            # Upd Note. API change.
            input_coords_list.append(coords)
            input_feats_list.append(cloud.astype(np.float32))

        # convert to torch
        obs_tcps = torch.from_numpy(obs_tcps).float()
        obs_tcps_normalized = torch.from_numpy(obs_tcps_normalized).float()
        obs_proprio_normalized = torch.from_numpy(obs_proprio_normalized).float()
        action_future = torch.from_numpy(action_future).float()
        action_future_normalized = torch.from_numpy(action_future_normalized).float()

        ret_dict = {
            'input_coords_list': input_coords_list,
            'input_feats_list': input_feats_list,
            'obs_force_torque_raw': obs_force_torque,
            'obs_force_torque': obs_force_torque_normalized,
            'obs_proprio': obs_proprio,
            'obs_proprio_normalized': obs_proprio_normalized,
            'action_future': action_future,
            'action_future_normalized': action_future_normalized,
            'action': action_future,
            'action_normalized': action_future_normalized,
            'timestamp': timestamp,
        }
        if images is not None:
            ret_dict["images"] = images
        
        if self.relative_action:
            ret_dict['base_absolute_action'] = base_absolute_action
        
        if self.with_cloud:  # warning: this may significantly slow down the training process.
            ret_dict["clouds_list"] = clouds

        return ret_dict
        

def collate_fn_vision(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {}
        for key in batch[0]:
            if key == "images":
                ret_dict[key] = torch.stack([d[key] for d in batch], dim=0)
            elif key in TO_TENSOR_KEYS:
                ret_dict[key] = collate_fn_vision([d[key] for d in batch])
            else:
                ret_dict[key] = [d[key] for d in batch]
        coords_batch = ret_dict['input_coords_list']
        feats_batch = ret_dict['input_feats_list']
        coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
        ret_dict['input_coords_list'] = coords_batch
        ret_dict['input_feats_list'] = feats_batch
        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        return [sample for b in batch for sample in b]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))

if __name__ == "__main__":
    from utils.config import load_config
    from torch.utils.data import DataLoader
    
    import argparse
    parser = argparse.ArgumentParser(description="Inspect the configured vision dataset.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--config", default="configs/components/vision_3d.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = RealWorldVisionDataset(
        path = args.data_path,
        config = config,
        split = "train",
    )
    
    # Use DataLoader with multiple workers for parallel loading
    # Set num_workers based on available CPUs (use 8-16 workers typically)
    num_workers = min(16, os.cpu_count() or 1)
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=collate_fn_vision,
        pin_memory=False,  # Set to True if using GPU, but we're just computing stats
    )
    
    print(f"Processing {len(dataset)} samples with {num_workers} workers...")
    for ret_dict in tqdm(dataloader):
        # Handle batched data from DataLoader
        # obs_proprio is not in TO_TENSOR_KEYS, so it's a list [tensor]
        obs_proprio_data = ret_dict['obs_proprio'][0] if isinstance(ret_dict['obs_proprio'], list) else ret_dict['obs_proprio']
        # action is in TO_TENSOR_KEYS, so it's stacked with batch dimension
        # Remove batch dimension (first dim) if it exists
        action_data = ret_dict['action']
        # action_data_normalized = ret_dict['action_normalized']
        # if action_data.dim() > 2:  # Has batch dimension: (batch, T, D)
        #     action_data = action_data.squeeze(0)  # Remove batch dim: (T, D)

        print(action_data)
        # print(action_data_normalized)
