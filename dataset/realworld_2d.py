import os
import sys
import torch
import numpy as np
import h5py
import torchvision.transforms as T
import collections.abc as container_abcs

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PIL import Image
from torch.utils.data import Dataset

from dataset.constants import TO_TENSOR_KEYS
from utils.dataset_schema import read_lowdim
from utils.runtime_config import resolve_path
from dataset.projector import Projector
from utils.config import get_action_rotation_repr
from utils.transformation import xyz_rot_transform, xyz_rot_to_mat, mat_to_xyz_rot
from utils.normalization import (
    normalize_action,
    normalize_global_action,
    normalize_obs_proprio,
)
from utils.action_utils import absolute_actions_to_relative_actions

class RealWorld2DDataset(Dataset):
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
        self.action_frame_ids = []

        self.obs_proprio_raw = []
        self.tcp_pose = []
        self.ee_command = []

        for i in range(self.num_demos):
            demo_id = self.all_demos[i]
            base_path, demo_name = self.demo_id_to_info[demo_id]
            demo_path = os.path.join(base_path, split, demo_name)
            lowdim_path = os.path.join(demo_path, "lowdim")
            with h5py.File(os.path.join(lowdim_path, getattr(config.data, "lowdim_file", "lowdim_updated.h5")), "r") as f:
                lowdim = read_lowdim(f, config, require_force=False)
                timestamp = lowdim["timestamp"]
                tcp_pose = lowdim["tcp_pose"]
                ee_command = lowdim["ee_command"]

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
                # get samples according to num_obs and num_action
                obs_frame_ids_list = []
                action_frame_ids_list = []
                padding_mask_list = []

                tcp_pose_list = []
                ee_command_list = []
                obs_proprio_list = []

                for cur_idx in range(len(frame_ids) - 1):
                    cur_frame_id = frame_ids[cur_idx]
                    obs_pad_before = max(0, self.num_obs - cur_idx - 1)
                    frame_begin = max(0, cur_idx - self.num_obs + 1)
                    obs_frame_ids = frame_ids[:1] * obs_pad_before + frame_ids[frame_begin: cur_idx + 1]
                    obs_frame_ids_list.append(obs_frame_ids)

                    # idx of cur_frame_id in timestamp
                    cur_low_dim_idx = np.argmin(np.abs(timestamp - cur_frame_id))
                    obs_proprio_ids = np.arange(cur_low_dim_idx - self.num_obs_proprio * self.obs_proprio_step + self.obs_proprio_step // 2, cur_low_dim_idx + self.obs_proprio_step // 2)
                    obs_proprio_ids = np.clip(obs_proprio_ids, 0, len(timestamp) - 1)
                    action_ids = np.arange(cur_low_dim_idx + 1, cur_low_dim_idx + self.num_action * self.action_step + 1, self.action_step) # (num_action,)
                    action_ids = np.clip(action_ids, 0, len(timestamp) - 1)

                    obs_proprio_list.append(tcp_pose[obs_proprio_ids])
                    tcp_pose_list.append(tcp_pose[action_ids])
                    ee_command_list.append(ee_command[action_ids])
                
                self.data_paths += [demo_path] * len(obs_frame_ids_list)
                self.cam_ids += [cam_id] * len(obs_frame_ids_list)
                self.obs_frame_ids += obs_frame_ids_list
                self.obs_proprio_raw += obs_proprio_list
                self.tcp_pose += tcp_pose_list
                self.ee_command += ee_command_list
        
    def _parse_config(self, config):
        self.num_obs = config.data.num_obs
        if self.num_obs != 1:
            raise ValueError(f"RealWorld2DDataset requires data.num_obs=1, got {self.num_obs}")
        self.num_obs_proprio = config.data.num_obs_proprio
        self.obs_proprio_freq = config.data.obs_proprio_freq
        self.obs_proprio_step = int(1000.0 / self.obs_proprio_freq)
        self.num_action = config.data.num_action
        self.action_freq = config.data.action_freq
        self.action_step = int(1000.0 / self.action_freq)
        self.relative_action = config.data.relative_action
        self.relative_proprio_obs_for_relative_action = config.data.relative_proprio_obs_for_relative_action
        self.obs_coord = config.data.obs_coord
        self.action_coord = config.data.action_coord
        self.action_rotation_repr = get_action_rotation_repr(config)
        self.vis = config.data.vis

        self.img_mean = np.array(config.data.normalization.img_mean)
        self.img_std = np.array(config.data.normalization.img_std)
        
        # Image size configuration for crop augmentation
        self.img_size = getattr(config.data, "img_size", [180, 320])
        self.crop_ratio = getattr(config.data, "crop_ratio", 0.9)
        # augmentation flag: True for training (RandomCrop), False for validation/test (CenterCrop)
        self.augmentation = (self.split == 'train')
        self.inhand_camera = getattr(config.data, "inhand_camera", False)
        if self.inhand_camera:
            if self.relative_action:
                assert self.obs_coord == "tcp" and self.action_coord == "tcp", (
                    "inhand_camera=True with relative_action=True requires "
                    "obs_coord == action_coord == 'tcp', "
                    f"got obs_coord={self.obs_coord}, "
                    f"action_coord={self.action_coord}."
                )
            else:
                assert self.obs_coord == "base" and self.action_coord == "base", (
                    "inhand_camera=True with global actions requires the fixed "
                    "base frame for both proprio and action labels; got "
                    f"obs_coord={self.obs_coord}, "
                    f"action_coord={self.action_coord}."
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

    @staticmethod
    def _project_tcp_base_to_tcp_coord(tcp_base, ref_tcp_base, rotation_rep="quaternion"):
        """Express base-frame tcp poses in the coordinate frame of ref_tcp_base."""
        tcp_base = np.asarray(tcp_base, dtype=np.float32)
        ref_tcp_base = np.asarray(ref_tcp_base, dtype=np.float32)
        T_base_ref = xyz_rot_to_mat(ref_tcp_base, rotation_rep=rotation_rep)
        T_ref_base = np.linalg.inv(T_base_ref)

        if tcp_base.ndim == 1:
            T_base_tcp = xyz_rot_to_mat(tcp_base, rotation_rep=rotation_rep)
            return mat_to_xyz_rot(T_ref_base @ T_base_tcp, rotation_rep=rotation_rep)

        out = []
        for pose in tcp_base:
            T_base_tcp = xyz_rot_to_mat(pose, rotation_rep=rotation_rep)
            out.append(mat_to_xyz_rot(T_ref_base @ T_base_tcp, rotation_rep=rotation_rep))
        return np.stack(out, axis=0)

    def crop_resize_image(self, image, image_size=None, crop_ratio=None):
        """
        Crop and resize image augmentation.
        
        Parameters
        ----------
        image : PIL.Image
            Input PIL Image
        image_size : list, optional
            Target image size [height, width]. If None, uses self.img_size
        crop_ratio : float, optional
            Crop ratio. If None, uses self.crop_ratio
        
        Returns
        -------
        PIL.Image
            Cropped and resized image
        """
        if image_size is None:
            image_size = self.img_size
        if crop_ratio is None:
            crop_ratio = self.crop_ratio
            
        h, w = image_size
        h_new, w_new = int(h * crop_ratio), int(w * crop_ratio)
        resize = T.Resize([h, w])
        if self.augmentation:
            crop = T.RandomCrop([h_new, w_new])
        else:
            crop = T.CenterCrop([h_new, w_new])
        image = resize(image)
        image = crop(image)
        image = resize(image)
        return image


    def __getitem__(self, index):
        data_path = self.data_paths[index]
        cam_id = self.cam_ids[index]
        obs_frame_ids = self.obs_frame_ids[index]

        # directories
        color_dir = os.path.join(data_path, "cam_{}".format(cam_id), 'color')

        # load colors as 2D images
        colors_list = []
        for frame_id in obs_frame_ids:
            colors = Image.open(os.path.join(color_dir, "{}.png".format(frame_id)))
            # Apply crop resize augmentation
            colors = self.crop_resize_image(colors)
            # Convert PIL Image to numpy array and normalize
            colors = np.array(colors, dtype=np.float32) / 255.0  # (H, W, 3)
            # Apply ImageNet normalization
            colors = (colors - self.img_mean) / self.img_std
            # Convert to (C, H, W) format
            colors = colors.transpose(2, 0, 1)  # (3, H, W)
            colors_list.append(colors)
        
        # Stack images: (num_obs, C, H, W) for colors
        colors_tensor = np.stack(colors_list, axis=0)  # (num_obs, 3, H, W)

        # obs proprio
        obs_proprio_raw = self.obs_proprio_raw[index]
        start_idx = self.obs_proprio_step // 2
        idx_proprio = start_idx + self.obs_proprio_step * np.arange(self.num_obs_proprio)
        obs_proprio = obs_proprio_raw[idx_proprio] # (num_obs_proprio, 7) in base coordinate
        action_tcps = self.tcp_pose[index]
        action_grippers = self.ee_command[index]

        if self.inhand_camera and self.relative_action:
            ref_tcp = obs_proprio[-1].copy()
            obs_proprio = self._project_tcp_base_to_tcp_coord(obs_proprio, ref_tcp)
            action_tcps = self._project_tcp_base_to_tcp_coord(action_tcps, ref_tcp)
        elif self.inhand_camera:
            # The camera remains wrist-mounted, but global action/proprio labels
            # stay in the fixed base frame.  Image coordinates and action
            # coordinates are intentionally independent in this mode.
            obs_proprio = obs_proprio.copy()
            action_tcps = action_tcps.copy()
        else:
            projector = Projector(os.path.join(self.calib_path, getattr(self.config.data, "calibration_file", "result.json")), global_cam_serial=str(self.config.data.cam_id))
            obs_proprio = projector.project_tcp(obs_proprio, source_coord="base", target_coord="camera")
            action_tcps = projector.project_tcp(action_tcps, source_coord="base", target_coord="camera")
            action_tcps = projector.project_tcp(action_tcps, source_coord="camera", target_coord=self.action_coord)
            obs_proprio = projector.project_tcp(obs_proprio, source_coord="camera", target_coord=self.obs_coord)
        # relative action
        if self.relative_action:
            base_absolute_action = obs_proprio[-1].copy()
            action_tcps = absolute_actions_to_relative_actions(action_tcps, base_absolute_action, rotation_rep = "quaternion")

            if self.relative_proprio_obs_for_relative_action:
                obs_proprio = absolute_actions_to_relative_actions(obs_proprio, base_absolute_action, rotation_rep = "quaternion")

        
        # visualization (if needed, can be implemented with matplotlib or other 2D visualization)
        if self.vis:
            try:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, len(obs_frame_ids), figsize=(5*len(obs_frame_ids), 5))
                if len(obs_frame_ids) == 1:
                    axes = [axes]
                for i, frame_id in enumerate(obs_frame_ids):
                    # Denormalize image for visualization
                    img = colors_tensor[i].transpose(1, 2, 0)  # (H, W, 3)
                    img = img * self.img_std + self.img_mean
                    img = np.clip(img, 0, 1)
                    axes[i].imshow(img)
                    axes[i].set_title(f"Frame {frame_id}")
                    axes[i].axis('off')
                plt.tight_layout()
                plt.show()
            except ImportError:
                print("matplotlib not available for visualization")
        
        # rotation transformation (to 6d)
        obs_proprio = xyz_rot_transform(obs_proprio, from_rep = "quaternion", to_rep = "rotation_6d")
        action_tcps = xyz_rot_transform(
            action_tcps,
            from_rep="quaternion",
            to_rep=self.action_rotation_repr,
        )

        actions = np.concatenate((action_tcps, action_grippers), axis = -1)

        # normalization
        # Use separate normalization functions for obs_proprio and action
        obs_proprio_normalized = normalize_obs_proprio(obs_proprio.copy(), config = self.config)
        if self.relative_action:
            actions_normalized = normalize_action(
                actions.copy(), config=self.config
            )
        else:
            actions_normalized = normalize_global_action(
                actions.copy(), config=self.config
            )

        # convert to torch
        colors_tensor = torch.from_numpy(colors_tensor).float()  # (num_obs, 3, H, W)
        obs_proprio_normalized = torch.from_numpy(obs_proprio_normalized).float()
        actions = torch.from_numpy(actions).float()
        actions_normalized = torch.from_numpy(actions_normalized).float()

        if self.num_obs == 1:
            colors_tensor = colors_tensor[0]

        ret_dict = {
            'images': colors_tensor,  # (3, H, W) or (num_obs, 3, H, W)
            'obs_proprio': obs_proprio,
            'obs_proprio_normalized': obs_proprio_normalized,
            'action': actions,
            'action_normalized': actions_normalized
        }
        
        if self.relative_action:
            ret_dict['base_absolute_action'] = base_absolute_action

        return ret_dict
        

def collate_fn_2d(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {}
        for key in batch[0]:
            if key in TO_TENSOR_KEYS:
                ret_dict[key] = collate_fn_2d([d[key] for d in batch])
            else:
                ret_dict[key] = [d[key] for d in batch]
        # Stack image tensors: (batch_size, num_obs, 3, H, W)
        if 'images' in ret_dict:
            ret_dict['images'] = torch.stack(ret_dict['images'], dim=0)
        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        return [sample for b in batch for sample in b]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))
