import os
import sys
import torch
import numpy as np
import h5py
import collections.abc as container_abcs

from torch.utils.data import Dataset

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from dataset.constants import TO_TENSOR_KEYS
from utils.dataset_schema import read_lowdim
from utils.runtime_config import resolve_path
from dataset.projector import Projector
from utils.config import get_action_rotation_repr
from utils.transformation import xyz_rot_transform, xyz_rot_to_mat
from utils.normalization import (
    normalize_force_torque,
    normalize_obs_proprio,
    normalize_action,
    normalize_global_action,
)
from utils.action_utils import absolute_actions_to_relative_actions, compute_desired_action_tcps

class RealWorldForceDataset(Dataset):
    """
    Real-world Dataset.
    """
    def __init__(
        self, 
        path, 
        config,
        split = 'train', 
        train_val_filter = (lambda sid: sid % 10 != 0)
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
            demos = sorted(os.listdir(data_path))
            # Apply train/val filter
            if split == 'train':
                # filtered_demos = [sid for sid in demos if train_val_filter(int(sid[-4:]))]
                filtered_demos = demos
            elif split == 'val':
                filtered_demos = [sid for sid in demos if not train_val_filter(int(sid[-4:]))]
            else:
                raise ValueError(f"Invalid split: {split}")
            
            for demo_name in filtered_demos:
                # Create unique identifier using base_path and demo_name
                demo_id = f"{base_path}::{demo_name}"
                if demo_id not in self.all_demos:
                    self.all_demos.append(demo_id)
                    self.demo_id_to_info[demo_id] = (base_path, demo_name)
        
        self.num_demos = len(self.all_demos)
        print(f"Number of {split} demos: {self.num_demos}")

        self.obs_force_torque_raw = []
        self.obs_tcp_pose_raw = []
        self.obs_proprio_raw = []
        self.sample_timestamps = []
        self.action_future_tcp_pose = []
        self.ee_command_future = []
        self.action_force_torque = []

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

            obs_force_torque_list = []
            obs_tcp_pose_list = []
            obs_proprio_list = []
            action_future_tcp_pose_list = []
            ee_command_future_list = []
            action_force_torque_list = []
            sample_timestamp_list = []

            for cur_idx in range(len(timestamp) - 1):
                obs_ft_ids = np.arange(cur_idx - self.num_obs_ft * self.obs_ft_step + self.obs_ft_step // 2, cur_idx + self.obs_ft_step // 2)
                obs_ft_ids = np.clip(obs_ft_ids, 0, len(timestamp) - 1) # (num_obs,)
                obs_proprio_ids = np.arange(cur_idx - self.num_obs_proprio * self.obs_proprio_step + self.obs_proprio_step // 2, cur_idx + self.obs_proprio_step // 2)
                obs_proprio_ids = np.clip(obs_proprio_ids, 0, len(timestamp) - 1) # (num_obs,)
                action_future_ids = np.arange(cur_idx + 1, cur_idx + self.action_horizon_future * self.action_step + 1, self.action_step) # (action_horizon_future,)
                action_future_ids = np.clip(action_future_ids, 0, len(timestamp) - 1) # (action_horizon_future,)

                obs_force_torque_list.append(force_torque[obs_ft_ids])
                obs_tcp_pose_list.append(tcp_pose[obs_ft_ids])
                obs_proprio_list.append(tcp_pose[obs_proprio_ids])
                action_future_tcp_pose_list.append(tcp_pose[action_future_ids])
                ee_command_future_list.append(ee_command[action_future_ids])
                action_force_torque_list.append(force_torque[action_future_ids])
                sample_timestamp_list.append(timestamp[cur_idx])

            self.obs_force_torque_raw += obs_force_torque_list
            self.obs_tcp_pose_raw += obs_tcp_pose_list
            self.obs_proprio_raw += obs_proprio_list
            self.sample_timestamps += sample_timestamp_list
            self.action_future_tcp_pose += action_future_tcp_pose_list
            self.ee_command_future += ee_command_future_list
            self.action_force_torque += action_force_torque_list
        
    def __len__(self):
        return len(self.obs_force_torque_raw)

    def _parse_config(self, config):
        self.num_obs_ft = config.data.num_obs_ft
        self.num_obs_proprio = config.data.num_obs_proprio
        self.obs_ft_freq = config.data.obs_ft_freq
        self.obs_ft_step = int(1000.0 / self.obs_ft_freq)
        self.obs_proprio_freq = config.data.obs_proprio_freq
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
        self.cam_id = config.data.cam_id
        self.translation_min = np.asarray(config.data.normalization.trans_min)
        self.translation_max = np.asarray(config.data.normalization.trans_max)
        self.aug_ft = config.train.augmentation.ft_sample
        self.stiffness = np.array(config.data.stiffness)
        self.action_type = config.data.action_type

        assert self.action_type in ['command', 'state'], "action_type must be either command or state"
        if self.relative_action:
            assert self.obs_coord == self.action_coord, (
                "relative_action=True requires obs_coord == action_coord, "
                f"got obs_coord={self.obs_coord}, action_coord={self.action_coord}. "
                "base_absolute_action is taken from obs_proprio[-1] (in obs_coord) and "
                "applied to actions (in action_coord); mismatched frames would invalidate the relative transform."
            )

    def __getitem__(self, index):
        timestamp = self.sample_timestamps[index]
        # load camera projector
        cam_id = self.cam_id
        projector = Projector(os.path.join(self.calib_path, getattr(self.config.data, "calibration_file", "result.json")), global_cam_serial=str(self.config.data.cam_id))

        # observations
        obs_force_torque_raw = self.obs_force_torque_raw[index]
        obs_tcp_pose_raw = self.obs_tcp_pose_raw[index]
        obs_proprio_raw = self.obs_proprio_raw[index]
        action_force_torque_raw = self.action_force_torque[index]

        if self.split == 'train' and self.aug_ft:
            start_idx = np.random.randint(0, self.obs_ft_step)
        else:
            start_idx = self.obs_ft_step // 2
        idx = start_idx + self.obs_ft_step * np.arange(self.num_obs_ft)
        idx_proprio = start_idx + self.obs_proprio_step * np.arange(self.num_obs_proprio)
        obs_force_torque = obs_force_torque_raw[idx]
        obs_tcp_pose = obs_tcp_pose_raw[idx]
        obs_proprio = obs_proprio_raw[idx_proprio]

        obs_tcps = projector.project_tcp(obs_tcp_pose, cam_id, source_coord = "base", target_coord = "camera") # (num_obs_ft, 7) in camera coordinate
        obs_proprio = projector.project_tcp(obs_proprio, cam_id, source_coord = "base", target_coord = "camera") # (num_obs_proprio, 7) in camera coordinate

        # actions
        action_future_tcps = self.action_future_tcp_pose[index]
        action_future_grippers = self.ee_command_future[index]
        action_future_tcps = projector.project_tcp(action_future_tcps, cam_id, source_coord = "base", target_coord = "camera") # (action_horizon_future, 7) in camera coordinate

        obs_tcps = projector.project_tcp(obs_tcps, cam_id, source_coord = "camera", target_coord = self.obs_coord)
        obs_proprio = projector.project_tcp(obs_proprio, cam_id, source_coord = "camera", target_coord = self.obs_coord)
        action_future_tcps = projector.project_tcp(action_future_tcps, cam_id, source_coord = "camera", target_coord = self.action_coord)
        obs_tcps_base = projector.project_tcp(obs_tcps, cam_id, source_coord = self.obs_coord, target_coord = "base")
        action_future_tcps_base = projector.project_tcp(action_future_tcps, cam_id, source_coord = self.action_coord, target_coord = "base")

        # force_torque coordinate transformation
        if self.force_torque_coord == "tcp":
            # Keep force_torque in tcp coordinate (independent of obs_coord/action_coord)
            obs_force_torque = obs_force_torque.copy()  # already in tcp coordinate
            action_force_torque = action_force_torque_raw.copy()  # already in tcp coordinate
        else:
            # Transform force_torque to obs_coord and action_coord (default behavior)
            obs_force_torque = projector.project_wrench(obs_force_torque, source_coord = "tcp", target_coord = self.obs_coord, tcp = xyz_rot_to_mat(obs_tcps_base, rotation_rep = "quaternion"))
            action_force_torque = projector.project_wrench(action_force_torque_raw, source_coord = "tcp", target_coord = self.action_coord, tcp = xyz_rot_to_mat(action_future_tcps_base, rotation_rep = "quaternion"))

        # compute desired action tcps
        if self.action_type == 'command':
            desired_action_tcps = compute_desired_action_tcps(
                projector,
                action_future_tcps,
                action_force_torque,
                action_future_tcps_base,
                self.force_torque_coord,
                self.action_coord,
                self.stiffness,
            )
            action_future_tcps = desired_action_tcps

        # relative action
        if self.relative_action:
            base_absolute_action = obs_proprio[-1].copy()
            action_future_tcps = absolute_actions_to_relative_actions(action_future_tcps, base_absolute_action, rotation_rep = "quaternion")

            if self.relative_proprio_obs_for_relative_action:
                obs_proprio = absolute_actions_to_relative_actions(obs_proprio, base_absolute_action, rotation_rep = "quaternion")

        # rotation transformation (to 6d)
        obs_tcps = xyz_rot_transform(obs_tcps, from_rep = "quaternion", to_rep = "rotation_6d")
        obs_proprio = xyz_rot_transform(obs_proprio, from_rep = "quaternion", to_rep = "rotation_6d")
        action_future_tcps = xyz_rot_transform(
            action_future_tcps, from_rep="quaternion", to_rep=self.action_rotation_repr
        )
        action_future = np.concatenate((action_future_tcps, action_future_grippers), axis = -1)

        # normalization
        obs_force_torque_normalized = normalize_force_torque(obs_force_torque.copy(), config = self.config)
        obs_proprio_normalized = normalize_obs_proprio(obs_proprio.copy(), config = self.config)
        if self.relative_action:
            action_future_normalized = normalize_action(
                action_future.copy(), config=self.config
            )
        else:
            action_future_normalized = normalize_global_action(
                action_future.copy(), config=self.config
            )

        # convert to torch
        obs_proprio_normalized = torch.from_numpy(obs_proprio_normalized).float() # (num_obs_proprio, 6)
        action_future = torch.from_numpy(action_future).float()
        action_future_normalized = torch.from_numpy(action_future_normalized).float()

        ret_dict = {
            'obs_force_torque': obs_force_torque_normalized,
            'obs_proprio': obs_proprio,
            'obs_proprio_normalized': obs_proprio_normalized,
            'action_future': action_future,
            'action_future_normalized': action_future_normalized,
            'action': action_future,
            'action_normalized': action_future_normalized,
            'timestamp': timestamp,
        }

        return ret_dict
        

def collate_fn_force(batch):
    if type(batch[0]).__module__ == 'numpy':
        return torch.stack([torch.from_numpy(b) for b in batch], 0)
    elif torch.is_tensor(batch[0]):
        return torch.stack(batch, 0)
    elif isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {}
        for key in batch[0]:
            if key in TO_TENSOR_KEYS:
                ret_dict[key] = collate_fn_force([d[key] for d in batch])
            else:
                ret_dict[key] = [d[key] for d in batch]
        return ret_dict
    elif isinstance(batch[0], container_abcs.Sequence):
        return [sample for b in batch for sample in b]
    
    raise TypeError("batch must contain tensors, dicts or lists; found {}".format(type(batch[0])))
