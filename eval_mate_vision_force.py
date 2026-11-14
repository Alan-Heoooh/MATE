import time
import argparse
import numpy as np
import open3d as o3d
import torch
import MinkowskiEngine as ME
import threading

from copy import deepcopy
from easydict import EasyDict as edict

from dataset.projector import Projector
from utils.constants import GRIPPER_THRESHOLD
from utils.training import set_seed
from utils.transformation import xyz_rot_transform
from utils.policy import make_policy
from utils.normalization import (
    unnormalize_action,
    action_translation_scale,
)
from utils.ensemble import EnsembleBuffer
from utils.config import load_config, resolve_agent_config
from utils.runtime_config import validate_device_settings, resolve_path
from utils.time_utils import precise_sleep
from utils.action_utils import relative_actions_to_absolute_actions
from utils.eval_input import create_vision_input, create_ft_input, create_proprio_input
from utils.eval_runtime import pack_action_trajectory_for_robot
from utils.eval import build_eval_action_head
from utils.shared_vision import SharedVisionFeature
from utils.mate import (
    action_horizon, validate_mate_configs, latency_aware_horizon_weights,
    validate_checkpoint_training_config,
)

default_args = edict({
    "calib": "calib/",
    "vision_ckpt": None,
    "vision_config": "configs/components/vision_3d.yaml",
    "ft_ckpt": None,
    "ft_config": "configs/components/force_flip.yaml",
    "latency_vision_weight_start": 0.8,
    "latency_vision_weight_end": 0.2,
    "latency_weight_max_index": 50,
})


def project_base_frame_to_obs_coord(tcp_base, config, projector, agent):
    """Project a ``(7,)`` tcp pose given in the robot base frame into ``config.data.obs_coord``.

    Returns None when ``tcp_base`` is None or when relative
    proprio/action mode is disabled (in which case the base is unused).
    """
    if tcp_base is None:
        return None
    if not (config.data.relative_action and config.data.relative_proprio_obs_for_relative_action):
        return None
    pose_np = np.asarray(tcp_base)
    obs_coord = config.data.obs_coord
    if obs_coord == "base":
        return pose_np
    if obs_coord == "camera":
        return projector.project_tcp(
            pose_np,
            source_coord="base",
            target_coord="camera",
            cam=agent.camera_serial,
            rotation_rep="quaternion",
        )
    raise ValueError(
        "Unsupported obs_coord for base-frame projection: {}".format(obs_coord)
    )


class VisionWorker(threading.Thread):
    """
    Dedicated thread for vision processing: continuously fetch observation, 
    process point cloud, encode vision feature, and update shared storage.
    """
    def __init__(self, vision_policy, vision_config, device, agent, shared_vision_feature: SharedVisionFeature):
        super().__init__()
        self.vision_policy = vision_policy
        self.vision_config = vision_config
        self.device = device
        self.agent = agent
        self.shared_vision_feature = shared_vision_feature
        self.model_type = vision_config.model.type

        self.stop_event = threading.Event()

        # Vision feature update frequency, independent of policy inference.
        self.VISION_HZ = getattr(vision_config.deploy, "vision_freq", 5.0)
        self.VISION_DT = 1.0 / self.VISION_HZ

    def stop(self):
        self.stop_event.set()

    def run(self):
        self.vision_policy.eval()
        with torch.inference_mode():
            while not self.stop_event.is_set():
                loop_start = time.time()

                # Get observation
                obs_time = time.time()
                colors, depths = self.agent.get_observation()
                vision_obs_tcp_base = self.agent.get_tcp_pose()

                # Process point cloud
                coords, feats, cloud = create_vision_input(
                    colors,
                    depths,
                    cam_intrinsics=self.agent.intrinsics,
                    voxel_size=self.vision_config.data.voxel_size,
                    config=self.vision_config,
                )
                feats, coords = feats.to(self.device), coords.to(self.device)
                cloud_data = ME.SparseTensor(feats, coords)

                if self.model_type == "rise":
                    # Encode vision feature
                    vision_feature = self.vision_policy.encode(cloud=cloud_data, batch_size=1)
                else:
                    raise ValueError("Unknown model type: {}".format(self.model_type))

                # Update shared storage
                self.shared_vision_feature.update(
                    vision_feature,
                    cloud,
                    observation_time=obs_time,
                    vision_obs_tcp_base=vision_obs_tcp_base,
                )


                # Control frequency
                elapsed = time.time() - loop_start
                print("Vision Worker processing time: {:.3f}s".format(elapsed))
                if elapsed < self.VISION_DT:
                    precise_sleep(self.VISION_DT - elapsed)


class PolicyWorker_Fusion(threading.Thread):
    """
    asynchronous inference thread: keep fetching observation & running policy, push chunks into EnsembleBuffer.
    Uses ensemble of vision_policy and ft_policy.
    Gets latest vision feature from SharedVisionFeature instead of processing it.
    """
    def __init__(
        self, ft_policy, eval_action_head, vision_config, ft_config, device,
        agent, projector, ensemble_buffer, shared_vision_feature,
        latency_vision_weight_start=0.8, latency_vision_weight_end=0.2,
        latency_weight_max_index=50,
    ):
        super().__init__()
        self.ft_policy = ft_policy
        self.eval_action_head = eval_action_head
        self.vision_config = vision_config
        self.ft_config = ft_config
        self.device = device
        self.agent = agent
        self.projector = projector
        self.ensemble_buffer = ensemble_buffer
        self.shared_vision_feature = shared_vision_feature
        self.stop_event = threading.Event()
        self.INFER_HZ = float(ft_config.deploy.infer_freq)
        self.INFER_DT = 1.0 / self.INFER_HZ
        self.action_discard_steps = ft_config.deploy.action_discard_steps
        self.action_translation_scale = action_translation_scale(vision_config)
        self.fusion_weights = latency_aware_horizon_weights(
            action_horizon(ft_config), slow_weight_start=latency_vision_weight_start,
            slow_weight_end=latency_vision_weight_end, max_index=latency_weight_max_index,
        )
        print("MATE vision-force composition initialized at {:.2f} Hz".format(self.INFER_HZ))

    def stop(self):
        self.stop_event.set()


    def _get_observations(self, base_absolute_action=None):
        """Capture the latest ft window and proprio window.

        ``base_absolute_action`` is the pose (in ``config.data.obs_coord``) used
        to relativize the proprio window. In async mode the caller should pass
        the same base pose that will be used to de-relativize predicted actions.
        Force / torque is base-independent once projected to the right
        coordinate frame, so it does not depend on this argument.
        """
        ft_obs_time = time.time()
        ft_data, _, _ = create_ft_input(self.agent, self.ft_config, self.projector, self.device)
        tcp_data, tcp_raw = create_proprio_input(
            self.agent,
            self.ft_config,
            self.projector,
            self.device,
            base_absolute_action=base_absolute_action,
        )
        return ft_data, tcp_data, tcp_raw, ft_obs_time


    def _peek_vision_feature(self):
        vision_feature, cloud, vision_obs_tcp_base, vision_obs_time = self.shared_vision_feature.get_latest()
        if vision_feature is None:
            return None, cloud, vision_obs_tcp_base, vision_obs_time
        vision_feature = vision_feature.to(self.device)
        return vision_feature, cloud, vision_obs_tcp_base, vision_obs_time


    def _encode_ft_feature(self, ft_data, tcp_data):
        return self.ft_policy.encode(force_torque=ft_data, proprio=tcp_data)

    def _tcp_base_to_rotation_matrix(self, tcp_base):
        tcp_mat = xyz_rot_transform(
            np.asarray(tcp_base),
            from_rep="quaternion",
            to_rep="matrix",
        )
        return tcp_mat[:3, :3]

    def _tcp_base_to_position(self, tcp_base):
        return np.asarray(tcp_base)[:3]

    def _predict_action(self, vision_feature, ft_feature, vision_ft_offset_steps, vision_tcp_base=None, ft_tcp_base=None):
        if vision_tcp_base is None or ft_tcp_base is None:
            raise RuntimeError("MATE requires both observation TCP poses for rebasing.")
        pred_raw_action = self.eval_action_head.predict_action_with_current_fast_guidance(
            vision_feature.unsqueeze(0), ft_feature.unsqueeze(0),
            slow_rotation=self._tcp_base_to_rotation_matrix(vision_tcp_base),
            fast_rotation=self._tcp_base_to_rotation_matrix(ft_tcp_base),
            slow_position=self._tcp_base_to_position(vision_tcp_base),
            fast_position=self._tcp_base_to_position(ft_tcp_base),
            offset_steps=vision_ft_offset_steps, fast_horizon=action_horizon(self.ft_config),
            position_scale=self.action_translation_scale,
            rotation_repr=self.vision_config.data.action_rotation_repr,
            fusion_weights=self.fusion_weights,
        ).squeeze(0).cpu().numpy()
        return unnormalize_action(pred_raw_action, self.vision_config)

    def _project_action_base_tcp_to_rot6d(self, tcp_base):
        if tcp_base is None:
            raise RuntimeError("Missing tcp base while converting relative action to absolute action.")
        # Align base absolute action frame with model action_coord explicitly.
        action_coord = self.vision_config.data.action_coord
        if action_coord not in ["base", "camera"]:
            raise ValueError("Unsupported action_coord for tcp projection: {}".format(action_coord))
        base_absolute_action = self.projector.project_tcp(
            tcp_base,
            source_coord="base",
            target_coord=action_coord,
            cam=self.agent.camera_serial,
            rotation_rep="quaternion",
        )
        return xyz_rot_transform(
            base_absolute_action,
            from_rep="quaternion",
            to_rep=self.vision_config.data.action_rotation_repr
        )

    def _to_absolute_action_if_needed(self, action, slow_tcp_base):
        # The sampler maps the fused noise back into the cached slow policy frame.
        slow_base = self._project_action_base_tcp_to_rot6d(slow_tcp_base)
        return relative_actions_to_absolute_actions(
            action, slow_base, rotation_rep=self.vision_config.data.action_rotation_repr,
        )

    def _project_to_base_and_pack(self, action):
        # action: [num_action, 3(trans)+6(rot6d)+1(width)]
        full_action = pack_action_trajectory_for_robot(
            action,
            self.projector,
            camera_serial=self.agent.camera_serial,
            action_coord=self.vision_config.data.action_coord,
            action_rotation_repr=self.vision_config.data.action_rotation_repr,
            clip_safe_workspace=True,
            safe_min=self.vision_config.deploy.safe_workspace.min,
            safe_max=self.vision_config.deploy.safe_workspace.max,
            safe_eps=getattr(self.vision_config.deploy.safe_workspace, "safe_eps", 0.002),
        )
        full_action = full_action[self.action_discard_steps:, :]
        return full_action

    def _visualize(self, action, tcp_raw, cloud):
        if not self.vision_config.deploy.vis or cloud is None:
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(
            cloud[:, 3:] * self.vision_config.data.normalization.img_std + self.vision_config.data.normalization.img_mean
        )
        tcp_vis_list = []
        for raw_tcp in action:
            tcp_vis = o3d.geometry.TriangleMesh.create_sphere(0.003).translate(raw_tcp[:3])
            tcp_vis_list.append(tcp_vis)

        obs_tcp_vis = self.projector.project_tcp(
            tcp_raw,
            source_coord=self.ft_config.data.obs_coord,
            target_coord="camera",
            cam=self.agent.camera_serial,
            rotation_rep="quaternion",
        )
        for raw_tcp in obs_tcp_vis:
            tcp_vis = o3d.geometry.TriangleMesh.create_sphere(0.003).translate(raw_tcp[:3])
            tcp_vis.paint_uniform_color(np.array([0.0, 1.0, 0.0]))
            tcp_vis_list.append(tcp_vis)

        curr_tcp_pose = self.projector.project_tcp(
            self.agent.get_tcp_pose(),
            source_coord='base',
            target_coord=self.vision_config.data.obs_coord
        )
        curr_tcp_vis = o3d.geometry.TriangleMesh.create_sphere(0.003)
        curr_tcp_vis.translate(curr_tcp_pose[:3])
        curr_tcp_vis.paint_uniform_color(np.array([0.0, 0.0, 1.0]))
        tcp_vis_list.append(curr_tcp_vis)

        o3d.visualization.draw_geometries([pcd, *tcp_vis_list])

    def run(self):
        self.ft_policy.eval()
        self.eval_action_head.eval()
        with torch.inference_mode():
            while not self.stop_event.is_set():
                loop_start = time.time()

                # --------- 1. peek the latest vision feature first, so we know
                #              which tcp frame to relativize proprio against ---
                vision_feature, cloud, vision_obs_tcp_base, vision_obs_time = self._peek_vision_feature()
                if vision_feature is None:
                    print("Warning: No vision feature available yet, skipping inference...")
                    time.sleep(0.01)
                    continue

                current_ft_tcp_base = self.agent.get_tcp_pose()
                proprio_base_in_obs_coord = project_base_frame_to_obs_coord(
                    current_ft_tcp_base,
                    self.ft_config,
                    self.projector,
                    self.agent,
                )

                # --------- 2. build FT + proprio with the model's relative-action base -----
                ft_data, tcp_data, tcp_raw, ft_obs_time = self._get_observations(
                    base_absolute_action=proprio_base_in_obs_coord,
                )
                ft_feature = self._encode_ft_feature(ft_data, tcp_data)

                print("Get obs features time: {:.3f}s".format(time.time() - loop_start))
                print("Vision observation time: {:.6f}".format(vision_obs_time))

                # Calculate vision-FT observation time offset in steps
                action_dt = 1.0 / self.vision_config.data.action_freq
                vision_ft_offset_steps = int(round((ft_obs_time - vision_obs_time) / action_dt))
                if vision_ft_offset_steps < 0 or vision_ft_offset_steps + action_horizon(self.ft_config) > action_horizon(self.vision_config):
                    time.sleep(0.001)
                    continue


                # Inference
                print("Inference...")
                start_inference_time = time.time()
                action = self._predict_action(
                    vision_feature,
                    ft_feature,
                    vision_ft_offset_steps,
                    vision_tcp_base=vision_obs_tcp_base,
                    ft_tcp_base=current_ft_tcp_base,
                )
                print("Action head forwarding time: {:.3f}s".format(time.time() - start_inference_time))
                print("Vision-FT offset: {} steps".format(vision_ft_offset_steps))
                action = self._to_absolute_action_if_needed(action, vision_obs_tcp_base)

                # --------- 3. camera → base, safety clipping ---------
                full_action = self._project_to_base_and_pack(action)

                # print("full_action[0] (xyz): ", full_action[0, :3])
                # print("current tcp (xyz): ", self.agent.get_tcp_pose()[:3])
                # print("delta pos (xyz): ", full_action[0, :3] - self.agent.get_tcp_pose()[:3])

                # --------- 4. temporal ensemble: anchor must be peeked HERE, not at loop start.
                # Long vision+FT+fusion latency lets the control thread advance ``timestep`` while
                # this inference runs; a stale anchor places chunks behind the consumer and
                # ``get_action`` never applies them (robot appears frozen).
                infer_anchor_t = self.ensemble_buffer.peek_timestep()
                self.ensemble_buffer.add_action(full_action, infer_anchor_t)

                self._visualize(action, tcp_raw, cloud)

                elapsed = time.time() - loop_start
                if elapsed < self.INFER_DT:
                    precise_sleep(self.INFER_DT - elapsed)


def evaluate(args_override):
    # 1. load args and device
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    vision_config = load_config(args.vision_config)
    ft_config = load_config(args.ft_config)
    summary = validate_mate_configs(
        vision_config, ft_config,
        slow_infer_hz=float(getattr(vision_config.deploy, "vision_freq", 5.0)),
        fast_infer_hz=float(ft_config.deploy.infer_freq),
    )
    print("Validated MATE configs: {}".format(summary))

    for label, checkpoint, config in (
        ("vision", args.vision_ckpt, vision_config),
        ("force", args.ft_ckpt, ft_config),
    ):
        if checkpoint is None:
            raise ValueError("Please provide the {} checkpoint.".format(label))
        saved = validate_checkpoint_training_config(checkpoint, config, label)
        if saved is None:
            print("Warning: {} checkpoint has no sibling hpara.yaml; verify its training representation before deployment.".format(label))

    for checkpoint in (args.vision_ckpt, args.ft_ckpt):
        if not resolve_path(checkpoint).is_file():
            raise FileNotFoundError("Checkpoint file does not exist: {}".format(checkpoint))
    calibration_path = resolve_path(args.calib)
    if calibration_path.is_dir():
        calibration_path = calibration_path / "result.json"
    if not calibration_path.is_file():
        raise FileNotFoundError("Calibration file does not exist: {}".format(calibration_path))
    args.calib = str(calibration_path)
    validate_device_settings(dict(resolve_agent_config(vision_config, ft_config).agent))

    # Use vision_config deploy.seed if available, otherwise ft_config
    seed = getattr(getattr(vision_config, "deploy", edict()), "seed", 
                   getattr(getattr(ft_config, "deploy", edict()), "seed", 233))
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. load policies
    print("Loading policies ...")
    vision_policy = make_policy(vision_config, device)
    ft_policy = make_policy(ft_config, device)
    
    vision_n_parameters = sum(p.numel() for p in vision_policy.parameters() if p.requires_grad)
    ft_n_parameters = sum(p.numel() for p in ft_policy.parameters() if p.requires_grad)
    print("Number of Vision policy parameters: {:.2f}M".format(vision_n_parameters / 1e6))
    print("Number of FT policy parameters: {:.2f}M".format(ft_n_parameters / 1e6))

    assert args.vision_ckpt is not None, "Please provide the vision checkpoint to evaluate."
    assert args.ft_ckpt is not None, "Please provide the ft checkpoint to evaluate."
    vision_policy.load_state_dict(torch.load(args.vision_ckpt, map_location=device), strict=False)
    print("Vision checkpoint {} loaded.".format(args.vision_ckpt))
    ft_policy.load_state_dict(torch.load(args.ft_ckpt, map_location=device), strict=False)
    print("FT checkpoint {} loaded.".format(args.ft_ckpt))

    # 3. build fused action head
    eval_action_head = build_eval_action_head(
        vision_policy=vision_policy,
        ft_policy=ft_policy,
        vision_config=vision_config,
        ft_config=ft_config,
        modal_weights=[1.0, 1.0],
        device=device,
    )

    # 4. initialize Agent / Projector / EnsembleBuffer
    agent_config = resolve_agent_config(vision_config, ft_config)
    
    from eval_agent import Agent
    agent = Agent(**dict(agent_config.agent))

    print("ready tcp pose (base):", agent.get_tcp_pose())

    input("Press Enter to start evaluation...")

    projector = Projector(
        args.calib,  
        global_cam_serial=agent_config.agent.camera_serial
    )
    ensemble_mode = getattr(
        ft_config.deploy, "ensemble_mode", getattr(vision_config.deploy, "ensemble_mode", "new")
    )
    max_past_inference_chunks = getattr(
        ft_config.deploy,
        "max_past_inference_chunks",
        getattr(vision_config.deploy, "max_past_inference_chunks", None),
    )
    ensemble_buffer = EnsembleBuffer(
        mode=ensemble_mode,
        max_past_inference_chunks=max_past_inference_chunks,
    )
    shared_vision_feature = SharedVisionFeature()

    # 5. start vision worker thread (processes vision continuously)
    vision_worker = VisionWorker(
        vision_policy,
        vision_config,
        device,
        agent,
        shared_vision_feature,
    )
    vision_worker.start()

    # 6. start asynchronous inference thread
    policy_worker = PolicyWorker_Fusion(
        ft_policy, 
        eval_action_head,
        vision_config, 
        ft_config, 
        device, 
        agent, 
        projector, 
        ensemble_buffer, 
        shared_vision_feature,
        latency_vision_weight_start=args.latency_vision_weight_start,
        latency_vision_weight_end=args.latency_vision_weight_end,
        latency_weight_max_index=args.latency_weight_max_index,
    )
    policy_worker.start()

    # 7. control loop parameters & smoothing state
    CTRL_HZ = vision_config.data.action_freq
    CTRL_DT = 1.0 / CTRL_HZ
    MAX_TIME = 100.0 # total running time, adjust it yourself
    gripper_threshold = float(
        getattr(agent_config, "gripper_threshold", GRIPPER_THRESHOLD)
    )
    prev_width = None

    start_time = time.time()

    try:
        while time.time() - start_time < MAX_TIME:
            loop_start = time.time()


            # get one step_action from temporal ensemble ([3+6+1])
            step_action = ensemble_buffer.get_action()
            if step_action is None:
                # no action in the buffer, can sleep for a while
                time.sleep(0.0001)
                continue

            target_tcp = step_action[:-1]   # 3+6
            step_width = step_action[-1]

            agent.set_tcp_pose(
                target_tcp,
                rotation_rep=vision_config.data.action_rotation_repr,
                blocking=False,
            )

            # send gripper width (thresholding to avoid repeating signals)
            if prev_width is None or abs(prev_width - step_width) > gripper_threshold:
                agent.set_gripper_width(step_width, blocking=False)
                prev_width = step_width

            # keep fixed control frequency
            elapsed = time.time() - loop_start
            if elapsed < CTRL_DT:
                precise_sleep(CTRL_DT - elapsed)

    finally:
        vision_worker.stop()
        vision_worker.join()
        policy_worker.stop()
        policy_worker.join()
        agent.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--calib', action='store', type=str, help='calibration path', required=True)
    parser.add_argument('--vision_ckpt', action='store', type=str, help='vision checkpoint path', required=True)
    parser.add_argument('--vision_config', action='store', type=str, help='vision config file', required=True)
    parser.add_argument('--ft_ckpt', action='store', type=str, help='force-torque checkpoint path', required=True)
    parser.add_argument('--ft_config', action='store', type=str, help='force-torque config file', required=True)
    parser.add_argument(
        '--latency_vision_weight_start',
        action='store',
        type=float,
        default=0.8,
        help='vision weight at selected-horizon index 0',
    )
    parser.add_argument(
        '--latency_vision_weight_end',
        action='store',
        type=float,
        default=0.2,
        help='vision weight at the maximum selected-horizon index',
    )
    parser.add_argument(
        '--latency_weight_max_index',
        action='store',
        type=int,
        default=50,
        help='maximum selected-horizon index for the linear composition weights',
    )
    _args = vars(parser.parse_args())
    evaluate(_args)
