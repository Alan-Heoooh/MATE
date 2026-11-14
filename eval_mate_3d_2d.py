"""Asynchronous MATE for a slow 3D and fast 2D vision policy.

The slow encoder runs at 2 Hz and publishes a cached scene feature.  The fast
in-hand RGB encoder and the composed DDIM sampler run at 5 Hz.  At every fast
query, the selected slow noisy-action segment is fused with current fast
guidance. Relative-action diffusion samples and noise are rebased between
the TCP frames at the two observation times.
"""

import argparse
import os
import threading
import time
from copy import deepcopy

import MinkowskiEngine as ME
import numpy as np
import torch

from dataset.projector import Projector
from utils.action_utils import relative_actions_to_absolute_actions
from utils.runtime_config import validate_device_settings
from utils.config import load_config
from utils.constants import GRIPPER_THRESHOLD
from utils.ensemble import EnsembleBuffer
from utils.eval import build_mate_action_head
from utils.eval_input import create_2d_vision_input, create_vision_input
from utils.eval_runtime import pack_action_trajectory_for_robot
from utils.mate import (
    action_horizon,
    latency_aware_horizon_weights,
    observation_offset_steps,
    validate_checkpoint_training_config,
    validate_mate_configs,
)
from utils.normalization import (
    action_translation_scale,
    unnormalize_action,
)
from utils.policy import make_policy
from utils.shared_vision import SharedVisionFeature
from utils.time_utils import precise_sleep
from utils.training import set_seed
from utils.transformation import xyz_rot_transform


class _StoppableWorker(threading.Thread):
    def __init__(self, *, name):
        super().__init__(name=name)
        self.stop_event = threading.Event()
        self.error = None

    def stop(self):
        self.stop_event.set()

    def _sleep_remainder(self, loop_start, period):
        remaining = float(period) - (time.monotonic() - loop_start)
        if remaining > 0:
            precise_sleep(remaining)


class Slow3DWorker(_StoppableWorker):
    def __init__(
        self,
        policy,
        config,
        device,
        agent,
        shared_feature,
        projector,
        *,
        infer_hz,
    ):
        super().__init__(name="mate-slow-3d")
        self.policy = policy
        self.config = config
        self.device = device
        self.agent = agent
        self.shared_feature = shared_feature
        self.projector = projector
        self.period = 1.0 / float(infer_hz)

    def run(self):
        self.policy.eval()
        try:
            with torch.inference_mode():
                while not self.stop_event.is_set():
                    loop_start = time.monotonic()
                    colors, depths = self.agent.get_observation()
                    observation_time = time.monotonic()
                    observation_tcp_base = self.agent.get_tcp_pose()

                    coords, feats, cloud = create_vision_input(
                        colors,
                        depths,
                        cam_intrinsics=self.agent.intrinsics,
                        depth_scale=self.agent.camera.depth_scale,
                        voxel_size=self.config.data.voxel_size,
                        config=self.config,
                        projector=self.projector,
                    )
                    cloud_data = ME.SparseTensor(feats.to(self.device), coords.to(self.device))
                    feature = self.policy.encode(cloud=cloud_data, batch_size=1)
                    self.shared_feature.update(
                        feature,
                        cloud,
                        observation_time=observation_time,
                        vision_obs_tcp_base=observation_tcp_base,
                    )
                    elapsed = time.monotonic() - loop_start
                    print("[slow-3d] feature update {:.3f}s".format(elapsed))
                    self._sleep_remainder(loop_start, self.period)
        except BaseException as exc:
            self.error = exc
            self.stop_event.set()


class Fast2DFusionWorker(_StoppableWorker):
    def __init__(
        self,
        fast_policy,
        action_head,
        slow_config,
        fast_config,
        device,
        agent,
        fast_camera,
        projector,
        shared_slow_feature,
        ensemble_buffer,
        *,
        infer_hz,
        slow_weight_start,
        slow_weight_end,
        weight_max_index,
    ):
        super().__init__(name="mate-fast-2d-fusion")
        self.fast_policy = fast_policy
        self.action_head = action_head
        self.slow_config = slow_config
        self.fast_config = fast_config
        self.device = device
        self.agent = agent
        self.fast_camera = fast_camera
        self.projector = projector
        self.shared_slow_feature = shared_slow_feature
        self.ensemble_buffer = ensemble_buffer
        self.period = 1.0 / float(infer_hz)
        self.action_freq = float(slow_config.data.action_freq)
        self.slow_horizon = action_horizon(slow_config)
        self.fast_horizon = action_horizon(fast_config)
        self.max_offset = self.slow_horizon - self.fast_horizon
        self.position_scale = action_translation_scale(slow_config)
        self.rotation_repr = slow_config.data.action_rotation_repr
        self.minimum_discard_steps = int(
            getattr(fast_config.deploy, "action_discard_steps", 0)
        )
        self.fusion_weights = latency_aware_horizon_weights(
            self.fast_horizon,
            slow_weight_start=slow_weight_start,
            slow_weight_end=slow_weight_end,
            max_index=weight_max_index,
        )

    @staticmethod
    def _rotation(tcp_base):
        matrix = xyz_rot_transform(
            np.asarray(tcp_base), from_rep="quaternion", to_rep="matrix"
        )
        return matrix[:3, :3]

    @staticmethod
    def _position(tcp_base):
        return np.asarray(tcp_base)[:3]

    def _to_robot_chunk(self, normalized_action, slow_tcp_base, fast_observation_time):
        action = unnormalize_action(normalized_action, self.slow_config)
        slow_base = xyz_rot_transform(
            np.asarray(slow_tcp_base),
            from_rep="quaternion",
            to_rep=self.rotation_repr,
        )
        # Relative actions are TCP-local transforms. Applying them to the
        # slow base-frame TCP yields base-frame absolute targets directly.
        action = relative_actions_to_absolute_actions(
            action,
            slow_base,
            rotation_rep=self.rotation_repr,
        )
        action_coord = "base"
        full_action = pack_action_trajectory_for_robot(
            action,
            self.projector,
            camera_serial=self.agent.camera_serial,
            action_coord=action_coord,
            action_rotation_repr=self.rotation_repr,
            clip_safe_workspace=True,
            safe_min=self.slow_config.deploy.safe_workspace.min,
            safe_max=self.slow_config.deploy.safe_workspace.max,
            safe_eps=getattr(self.slow_config.deploy.safe_workspace, "safe_eps", 0.002),
        )

        runtime_steps = int(
            np.ceil(
                max(0.0, time.monotonic() - fast_observation_time)
                * self.action_freq
            )
        )
        discard_steps = max(self.minimum_discard_steps, runtime_steps)
        if discard_steps >= len(full_action):
            raise RuntimeError(
                "Inference latency consumed the full fast horizon: discard={} >= {}.".format(
                    discard_steps, len(full_action)
                )
            )
        return full_action[discard_steps:], discard_steps

    def run(self):
        self.fast_policy.eval()
        self.action_head.eval()
        try:
            with torch.inference_mode():
                while not self.stop_event.is_set():
                    loop_start = time.monotonic()

                    # Snapshot slow guidance first. The subsequent fast frame is
                    # therefore never older than the selected slow observation.
                    slow_feature, _, slow_tcp_base, slow_time = (
                        self.shared_slow_feature.get_latest()
                    )
                    if slow_feature is None:
                        self._sleep_remainder(loop_start, self.period)
                        continue

                    fast_colors = self.fast_camera.get_rgb_image()
                    fast_observation_time = time.monotonic()
                    fast_tcp_base = self.agent.get_tcp_pose()
                    offset_steps = observation_offset_steps(
                        slow_time, fast_observation_time, self.action_freq
                    )
                    if offset_steps > self.max_offset:
                        print(
                            "[fast-2d] skip stale slow feature: offset={} > max={}".format(
                                offset_steps, self.max_offset
                            )
                        )
                        self._sleep_remainder(loop_start, self.period)
                        continue

                    fast_input = create_2d_vision_input(
                        fast_colors, self.fast_config, self.device
                    )
                    fast_feature = self.fast_policy.encode(
                        vision=fast_input, batch_size=1
                    )
                    slow_feature = slow_feature.to(self.device)

                    pred_normalized = self.action_head.predict_action_with_current_fast_guidance(
                        slow_feature,
                        fast_feature,
                        slow_rotation=self._rotation(slow_tcp_base),
                        fast_rotation=self._rotation(fast_tcp_base),
                        slow_position=self._position(slow_tcp_base),
                        fast_position=self._position(fast_tcp_base),
                        offset_steps=offset_steps,
                        fast_horizon=self.fast_horizon,
                        position_scale=self.position_scale,
                        rotation_repr=self.rotation_repr,
                        fusion_weights=self.fusion_weights,
                    ).squeeze(0).cpu().numpy()

                    full_action, discard_steps = self._to_robot_chunk(
                        pred_normalized, slow_tcp_base, fast_observation_time
                    )
                    anchor = self.ensemble_buffer.peek_timestep()
                    self.ensemble_buffer.add_action(full_action, anchor)

                    elapsed = time.monotonic() - loop_start
                    print(
                        "[fast-2d] fused {:.3f}s, slow_offset={} steps, "
                        "runtime_discard={} steps".format(
                            elapsed, offset_steps, discard_steps
                        )
                    )
                    self._sleep_remainder(loop_start, self.period)
        except BaseException as exc:
            self.error = exc
            self.stop_event.set()


def _load_checkpoint(policy, path, device, label):
    state_dict = torch.load(path, map_location=device)
    policy.load_state_dict(state_dict, strict=True)
    print("Loaded {} checkpoint: {}".format(label, path))


def _calibration_file(path):
    path = os.path.expanduser(path)
    return os.path.join(path, "result.json") if os.path.isdir(path) else path


def evaluate(args_override):
    args = deepcopy(args_override)
    slow_config = load_config(args["slow_config"])
    fast_config = load_config(args["fast_config"])
    slow_hz = float(
        args.get("slow_hz")
        or getattr(slow_config.deploy, "infer_freq", 2.0)
    )
    fast_hz = float(
        args.get("fast_hz")
        or getattr(fast_config.deploy, "infer_freq", 5.0)
    )
    summary = validate_mate_configs(
        slow_config,
        fast_config,
        slow_infer_hz=slow_hz,
        fast_infer_hz=fast_hz,
    )
    print("Validated MATE configs: {}".format(summary))

    for label, checkpoint, config in (
        ("slow", args["slow_ckpt"], slow_config),
        ("fast", args["fast_ckpt"], fast_config),
    ):
        saved = validate_checkpoint_training_config(checkpoint, config, label)
        if saved is None:
            message = (
                "{} checkpoint has no sibling hpara.yaml; training/eval "
                "normalization cannot be independently verified.".format(label)
            )
            if not args.get("allow_missing_hpara", False):
                raise FileNotFoundError(message)
            print("Warning: " + message)
        else:
            print("Verified {} checkpoint config: {}".format(label, saved))

    for name in ("slow_ckpt", "fast_ckpt"):
        if not os.path.isfile(args[name]):
            raise FileNotFoundError("Checkpoint file does not exist: {}".format(args[name]))
    if not os.path.isfile(_calibration_file(args["calib"])):
        raise FileNotFoundError("Calibration file does not exist: {}".format(args["calib"]))

    validate_device_settings(dict(slow_config.deploy.agent))
    validate_device_settings(dict(fast_config.deploy.agent))
    seed = int(getattr(slow_config.deploy, "seed", 233))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    slow_policy = make_policy(slow_config, device)
    fast_policy = make_policy(fast_config, device)
    _load_checkpoint(slow_policy, args["slow_ckpt"], device, "slow 3D")
    _load_checkpoint(fast_policy, args["fast_ckpt"], device, "fast 2D")

    num_inference_steps = int(
        args.get("num_inference_steps")
        or getattr(fast_config.deploy, "num_inference_steps", 10)
    )
    action_head = build_mate_action_head(
        slow_policy,
        fast_policy,
        slow_config,
        fast_config,
        modal_weights=[1.0, 1.0],
        device=device,
        num_inference_steps=num_inference_steps,
    )

    slow_agent_cfg = slow_config.deploy.agent
    fast_camera_serial = str(fast_config.deploy.agent.camera_serial)
    if str(slow_agent_cfg.camera_serial) == fast_camera_serial:
        raise ValueError("Slow and fast policies must use distinct camera serials.")

    # Load device SDKs after validating the model and action-space settings.
    try:
        from device.camera.realsense import RealSenseRGBDCamera
        from eval_agent import Agent
    except ImportError as exc:
        raise RuntimeError(
            "Real-robot evaluation requires the deployment environment to "
            "provide pyrealsense2 and the Flexiv SDK. Config validation and "
            "training do not require these packages."
        ) from exc

    agent = Agent(**dict(slow_agent_cfg))
    fast_camera = RealSenseRGBDCamera(serial=fast_camera_serial)
    for _ in range(30):
        fast_camera.get_rgb_image()

    print("Ready TCP pose (base):", agent.get_tcp_pose())
    input("Press Enter to start 3D+2D MATE evaluation...")

    projector = Projector(
        _calibration_file(args["calib"]),
        global_cam_serial=slow_agent_cfg.camera_serial,
    )
    ensemble_buffer = EnsembleBuffer(
        mode=getattr(fast_config.deploy, "ensemble_mode", "new")
    )
    shared_slow_feature = SharedVisionFeature()
    slow_worker = Slow3DWorker(
        slow_policy,
        slow_config,
        device,
        agent,
        shared_slow_feature,
        projector,
        infer_hz=slow_hz,
    )
    fast_worker = Fast2DFusionWorker(
        fast_policy,
        action_head,
        slow_config,
        fast_config,
        device,
        agent,
        fast_camera,
        projector,
        shared_slow_feature,
        ensemble_buffer,
        infer_hz=fast_hz,
        slow_weight_start=args["slow_weight_start"],
        slow_weight_end=args["slow_weight_end"],
        weight_max_index=args.get("weight_max_index"),
    )
    slow_worker.start()
    fast_worker.start()

    control_dt = 1.0 / float(slow_config.data.action_freq)
    max_seconds = args.get("max_seconds")
    if max_seconds is None:
        max_seconds = float(fast_config.deploy.max_steps) / fast_hz
    gripper_threshold = float(
        getattr(fast_config.deploy, "gripper_threshold", GRIPPER_THRESHOLD)
    )
    previous_width = None
    start_time = time.monotonic()

    try:
        while time.monotonic() - start_time < float(max_seconds):
            loop_start = time.monotonic()
            for worker in (slow_worker, fast_worker):
                if worker.error is not None:
                    raise RuntimeError(
                        "{} failed".format(worker.name)
                    ) from worker.error

            step_action = ensemble_buffer.get_action()
            if step_action is None:
                # ``EnsembleBuffer.get_action`` advances its logical timestep
                # even when empty, so empty polling must remain at control rate.
                remaining = control_dt - (time.monotonic() - loop_start)
                if remaining > 0:
                    precise_sleep(remaining)
                continue

            agent.set_tcp_pose(
                step_action[:-1],
                rotation_rep=slow_config.data.action_rotation_repr,
                blocking=False,
            )
            width = float(step_action[-1])
            if previous_width is None or abs(previous_width - width) > gripper_threshold:
                agent.set_gripper_width(width, blocking=False)
                previous_width = width

            remaining = control_dt - (time.monotonic() - loop_start)
            if remaining > 0:
                precise_sleep(remaining)
    finally:
        fast_worker.stop()
        slow_worker.stop()
        fast_worker.join(timeout=5.0)
        slow_worker.join(timeout=5.0)
        try:
            fast_camera.pipeline.stop()
        except Exception:
            pass
        agent.stop()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slow-ckpt", required=True)
    parser.add_argument("--fast-ckpt", required=True)
    parser.add_argument(
        "--slow-config",
        default="configs/components/pnp_vision_3d.yaml",
    )
    parser.add_argument(
        "--fast-config",
        default="configs/components/pnp_vision_2d_h64.yaml",
    )
    parser.add_argument(
        "--calib", default="calib"
    )
    parser.add_argument("--slow-hz", type=float, default=None)
    parser.add_argument("--fast-hz", type=float, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--slow-weight-start", type=float, default=0.8)
    parser.add_argument("--slow-weight-end", type=float, default=0.2)
    parser.add_argument(
        "--weight-max-index",
        type=int,
        default=None,
        help="Default is fast_horizon-1 (63 for the supplied 2D policy).",
    )
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument(
        "--allow-missing-hpara",
        action="store_true",
        help="Allow legacy checkpoints whose training normalization cannot be verified.",
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    evaluate(parse_args())
