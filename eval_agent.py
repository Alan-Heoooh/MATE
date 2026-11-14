"""
Evaluation Agent.
"""

import time
import numpy as np
import flexivrdk
from typing import Optional
from device.robot.flexiv_api import FlexivApi
from utils.transformation import xyz_rot_transform
from utils.lowdim_window import LowDimWindowProvider
from device.camera.realsense import RealSenseRGBDCamera
import importlib
from utils.runtime_config import validate_device_settings


class Agent:
    """
    Evaluation agent with Flexiv arm, Flexiv gripper and Intel RealSense RGB-D camera.

    Follow the implementation here to create your own real-world evaluation agent.
    """
    def __init__(
        self,
        robot_serial,
        camera_serial,
        **kwargs
    ): 
        settings = dict(kwargs, robot_serial=robot_serial, camera_serial=camera_serial)
        validate_device_settings(settings)
        self._ready_pose = np.asarray(kwargs["ready_pose"], dtype=np.float64)
        self._ready_pose_joints = np.asarray(kwargs["ready_pose_joints"], dtype=np.float64)
        self._camera_intrinsics = kwargs.get("camera_intrinsics")
        if self._camera_intrinsics == "":
            self._camera_intrinsics = None
        # Resolve the deployment adapter before enabling a physical robot.
        single = importlib.import_module(kwargs.get(
            "robot_interface_module", "flexiv_robot_interface.flexiv_rizon_interface"
        ))
        self.camera_serial = camera_serial

        print("Init robot, gripper, and camera.")
        self.robot = FlexivApi(serial=robot_serial)
        self.gripper = None
        if bool(kwargs.get("enable_gripper", True)):
            self.gripper = flexivrdk.Gripper(self.robot.robot)
            self.gripper.Enable(kwargs.get("gripper_name", "Flexiv-GN01"))
            if bool(kwargs.get("initialize_gripper", True)):
                self.gripper.Init()
            self.gripper_params = self.gripper.params()
            self.gripper_velocity = float(np.clip(
                kwargs.get("gripper_velocity", 0.1),
                self.gripper_params.min_vel,
                self.gripper_params.max_vel,
            ))
            self.gripper_force_limit = float(np.clip(
                kwargs.get("gripper_force_limit", 20.0),
                self.gripper_params.min_force,
                self.gripper_params.max_force,
            ))

        self.robot.send_joint_position(self.ready_pose_joints)
        time.sleep(5.0)

        single.zero_ft_sensor(self.robot.robot)
        time.sleep(1.0)

        single.move_l_pose(self.robot.robot, pose=self.ready_pose, is_blocking=True)
        time.sleep(1.0)

        # single.zero_ft_sensor(self.robot.robot)
        # time.sleep(2.0)

        # self.robot.send_tcp_pose(self.ready_pose)

        # time.sleep(1.5)

        self.camera = RealSenseRGBDCamera(
            serial=camera_serial, resolution=tuple(kwargs.get("camera_resolution", [1280, 720])),
            frame_rate=int(kwargs.get("camera_frame_rate", 30)),
        )
        for _ in range(30): 
            self.camera.get_rgbd_image()
        print("Initialization Finished.")

        # -------- Optional FT provider --------
        self._lowdim_provider: Optional[LowDimWindowProvider] = None
        enable_lowdim = bool(kwargs.get("enable_lowdim_provider", False))
        lowdim_hz = float(kwargs.get("lowdim_sample_hz", 100.0))
        lowdim_buf_secs = float(kwargs.get("lowdim_buffer_secs", 60.0))
        if enable_lowdim:
            self.enable_lowdim_provider(sample_hz=lowdim_hz, buffer_secs=lowdim_buf_secs)
            time.sleep(2.0)
    
    @property
    def intrinsics(self):
        if self._camera_intrinsics is not None:
            return np.asarray(self._camera_intrinsics, dtype=np.float64)
        return self.camera.intrinsics
    
    @property
    def ready_pose(self):
        return self._ready_pose.copy()
    
    @property
    def ready_pose_joints(self):
        return self._ready_pose_joints.copy()


    def get_observation(self):
        colors, depths = self.camera.get_rgbd_image()
        return colors, depths
    
    def set_tcp_pose(self, pose, rotation_rep, rotation_rep_convention = None, blocking = False):
        tcp_pose = xyz_rot_transform(
            pose,
            from_rep = rotation_rep, 
            to_rep = "quaternion",
            from_convention = rotation_rep_convention
        )
        # print("Moving to pose:", tcp_pose)
        # input("Press Enter to move to the next pose...")
        # self.robot.send_tcp_pose(tcp_pose, max_linear_vel = 0.1, max_linear_acc = 0.3, max_angular_vel = 0.1, max_angular_acc = 0.3)
        self.robot.send_tcp_pose(tcp_pose, max_linear_vel = 0.3, max_linear_acc = 0.3, max_angular_vel = 0.3, max_angular_acc = 0.3)
        
        # self.robot.send_tcp_pose(tcp_pose)
        if blocking:
            time.sleep(0.02)
            # time.sleep(0.05)
            
    def get_tcp_pose(self):
        tcp_pose = self.robot.get_tcp_pose()
        return tcp_pose
    
    def get_tcp_ft(self):
        return self.robot.get_force_torque_tcp()
    
    def set_gripper_width(self, width, blocking = True):
        if self.gripper is None:
            return

        target_width_m = float(np.clip(
            width,
            self.gripper_params.min_width,
            self.gripper_params.max_width,
        ))
        self.gripper.Move(
            target_width_m,
            self.gripper_velocity,
            self.gripper_force_limit,
        )

        if not blocking:
            return

        start_time = time.time()
        while time.time() - start_time <= 0.5:
            time.sleep(0.1)
            if abs(self.get_gripper_width() - target_width_m) < 0.005:
                break
    
    
    def get_gripper_width(self):
        if self.gripper is None:
            return 0.0
        return float(self.gripper.states().width)
    
    def enable_lowdim_provider(self, sample_hz: float = 100.0, buffer_secs: float = 60.0):
        if self._lowdim_provider is not None:
            return
        if LowDimWindowProvider is None:
            raise ImportError("LowDimWindowProvider not found")
        self._lowdim_provider = LowDimWindowProvider(
            get_pose_fn=self.get_tcp_pose,
            get_ft_fn=self.get_tcp_ft,
            sample_hz=sample_hz,
            buffer_secs=buffer_secs
        )

    def disable_lowdim_provider(self):
        if self._lowdim_provider is not None:
            try:
                self._lowdim_provider.stop()
            except Exception:
                pass
            self._lowdim_provider = None

    def get_ft_window(self, *, freq, length, coordinate, projector, remove_first=True):
        """
        Returns a ft:(length,6), tcps:(length, 7) window only if the FT provider is enabled; returns None otherwise.
        """
        if self._lowdim_provider is None:
            print("LowDim provider is not enabled.")
            return None
        return self._lowdim_provider.get_ft_window(
            freq=freq,
            length=int(length),
            coordinate=coordinate,
            projector=projector,
            remove_first=bool(remove_first)
        )

    def get_tcp_window(self, *, freq, length, coordinate, projector, remove_first=True):
        if self._lowdim_provider is None:
            print("LowDim provider is not enabled.")
            return None
        return self._lowdim_provider.get_tcp_window(
            freq=freq,
            length=int(length),
            coordinate=coordinate,
            projector=projector,
            remove_first=bool(remove_first)
        )

    def stop(self):
        self.robot.stop()
        self.disable_lowdim_provider()
        if self.gripper is not None:
            try:
                self.gripper.Stop()
                self.gripper.Disable()
            except Exception:
                pass
