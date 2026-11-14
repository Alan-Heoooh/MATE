"""Flexiv robot ethernet communication python wrapper 2.8.2 api.

Author: Junfeng Ding, Hongjie Fang, Hao-Shu Fang
"""

import time

import flexivrdk
import numpy as np


class FlexivApi:
    """Read TCP pose/wrench and command Cartesian or joint motion."""


    def __init__(self, serial):
        """initialize.

        Args:
            serial: robot serial number

        Raises:
            RuntimeError: an existing robot fault cannot be cleared.
        """
        self.mode = flexivrdk.Mode
        self.robot = flexivrdk.Robot(serial)
        if self.robot.fault():
            if not self.robot.ClearFault():
                raise RuntimeError("Cannot clear faults on the robot.")
        self.robot.Enable()
        while not self.robot.operational():
            time.sleep(0.5)
        self.current_mode = None
        time.sleep(2)


    def _get_robot_status(self):
        return self.robot.states()


    def switch_mode(self, mode):
        mode = getattr(self.mode, mode)
        if mode != self.current_mode:
            self.robot.SwitchMode(mode)
            self.current_mode = mode

        
    def get_tcp_pose(self):
        """get current robot's tool pose in world frame.

        Returns:
            7-dim list consisting of (x,y,z,rw,rx,ry,rz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().tcp_pose)


    def get_force_torque_tcp(self):
        """get current robot's tool force torque wrench in TCP frame.

        Returns:
            6-dim list consisting of (fx,fy,fz,wx,wy,wz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().ext_wrench_in_tcp[:6])


    def stop(self):
        """Stop current motion and switch mode to idle."""
        self.robot.Stop()


    def send_impedance_online_pose(
        self, tcp, max_linear_vel = 0.5, max_linear_acc = 1.0, max_angular_vel = 0.5, max_angular_acc = 1.0
    ):
        """make robot move towards target pose in impedance control mode,
        combining with sleep time makes robot move smmothly.

        Args:
            tcp: 7-dim list or numpy array, target pose (x,y,z,rw,rx,ry,rz) in world frame

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        self.switch_mode("NRT_SUPER_PRIMITIVE")
        pose = np.array(tcp)
        self.robot.SendCartesianMotionForce(pose, max_linear_vel = max_linear_vel, max_linear_acc = max_linear_acc, max_angular_vel = max_angular_vel, max_angular_acc = max_angular_acc)

    def send_tcp_pose(
        self, tcp, max_linear_vel = 0.5, max_linear_acc = 1.0, max_angular_vel = 0.5, max_angular_acc = 1.0
    ):
        """make api align with 2.6
        """
        self.send_impedance_online_pose(tcp, max_linear_vel = max_linear_vel, max_linear_acc = max_linear_acc, max_angular_vel = max_angular_vel, max_angular_acc = max_angular_acc)


    def send_joint_position(
        self,
        target_jnt_pos,
        target_jnt_vel=[0, 0, 0, 0, 0, 0, 0],
        max_jnt_vel=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        max_jnt_acc=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    ):
        """make robot move towards target joint position in position control
        online mode without reach check.

        Args:
            target_jnt_pos: 7-dim list or numpy array, target joint position of 7 joints
            target_jnt_vel: 7-dim list or numpy array, target joint velocity of 7 joints
            max_jnt_vel: 7-dim list or numpy array, maximum joint velocity of 7 joints
            max_jnt_acc: 7-dim list or numpy array, maximum joint acceleration of 7 joints

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        self.switch_mode("NRT_JOINT_IMPEDANCE")
        self.robot.SetJointImpedance(np.array(self.robot.info().K_q_nom) * 0.1)
        self.robot.SendJointPosition(
            target_jnt_pos,
            target_jnt_vel,
            max_jnt_vel,
            max_jnt_acc
        )
