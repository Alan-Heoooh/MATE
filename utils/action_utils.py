import numpy as np

from utils.transformation import xyz_rot_to_mat, mat_to_xyz_rot, ROTATION_REPRESENTATION_DIMS

def absolute_actions_to_relative_actions(actions: np.ndarray, base_absolute_action=None, rotation_rep="rotation_6d", rotation_rep_convention=None):
    actions = actions.copy()
    T, D = actions.shape

    # Calculate pose dimension based on rotation representation
    rot_dim = ROTATION_REPRESENTATION_DIMS[rotation_rep]
    pose_dim = 3 + rot_dim  # xyz + rotation
    
    if D == pose_dim or D == pose_dim + 1:  # (x, y, z, rotation(, gripper_width))
        tcp_dim = np.arange(pose_dim)
    else:
        raise NotImplementedError(f"Unsupported action dimension: {D} for rotation_rep: {rotation_rep}")

    if base_absolute_action is None:
        base_absolute_action = actions[0].copy()
    
    # Convert base pose to homogeneous matrix
    base_pose = base_absolute_action[tcp_dim]  # [pose_dim]
    base_tcp_pose_mat = xyz_rot_to_mat(
        xyz_rot=base_pose[None, :],  # [1, pose_dim]
        rotation_rep=rotation_rep,
        rotation_rep_convention=rotation_rep_convention
    )[0]  # [4, 4]
    
    # Convert actions to homogeneous matrices
    actions_pose = actions[:, tcp_dim]  # [T, pose_dim]
    actions_mat = xyz_rot_to_mat(
        xyz_rot=actions_pose,  # [T, pose_dim]
        rotation_rep=rotation_rep,
        rotation_rep_convention=rotation_rep_convention
    )  # [T, 4, 4]
    
    # Apply relative transformation
    base_tcp_pose_mat_inv = np.linalg.inv(base_tcp_pose_mat)
    relative_mat = base_tcp_pose_mat_inv @ actions_mat  # [T, 4, 4]
    
    # Convert back to pose representation
    relative_pose = mat_to_xyz_rot(
        mat=relative_mat,  # [T, 4, 4]
        rotation_rep=rotation_rep,
        rotation_rep_convention=rotation_rep_convention
    )  # [T, pose_dim]
    
    actions[:, tcp_dim] = relative_pose

    return actions

def relative_actions_to_absolute_actions(actions: np.ndarray, base_absolute_action: np.ndarray, rotation_rep="rotation_6d", rotation_rep_convention=None):
    actions = actions.copy()
    T, D = actions.shape

    # Calculate pose dimension based on rotation representation
    rot_dim = ROTATION_REPRESENTATION_DIMS[rotation_rep]
    pose_dim = 3 + rot_dim  # xyz + rotation
    
    if D == pose_dim or D == pose_dim + 1:  # (x, y, z, rotation(, gripper_width))
        tcp_dim = np.arange(pose_dim)
    else:
        raise NotImplementedError(f"Unsupported action dimension: {D} for rotation_rep: {rotation_rep}")
    
    # Convert base pose to homogeneous matrix
    base_pose = base_absolute_action[tcp_dim]  # [pose_dim]
    base_tcp_pose_mat = xyz_rot_to_mat(
        xyz_rot=base_pose[None, :],  # [1, pose_dim]
        rotation_rep=rotation_rep,
        rotation_rep_convention=rotation_rep_convention
    )[0]  # [4, 4]
    
    # Convert relative actions to homogeneous matrices
    relative_pose = actions[:, tcp_dim]  # [T, pose_dim]
    relative_mat = xyz_rot_to_mat(
        xyz_rot=relative_pose,  # [T, pose_dim]
        rotation_rep=rotation_rep,
        rotation_rep_convention=rotation_rep_convention
    )  # [T, 4, 4]
    
    # Apply absolute transformation
    absolute_mat = base_tcp_pose_mat @ relative_mat  # [T, 4, 4]
    
    # Convert back to pose representation
    absolute_pose = mat_to_xyz_rot(
        mat=absolute_mat,  # [T, 4, 4]
        rotation_rep=rotation_rep,
        rotation_rep_convention=rotation_rep_convention
    )  # [T, pose_dim]
    
    actions[:, tcp_dim] = absolute_pose

    return actions


def compute_desired_action_tcps(
    projector,
    action_tcps: np.ndarray,
    action_force_torque: np.ndarray,
    action_tcps_base: np.ndarray,
    force_torque_coord,
    action_coord: str,
    stiffness: np.ndarray,
) -> np.ndarray:
    """
    Compute desired action TCPs based on force/torque and stiffness.

    Parameters
    ----------
    projector : Projector
        Projector instance used for coordinate transforms.
    action_tcps : np.ndarray
        Action TCPs in action coordinate.
    action_force_torque : np.ndarray
        Action force/torque in either tcp or action coordinate.
    action_tcps_base : np.ndarray
        Action TCPs in base coordinate (for tcp -> action_coord wrench transform when needed).
    force_torque_coord : str or None
        "tcp" to indicate force_torque is in tcp coordinate; otherwise already in action coordinate.
    action_coord : str
        Coordinate name for action representation.
    stiffness : np.ndarray
        Stiffness vector used for compliant offset calculation.
    """
    if force_torque_coord == "tcp":
        # If force_torque is in tcp coordinate, convert to action_coord for calculation.
        action_force_torque_for_calc = projector.project_wrench(
            action_force_torque,
            source_coord="tcp",
            target_coord=action_coord,
            tcp=xyz_rot_to_mat(action_tcps_base, rotation_rep="quaternion"),
        )
    else:
        # force_torque is already in action_coord.
        action_force_torque_for_calc = action_force_torque

    dx = action_force_torque_for_calc[..., :3] / stiffness
    desired_action_tcps = action_tcps.copy()
    desired_action_tcps[..., :3] -= dx

    return desired_action_tcps