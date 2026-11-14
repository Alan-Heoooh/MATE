import numpy as np
import open3d as o3d

def create_force_torque_arrow(force_torque, tcp, color = [0, 1, 0]):
    """
    Create a force-torque arrow for visualization.
    """
    force = force_torque[:3]
    force_val = np.linalg.norm(force)
    force_dir = force / (force_val + 1e-6)
    force_arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius = 0.001,
        cone_radius = 0.002,
        cylinder_height = force_val * 0.01,
        cone_height = force_val * 0.01
    )
    
    # Calculate rotation matrix to align arrow with force direction
    # Arrow defaults to pointing along Z-axis
    z_axis = np.array([0, 0, 1])
    v = np.cross(z_axis, force_dir)
    c = np.dot(z_axis, force_dir)
    s = np.linalg.norm(v)
    
    if s > 1e-6:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
        force_arrow.rotate(R, center=(0, 0, 0))

    force_arrow.translate(tcp[:3])
    force_arrow.paint_uniform_color(color)
    return force_arrow
