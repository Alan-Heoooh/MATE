import numpy as np
import open3d as o3d


def create_point_cloud(colors, depths, cam_intrinsics, config, depth_scale=1000.0, voxel_size=0.005):
    """
    Create point cloud from color and depth images.
    
    Args:
        colors: (H, W, 3) RGB image, uint8 or float32
        depths: (H, W) depth image, float32
        cam_intrinsics: (3, 3) camera intrinsic matrix
        config: config object with deploy.workspace and data.normalization fields
        depth_scale: depth scale factor (default: 1000.0)
        voxel_size: voxel size for downsampling (default: 0.005)
    
    Returns:
        cloud_final: (N, 6) numpy array [points (3) + normalized colors (3)]
    """
    h, w = depths.shape
    fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
    cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]

    colors_img = o3d.geometry.Image(colors.astype(np.uint8))
    depths_img = o3d.geometry.Image(depths.astype(np.float32))

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy
    )
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        colors_img, depths_img, depth_scale=depth_scale, convert_rgb_to_intensity=False
    )
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)

    # crop point cloud
    bbox3d = o3d.geometry.AxisAlignedBoundingBox(config.deploy.workspace.min, config.deploy.workspace.max)
    cloud = cloud.crop(bbox3d)

    # downsample
    cloud = cloud.voxel_down_sample(voxel_size)
    points = np.array(cloud.points).astype(np.float32)
    colors = np.array(cloud.colors).astype(np.float32)

    # imagenet normalization
    colors = (colors - config.data.normalization.img_mean) / config.data.normalization.img_std
    # final cloud
    cloud_final = np.concatenate([points, colors], axis=-1).astype(np.float32)
    return cloud_final

