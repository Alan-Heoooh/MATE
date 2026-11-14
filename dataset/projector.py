import os
import json
import numpy as np

from utils.transformation import xyz_rot_to_mat, mat_to_xyz_rot, apply_batch_mat_to_wrench


class Projector:
    def __init__(self, calib_path, global_cam_serial = None, cam_serial = None):
        """
        Initialize Projector with calibration data.
        
        Parameters
        ----------
        calib_path : str
            Path to calibration file (.npy or .json)
        global_cam_serial : str, optional
            Global camera serial number (for .npy format)
        cam_serial : str, optional
            Camera serial number (for .json format, defaults to value in JSON or "default")
        """
        # Check file extension to determine format
        file_ext = os.path.splitext(calib_path)[1].lower()
        
        if file_ext == '.npy':
            # Load from .npy file (original format)
            self.calib_data = np.load(calib_path, allow_pickle = True).item()
            if global_cam_serial is None:
                global_cam_serial = self.calib_data["camera_serials_global"][0]
            else:
                assert global_cam_serial in self.calib_data["camera_serials_global"]

            self.camera_serials = {
                "global": global_cam_serial,
                "inhand": self.calib_data["camera_serial_inhand"]
            }
            self.cam_intrinsics = self.calib_data["intrinsics"]
            self.depth_scales = {}
            for cam_serial_item in self.calib_data["camera_serials"]:
                # TODO: check the depth scale for your own camera
                if cam_serial_item[0] == 'f':   # L515 camera
                    self.depth_scales[cam_serial_item] = 4000.
                else:
                    self.depth_scales[cam_serial_item] = 1000.
            self.cam_to_base = self.calib_data["camera_to_robot"]
            self.global_camera_pose = np.linalg.inv(self.calib_data["camera_to_robot"][global_cam_serial]).astype(np.float32)
            
        elif file_ext == '.json':
            # Load from .json file (new format)
            with open(calib_path, 'r') as f:
                json_data = json.load(f)
            
            # Extract pose information
            pose_in_link = np.array(json_data["pose_in_link"], dtype=np.float32)
            # pose_in_link format: [x, y, z, w, x, y, z] (position + quaternion in wxyz order)
            # This represents the camera pose in the parent_link (world/base) coordinate
            # The quaternion is already in [w, x, y, z] format, so no conversion needed
            
            # Convert pose to transformation matrix
            # pose_in_link is T^parent_cam (parent to camera)
            T_parent_to_cam = xyz_rot_to_mat(
                pose_in_link,
                rotation_rep="quaternion",
                rotation_rep_convention=None
            )
            
            # If parent_link_name is "world" or "base", this is T^base_cam
            # camera_to_robot should be T^cam_base = inv(T^base_cam)
            T_base_to_cam = T_parent_to_cam
            T_cam_to_base = np.linalg.inv(T_base_to_cam).astype(np.float32)
            
            # Set camera serial (prefer from JSON, then parameter, then default)
            if cam_serial is None:
                cam_serial = json_data.get("cam_serial", "default")
            if global_cam_serial is None:
                global_cam_serial = cam_serial
            
            self.camera_serials = {
                "global": global_cam_serial,
                "inhand": cam_serial  # Use same serial for inhand if not specified
            }
            
            # Set intrinsics (prefer from JSON, then parameter)
            if "intrinsics" in json_data:
                # Convert list to numpy array and organize as dict
                intrinsics_array = np.array(json_data["intrinsics"], dtype=np.float32)
                self.cam_intrinsics = {cam_serial: intrinsics_array}
            else:
                raise ValueError("intrinsics must be provided either in JSON file or as parameter when using .json calibration file")
            
            # Set depth scale (prefer from JSON, then parameter, then default)
            self.depth_scales = {cam_serial: json_data.get("depth_scale", 1000.0)}
            
            # Set camera to base transformation
            self.cam_to_base = {cam_serial: T_cam_to_base}
            self.global_camera_pose = T_base_to_cam.astype(np.float32)
            
        else:
            raise ValueError(f"Unsupported calibration file format: {file_ext}. Supported formats: .npy, .json")
        

    
    def project_tcp(self, tcp, cam = None, source_coord = None, target_coord = None, rotation_rep = "quaternion", rotation_rep_convention = None):
        if cam is None:
            cam = self.camera_serials["global"]
        assert source_coord in ["base", "camera"]
        assert target_coord in ["base", "camera"]
        
        if source_coord == target_coord:
            return np.array(tcp, copy=True)
        
        T_src_to_tcp = xyz_rot_to_mat(
            tcp,
            rotation_rep = rotation_rep,
            rotation_rep_convention = rotation_rep_convention
        )

        if source_coord == "base" and target_coord == "camera":
            T_tgt_to_tcp = self.cam_to_base[cam] @ T_src_to_tcp
        elif source_coord == "camera" and target_coord == "base":
            T_tgt_to_tcp = np.linalg.inv(self.cam_to_base[cam]) @ T_src_to_tcp
        else:
            raise ValueError(f"Invalid source_coord or target_coord: {source_coord}, {target_coord}")
        
        return mat_to_xyz_rot(
            T_tgt_to_tcp, 
            rotation_rep = rotation_rep, 
            rotation_rep_convention = rotation_rep_convention
        )
        

        
    def project_wrench(
        self,
        wrench,
        source_coord,
        target_coord,
        tcp = None
    ):
        """
        Transform wrench from source coordinate to target coordinate.
        - wrench: 6D wrench in source coordinate. (6,) or (N, 6)
        - source_coord: source coordinate. (base, camera, tcp)
        - target_coord: target coordinate. (base, camera, tcp)
        - tcp: 4x4 transformation matrix in base coordinate. (4, 4) or (N, 4, 4)
        """
        assert source_coord in ["base", "camera", "tcp"]
        assert target_coord in ["base", "camera", "tcp"]

        if source_coord == target_coord:
            return np.array(wrench, copy=True)
        
        tile_shape = tuple(wrench.shape[:-1]) + (1, 1)
        
        if source_coord == "tcp":
            assert tcp is not None
            T_base_to_source = tcp
            if tcp.ndim == 2:
                T_base_to_source = np.tile(T_base_to_source, tile_shape)
        elif source_coord == "base":
            T_base_to_source = np.tile(np.eye(4, dtype = np.float32), tile_shape)
        else: # camera
            T_base_to_source = np.tile(self.global_camera_pose, tile_shape)
        
        if target_coord == "tcp":
            assert tcp is not None
            T_target_to_base = np.linalg.inv(tcp)
            if tcp.ndim == 2:
                T_target_to_base = np.tile(T_target_to_base, tile_shape)
        elif target_coord == "base":
            T_target_to_base = np.tile(np.eye(4, dtype = np.float32), tile_shape)
        else: # camera
            T_target_to_base = np.tile(np.linalg.inv(self.global_camera_pose), tile_shape)

        return apply_batch_mat_to_wrench(
            wrench,
            batch_mat = np.einsum('...ij,...jk->...ik', T_target_to_base, T_base_to_source)
        )

    def project_pcd(
        self,
        pcd,
        source_coord,
        target_coord,
        tcp=None,
    ):
        """
        Project point cloud between {base, camera, tcp} coordinates.

        Parameters
        ----------
        pcd : (..., 3)
            Point cloud in `source_coord` frame.
        source_coord : {"base", "camera", "tcp"}
        target_coord : {"base", "camera", "tcp"}
        tcp : (4, 4) or (..., 4, 4), optional
            Homogeneous transform T_base_to_tcp (same convention as in project_wrench).
        """
        assert source_coord in ["base", "camera", "tcp"]
        assert target_coord in ["base", "camera", "tcp"]

        pcd = np.asarray(pcd, dtype=np.float32)

        # 同一坐标系，直接返回
        if source_coord == target_coord:
            return pcd

        # 形状：wrench 那套逻辑，按最后一维是坐标维度
        tile_shape = tuple(pcd.shape[:-1]) + (1, 1)

        # ---- 构造 T_base_to_source = T^base_source ----
        if source_coord == "tcp":
            assert tcp is not None, "tcp transform must be provided when source_coord == 'tcp'"
            T_base_to_source = tcp
            if tcp.ndim == 2:
                # (4,4) -> (...,4,4)
                T_base_to_source = np.tile(T_base_to_source, tile_shape)
        elif source_coord == "base":
            # base 自己到自己
            T_base_to_source = np.tile(np.eye(4, dtype=np.float32), tile_shape)
        else:  # "camera"
            # global_camera_pose: T^base_cam
            T_base_to_source = np.tile(self.global_camera_pose, tile_shape)

        # ---- 构造 T_target_to_base = T^target_base ----
        if target_coord == "tcp":
            assert tcp is not None, "tcp transform must be provided when target_coord == 'tcp'"
            T_target_to_base = np.linalg.inv(tcp)
            if tcp.ndim == 2:
                T_target_to_base = np.tile(T_target_to_base, tile_shape)
        elif target_coord == "base":
            T_target_to_base = np.tile(np.eye(4, dtype=np.float32), tile_shape)
        else:  # "camera"
            # inv(global_camera_pose): T^cam_base
            T_target_to_base = np.tile(
                np.linalg.inv(self.global_camera_pose),
                tile_shape,
            )

        # 得到 T^target_source = T^target_base @ T^base_source
        T_target_to_source = np.einsum(
            "...ij,...jk->...ik",
            T_target_to_base,
            T_base_to_source,
        )

        # ---- 对点云做齐次变换 ----
        # p^h_source = [x, y, z, 1]^T
        ones = np.ones_like(pcd[..., :1])
        pcd_h = np.concatenate([pcd, ones], axis=-1)  # (..., 4)

        # p^target = T^target_source @ p^source
        pcd_target_h = np.einsum("...ij,...j->...i", T_target_to_source, pcd_h)
        pcd_target = pcd_target_h[..., :3]

        # 单点的话，返回形状 (3,) 而不是 (1,3)
        if pcd.ndim == 1 and pcd_target.shape[0] == 1:
            return pcd_target[0]

        return pcd_target
