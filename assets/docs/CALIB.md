# Calibration

## Files and camera IDs

Dataset loaders default to `<dataset>/calib/result.json`. `data.calib_path` / `MATE_DATA_CALIB` selects another directory; `data.calibration_file` / `MATE_CALIB_FILE` changes the filename. Deployment uses `--calib`, `CALIB`, or `MATE_CALIB`; a directory always selects `result.json`, so pass a full path for another filename or format.

The calibration's `cam_serial` must match `data.cam_id` during training and the Agent's live camera serial during deployment. When recorded and live IDs differ, keep separate training and deployment calibration files with the appropriate identifier and the same measured transform. Camera directory names are `cam_<data.cam_id>`.

## Global-camera JSON

[Projector](../../dataset/projector.py) reads this schema:

```json
{
  "cam_serial": "camera",
  "pose_in_link": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
  "intrinsics": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
  "depth_scale": 1000.0
}
```

These are schema examples. Replace them with measured calibration values:

- `pose_in_link`: camera pose in robot base coordinates, with translation in metres and rotation as a wxyz quaternion. Its transform maps camera points into the base frame; the projector stores the inverse for base-to-camera projection.
- `intrinsics`: measured 3×3 matrix for the recorded aligned RGB-D images and their resolution.
- `depth_scale`: raw depth units per metre for the recorded depth images.

The 3D training loader reads intrinsics and depth scale from this calibration. Live evaluation reads intrinsics from the Agent's configured matrix or the live color stream. The 3D–2D entry forwards the camera wrapper's `depth_scale`; the vision–force entry currently uses the point-cloud helper's default of 1000.0. A JSON depth scale therefore does not override live evaluation scaling.

Re-measure the transform when the camera, robot base, or mount moves. The TCP-relative wrist RGB encoder uses color images without a global-camera extrinsic; the slow 3D branch still requires its calibration.

## NumPy calibration

`Projector` also accepts the trusted `.npy` dictionary format containing `camera_serials_global`, `camera_serial_inhand`, `camera_serials`, `intrinsics`, and `camera_to_robot`. Its `camera_to_robot` matrices follow the base-to-camera convention, so use the inverse of the JSON pose transform for that field. Keys in its per-camera maps must match the configured camera ID.
