# Dataset layout and preprocessing

Run the commands here from the repository root. Supply recordings and measured calibration separately.

## Layout and fields

```text
<dataset>/
  calib/result.json
  train/scene_0001/
    lowdim/lowdim_updated.h5
    cam_camera/color/<integer_timestamp>.png
    cam_camera/depth/<integer_timestamp>.png
    cam_wrist_camera/color/<integer_timestamp>.png
  val/scene_0010/
    ...
```

The camera directories use the recorded `data.cam_id`. Public templates select `camera` for the global RGB-D camera and `wrist_camera` for the wrist RGB camera. Set `MATE_DATA_CAMERA_ID` and `MATE_DATA_WRIST_CAMERA_ID` for other recordings; these identifiers are independent of live serials.

Filename timestamps and the low-dimensional `timestamp` field must share units. The loaders use milliseconds and assume **1000 Hz** resampled low-dimensional data when converting observation/action frequencies into sample strides. The templates use a 50 Hz action frequency.

| Logical HDF5 field | Shape | Meaning |
| --- | --- | --- |
| `timestamp` | `(N,)` | Increasing timestamps aligned with image filenames |
| `tcp_pose` | `(N, 7)` | Base-frame position in metres, then wxyz quaternion |
| `ee_command` | `(N, 1)` | Gripper command/width used for action labels |
| `force_torque` | `(N, 6)` | TCP-frame force in N and torque in N m |

Force and 3D loaders require `force_torque`; the wrist RGB loader requires the other three fields. All loaders use `data.hdf5_keys` to map logical signals to the **resampled output file**. For example, the local configuration may contain:

```yaml
variables:
  MATE_TCP_POSE_KEY: recorded_pose
  MATE_FORCE_TORQUE_KEY: recorded_wrench
  MATE_EE_COMMAND_KEY: recorded_gripper
  MATE_DATA_CAMERA_ID: global_camera
  MATE_DATA_WRIST_CAMERA_ID: wrist_camera
```

`MATE_TIMESTAMP_KEY` changes the loader's timestamp key and `MATE_LOWDIM_FILE` changes its default `lowdim_updated.h5` filename. The preprocessing tools below have their own input conventions.

## Splits and component settings

Vision training reads `<dataset>/train/`. Force training reads all scenes under `train/`; its validation loader reads `val/` and selects names whose last four digits are divisible by ten, such as `scene_0010`. Prepare held-out scenes explicitly: the loader does not create a validation split from `train/`.

3D templates use `data.voxel_size`, `train.workspace`, and point/color augmentation settings. Wrist RGB templates use `data.img_size`, `data.crop_ratio`, and `data.vis` for dataset preview. Force templates use `train.augmentation.ft_sample` for temporal sampling. Encoder dimensions live under `model`.

Preserve each trained component's normalization. Composition requires matching action translation bounds and maximum gripper width; see [DEPLOY.md](DEPLOY.md). Measured training calibration is described in [CALIB.md](CALIB.md).

## Preprocessing tools

`fill_hdf5_fields.py` and `trim_static_frames.py` prefer a `train/` subdirectory when given the dataset root; they do not then process `val/`. Pass each split directory explicitly. These tools write files or delete frames, so run them on working copies.

Resample raw `lowdim/lowdim.h5` fields to 1 ms timestamps:

```bash
for split in train val; do
  python scripts/data/fill_hdf5_fields.py \
    --data_path "/path/to/dataset/${split}" \
    --fields_info tcp_pose:lowdim:tcp_pose:float32:pose_linear \
                  ee_command:lowdim:ee_command:float32:linear \
                  force_torque:lowdim:force_torque:float32:linear \
    --save_name lowdim_updated
done
```

Each field specification is `output_key:source_file_stem:source_key:dtype:interpolation`. Source files are read from each scene's `lowdim/` directory and must contain a `timestamp` dataset. Camera image filenames determine the resampling interval. The command writes or overwrites `lowdim/lowdim_updated.h5` with a `timestamp` field plus the selected outputs. Adjust source field names for the raw recording; configure loader mappings for the resulting outputs.

Inspect resampled timing without modifying data:

```bash
python scripts/data/inspect_timestamp_intervals.py \
  --h5 /path/to/dataset/train/scene_0001/lowdim/lowdim_updated.h5
```

Trim static image frames at the start and end of scenes:

```bash
for split in train val; do
  python scripts/data/trim_static_frames.py \
    --data_dir "/path/to/dataset/${split}" --tcp_key tcp_pose
done
```

Trimming reads each scene's raw `lowdim/lowdim.h5`, its `timestamp` field, and the selected TCP key; it removes matching color/depth PNGs in place and keeps the HDF5 file. `--tcp_key` refers to that raw file, independently of the training loader's field mapping.
