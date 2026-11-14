# Deployment

Complete [INSTALL.md](INSTALL.md), [DATA.md](DATA.md), and [CALIB.md](CALIB.md) first. Starting either evaluation entry creates the Agent, enables the robot, and moves it to the configured ready poses **before** the interactive start prompt.

## Machine settings

Copy `configs/local.example.yaml` to `configs/local.yaml` as described in the [README](../../README.md). Supply quoted `MATE_ROBOT_SERIAL`, `MATE_CAMERA_SERIAL`, and, for 3D–2D, `MATE_WRIST_CAMERA_SERIAL` values. Recorded camera IDs may differ from live serials.

`MATE_READY_POSE` is `[x_m, y_m, z_m, qw, qx, qy, qz]` in the robot base frame. `MATE_READY_JOINTS` contains seven joint angles in radians. Set measured task-specific values; both arrays are required by deployment validation.

The Agent reads `deploy.agent` from the force config in vision–force evaluation, falling back to the vision config if that block is absent. In 3D–2D evaluation it reads the slow 3D config. This block controls its RGB-D stream (`camera_resolution`, `camera_frame_rate`), optional measured `camera_intrinsics`, low-dimensional sampling, and gripper settings such as `enable_gripper`, `initialize_gripper`, `gripper_name`, `gripper_velocity`, and `gripper_force_limit`. `MATE_CAMERA_INTRINSICS` overrides the intrinsics; otherwise the live color stream supplies them.

The separate fast RGB camera uses `MATE_WRIST_CAMERA_SERIAL` and the camera wrapper's defaults: 1280×720 at 30 Hz. `data.img_size` and `data.crop_ratio` control policy preprocessing. The fast config also retains robot/ready-pose fields consumed by validation.

Configure the site adapter as described in [INSTALL.md](INSTALL.md). The Agent imports it before enabling the robot.

## Checkpoints and paths

Both pipelines require 10D relative actions with column-wise 6D rotations, a shared action frequency, and matching action translation/gripper normalization. Translation normalization must be zero-centred, positive, and isotropic.

Shell launcher paths resolve from the repository root. Configuration selection is explicit `--slow-config` / `--fast-config`, then the corresponding environment variables, then a sibling `hpara.yaml`, then the component template. Vision–force uses `VISION_CONFIG` / `FT_CONFIG`; 3D–2D uses `SLOW_CONFIG` / `FAST_CONFIG`. Checkpoint environment variables are `VISION_CKPT` / `FT_CKPT` and `SLOW_CKPT` / `FAST_CKPT`, respectively.

Keep each checkpoint beside its training `hpara.yaml`. Checks compare action dimension, frequency, horizon, relative-action flags, rotation representation, and action translation/gripper normalization. Preserve the remaining input normalization and model settings from training as well. Editing YAML does not convert learned checkpoint parameters.

Vision–force reports missing `hpara.yaml` and continues. 3D–2D requires it by default; `--allow-missing-hpara` permits a checkpoint without that metadata. This option only relaxes the missing-file check.

Use `--calib`, `CALIB`, or `MATE_CALIB` for the deployment calibration path; a directory selects `result.json`. See [CALIB.md](CALIB.md) for camera IDs, transforms, and depth scaling. In the 3D config, `deploy.workspace` crops point clouds and `deploy.safe_workspace` supplies base-frame translation bounds for both composition pipelines. Template bounds are experiment values and must match the actual setup.

`PYTHON_BIN` selects the interpreter. `MATE_LOCAL_CONFIG=''` disables local YAML loading; environment variables still apply. Keep machine configuration and generated checkpoint metadata outside the public source release, as configured by `.gitignore`.

## Vision–force

```bash
bash eval_mate_vision_force.sh --task flip \
  --slow-ckpt /path/to/vision/policy_last.ckpt \
  --fast-ckpt /path/to/force/policy_last.ckpt \
  --calib /path/to/deployment/result.json
```

Use `--task polish` for Polish defaults. The target rates come from `vision_config.deploy.vision_freq` (5 Hz) and `ft_config.deploy.infer_freq` (25 Hz). The composed sampler uses 10 DDIM steps, and the entry's control loop runs for a fixed 100 seconds. `model.action_decoder.num_inference_steps` controls standalone component inference.

Extra entry-script flags pass through the launcher unchanged:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--latency_vision_weight_start` | `0.8` | Slow weight at horizon index 0 |
| `--latency_vision_weight_end` | `0.2` | Slow weight at the maximum schedule index |
| `--latency_weight_max_index` | `50` | Maximum schedule index |

For selected-horizon index `i`, `w_s(i) = start + (end - start) * clip(i / max_index, 0, 1)` and the force weight is `1 - w_s(i)`. The supplied fast horizon has indices 0–49, so the default last slow weight is approximately 0.212. An end weight of 0.2 is reached at schedule index 50.

## 3D–2D vision

```bash
bash eval_mate_3d_2d.sh \
  --slow-ckpt /path/to/3d/policy_last.ckpt \
  --fast-ckpt /path/to/2d/policy_last.ckpt \
  --calib /path/to/deployment/result.json
```

The two cameras require distinct serials. The shell launcher supplies 2 Hz / 5 Hz and equal weights. Extra `--slow-hz`, `--fast-hz`, `--slow-weight-start`, and `--slow-weight-end` arguments override these defaults. Direct invocation of `eval_mate_3d_2d.py` uses the config rates and weight defaults 0.8 / 0.2; its default maximum schedule index is `fast_horizon - 1`.

The fast config's `deploy.num_inference_steps` controls composed DDIM steps unless `--num-inference-steps` is supplied. Duration defaults to `fast_config.deploy.max_steps / fast_hz` (200 seconds for the supplied template); `--max-seconds` overrides it.

Use the 64-step RGB checkpoint/config with the 100-step 3D checkpoint for deployment. Full fast-window overlap requires `ceil(action_freq / slow_hz) + fast_horizon <= slow_horizon`. The retained 100-step RGB template is a training option; it cannot fit a delayed 100-step slow window in this sampler.
