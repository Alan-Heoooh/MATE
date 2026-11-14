"""
Script to fill lowdim data to obtain 1000hz data.

Authors: Hongjie Fang & Shirun Tang

Usage:
    python scripts/data/fill_hdf5_fields.py
           --data_path data/test
           --fields_info tcp_pose:lowdim:tcp_pose:float32:pose_linear
                         force_torque:lowdim:force_torque:float32:linear
           --save_name lowdim_updated
           --pedal_name lowdim
           --pedal_field pedal_0

Example:
    python scripts/data/fill_hdf5_fields.py --data_path '/path/to/dataset' --fields_info tcp_pose:lowdim:tcp_pose:float32:pose_linear force_torque:lowdim:force_torque:float32:linear ee_command:lowdim:ee_command:float32:linear ee_state:lowdim:ee_state:float32:linear tcp_vel:lowdim:tcp_vel:float32:linear --save_name lowdim_updated --pedal_name lowdim --pedal_field pedal_0
"""

import os
import h5py
import copy
import argparse
import tempfile
import subprocess
import numpy as np

from tqdm import tqdm
from easydict import EasyDict as edict
from typing import Union, Optional, Tuple

import torch
import torch.nn.functional as F


def _is_h5_layout_version_error(exc: Exception) -> bool:
    return "bad version number for layout message" in str(exc).lower()


def _read_dataset_with_python(
    file_path: str,
    dataset_name: str,
    python_exec: str,
) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix = ".npy", delete = False) as tmp:
        tmp_path = tmp.name
    try:
        script = (
            "import sys, h5py, numpy as np\n"
            "file_path, dataset_name, save_path = sys.argv[1:4]\n"
            "with h5py.File(file_path, 'r') as f:\n"
            "    np.save(save_path, f[dataset_name][:], allow_pickle = False)\n"
        )
        subprocess.run(
            [python_exec, "-c", script, file_path, dataset_name, tmp_path],
            check = True,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
        )
        return np.load(tmp_path, allow_pickle = False)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Fallback read failed with python '{python_exec}' for {file_path}:{dataset_name}.\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def read_h5_dataset(
    file_path: str,
    dataset_name: str,
    fallback_python: Optional[str] = None,
) -> np.ndarray:
    try:
        with h5py.File(file_path, "r") as f:
            return f[dataset_name][:]
    except Exception as exc:
        if _is_h5_layout_version_error(exc):
            if fallback_python:
                return _read_dataset_with_python(
                    file_path = file_path,
                    dataset_name = dataset_name,
                    python_exec = fallback_python,
                )
            raise RuntimeError(
                f"Failed to read {file_path}:{dataset_name} with h5py {h5py.__version__}. "
                "This file likely requires newer HDF5 support. "
                "Please upgrade h5py (recommended >= 3.16.0) or pass "
                "--h5py_fallback_python to use another python executable."
            ) from exc
        raise


def _prepare_input(
    x: Union[np.ndarray, torch.Tensor]
) -> Tuple[torch.Tensor, bool]:
    """
    Performance-aware Input Standardization.

    Returns:
        tensor_x: The input converted to a Torch Tensor (Zero-Copy if possible).
        is_numpy: Boolean flag indicating if original input was Numpy.
    """
    if isinstance(x, torch.Tensor):
        return x, False
    tensor_x = torch.as_tensor(x)
    if not tensor_x.is_floating_point():
        tensor_x = tensor_x.float()
    return tensor_x, True


def _to_output(
    tensor_x: torch.Tensor,
    is_numpy: bool
) -> Union[np.ndarray, torch.Tensor]:
    """
    Restore output to original type.
    """
    return tensor_x.detach().cpu().numpy() if is_numpy else tensor_x


def quat_slerp(
    q1: Union[np.ndarray, torch.Tensor],
    q2: Union[np.ndarray, torch.Tensor],
    alpha: Union[float, int, list, np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Spherical linear interpolation between quaternions with PyTorch backend.
    Supports Numpy/Tensor inputs and automatic broadcasting.

    Args:
        q1: (..., 4) First quaternion
        q2: (..., 4) Second quaternion
        alpha: Interpolation parameter. Scalar or Tensor/Array broadcastable to result.
               e.g., if q1 is (4,), alpha can be (N,) -> Result (N, 4)
    """
    q1_t, is_numpy_1 = _prepare_input(q1)
    q2_t, is_numpy_2 = _prepare_input(q2)

    if isinstance(alpha, (float, int, list)):
        alpha_t = torch.tensor(alpha, device = q1_t.device, dtype = q1_t.dtype)
    else:
        alpha_t = torch.as_tensor(alpha, device = q1_t.device, dtype = q1_t.dtype)

    q1_t = F.normalize(q1_t, p = 2, dim = -1)
    q2_t = F.normalize(q2_t, p = 2, dim = -1)

    if alpha_t.ndim > 0 and alpha_t.shape[-1] != 1:
        alpha_t = alpha_t.unsqueeze(-1)

    dot = (q1_t * q2_t).sum(dim = -1, keepdim = True)
    q2_t = torch.where(dot < 0, -q2_t, q2_t)
    theta = torch.acos(torch.clamp(torch.abs(dot), -1.0, 1.0))
    sin_theta = torch.sin(theta)

    safe_sin_theta = torch.where(sin_theta < 1e-6, torch.ones_like(sin_theta), sin_theta)

    ratio_a = torch.sin((1.0 - alpha_t) * theta) / safe_sin_theta
    ratio_b = torch.sin(alpha_t * theta) / safe_sin_theta
    res_slerp = ratio_a * q1_t + ratio_b * q2_t
    res_lerp = F.normalize((1.0 - alpha_t) * q1_t + alpha_t * q2_t, dim = -1)

    res = torch.where(sin_theta < 1e-6, res_lerp, res_slerp)

    return _to_output(res, is_numpy_1 or is_numpy_2)


def interpolate_value(t0, x0, t1, x1, t, interp_type):
    """
    Given data point (t0, x0) and (t1, x1), interpolate x at the given t.
    """
    if t == t0:
        return x0
    if t == t1:
        return x1
    alpha = 0.0 if t0 == t1 else (t - t0) / (t1 - t0)
    if interp_type == "nearest":
        return x1 if alpha >= 0.5 else x0
    elif interp_type == "linear":
        return (1 - alpha) * x0 + alpha * x1
    elif interp_type == "pose_linear":
        pos0, quat0 = np.asarray(x0[:3], dtype = np.float32), np.asarray(x0[3:], dtype = np.float32)
        pos1, quat1 = np.asarray(x1[:3], dtype = np.float32), np.asarray(x1[3:], dtype = np.float32)
        pos = (1 - alpha) * pos0 + alpha * pos1
        quat = quat_slerp(quat0, quat1, alpha)
        return np.concatenate([pos, quat])
    else:
        raise ValueError(f"Unsupported interp_type: {interp_type}")


def seq_fill(
    src_values,
    src_timestamps,
    timestamps,
    dtype,
    interp_type
):
    """
    Fill the sequence given source (value, timestamp) pairs.
    Require that src_timestamps and timestamps are ordered by time.
    """
    dtype = getattr(np, dtype)

    src_idx = 0
    values = []
    for t in timestamps:
        while src_idx < len(src_timestamps) and src_timestamps[src_idx] < t:
            src_idx += 1
        if src_idx == 0:
            values.append(src_values[src_idx])
        elif src_idx == len(src_timestamps):
            values.append(src_values[src_idx - 1])
        else:
            interp_value = interpolate_value(
                t0 = src_timestamps[src_idx - 1],
                x0 = src_values[src_idx - 1],
                t1 = src_timestamps[src_idx],
                x1 = src_values[src_idx],
                t = t,
                interp_type = interp_type
            )
            values.append(interp_value)

    return np.stack(values).astype(dtype)


def interpolate_h5_ms(
    fields_info,
    save_path,
    start_ts = None,
    end_ts = None,
    pedal_path = None,
    pedal_field = None,
    h5py_fallback_python = None,
):
    # calculate valid timestamps (excluding pedal)
    if end_ts - start_ts > 100000000:
        raise ValueError(f"Interval between start_ts ({start_ts}) and end_ts ({end_ts}) is excessively large ({end_ts - start_ts}).\n"
                         "Please check timestamp units or system clock sync across cameras.")
    new_timestamps = np.arange(start_ts, end_ts + 1, 1, dtype = np.int64)
    if pedal_path is not None and pedal_field is not None:
        pedal_values = seq_fill(
            src_timestamps = read_h5_dataset(
                file_path = pedal_path,
                dataset_name = "timestamp",
                fallback_python = h5py_fallback_python,
            ),
            src_values = read_h5_dataset(
                file_path = pedal_path,
                dataset_name = pedal_field,
                fallback_python = h5py_fallback_python,
            ),
            timestamps = new_timestamps,
            dtype = "float32",
            interp_type = "nearest"
        )
        pedal_values = pedal_values.reshape(-1)
        new_timestamps = new_timestamps[pedal_values == 0]

    # generated new interpolated files
    fields_info = edict(fields_info)
    interp_file = h5py.File(save_path, "w")
    interp_file.create_dataset("timestamp", data = new_timestamps, dtype = "int64")

    for field, info in fields_info.items():
        field_values = seq_fill(
            src_timestamps = read_h5_dataset(
                file_path = info.file,
                dataset_name = "timestamp",
                fallback_python = h5py_fallback_python,
            ),
            src_values = read_h5_dataset(
                file_path = info.file,
                dataset_name = info.field,
                fallback_python = h5py_fallback_python,
            ),
            timestamps = new_timestamps,
            dtype = info.dtype,
            interp_type = info.interp_type
        )
        interp_file.create_dataset(field, data = field_values, dtype = field_values.dtype)

    # close files
    interp_file.close()


def parse_fields_info(fields_info_list):
    """
    Parses a list of field info arguments into a dictionary.
    Each element should be a string: field=file,field, dtype, interp_type
    """
    result = {}
    for item in fields_info_list:
        # Format: field:name:field:dtype:interp_type
        parts = item.split(':')
        if len(parts) != 5:
            raise ValueError(f"Invalid field info format: {item}")
        field_name, name, field, dtype, interp_type = parts
        result[field_name] = {
            'file': name,
            'field': field,
            'dtype': dtype,
            'interp_type': interp_type
        }
    return result


def find_ts(scene_path):
    min_ts, max_ts = np.inf, -np.inf
    for cam_dir in sorted(os.listdir(scene_path)):
        if cam_dir[:4] != "cam_":
            continue
        color_list = sorted(os.listdir(os.path.join(scene_path, cam_dir, "color")))
        min_ts = min(min_ts, int(color_list[0][:-4]))
        max_ts = max(max_ts, int(color_list[-1][:-4]))
    return min_ts, max_ts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Interpolate HDF5 file fields with timestamp selection.')
    parser.add_argument('--data_path', required = True, help = 'data path')
    parser.add_argument('--fields_info', nargs = '+', required = True, help = 'List of field info in the format field:name:field:dtype:interp_type')
    parser.add_argument('--save_name', required = True, help = 'Output HDF5 file name')
    parser.add_argument('--pedal_name', help = 'Pedal HDF5 file name (optional)')
    parser.add_argument('--pedal_field', help = 'Pedal field (optional)')
    parser.add_argument(
        '--h5py_fallback_python',
        default = None,
        help = 'Optional python executable for fallback HDF5 reads (e.g. env with h5py>=3.16).',
    )

    args = parser.parse_args()
    fields_info = parse_fields_info(args.fields_info)

    from pathlib import Path
    import sys

    data_dir = Path(args.data_path)
    train_dir = data_dir / "train"
    scene_root = train_dir if train_dir.is_dir() else data_dir

    if not scene_root.exists():
        print(f"Directory {scene_root} does not exist.")
        sys.exit(0)

    scenes = sorted([p for p in scene_root.iterdir() if p.is_dir() and p.name.startswith("scene_")])
    if not scenes:
        print(f"No scene_* directories found in {scene_root}.")
        sys.exit(0)

    for scene_path_obj in tqdm(scenes):
        scene_name = scene_path_obj.name
        scene_path = str(scene_path_obj)
        lowdim_path = str(scene_path_obj / "lowdim")
        start_ts, end_ts = find_ts(scene_path)

        if start_ts == np.inf:
            print(f"\n[Warning] No valid camera timestamps found in {scene_name}. Skipping!")
            continue

        finfo = copy.deepcopy(fields_info)
        for field_name in fields_info.keys():
            finfo[field_name]['file'] = os.path.join(lowdim_path, "{}.h5".format(finfo[field_name]['file']))

        interpolate_h5_ms(
            fields_info = finfo,
            save_path = os.path.join(lowdim_path, "{}.h5".format(args.save_name)),
            start_ts = start_ts,
            end_ts = end_ts,
            pedal_path = os.path.join(lowdim_path, "{}.h5".format(args.pedal_name)) if args.pedal_name else None,
            pedal_field = args.pedal_field,
            h5py_fallback_python = args.h5py_fallback_python,
        )
