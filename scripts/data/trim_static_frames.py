"""
Remove static frames at the beginning and the end of each scene.

Usage:
    python scripts/data/trim_static_frames.py --data_dir /path/to/data [--tcp_key tcp_pose]

What it does:
    1. Scans the given data_dir for scene_* folders (either directly under data_dir or in data_dir/train/)
    2. Modifies the scene by removing the static image frames at the start and end of the trajectory.
    3. Finds all camera directories automatically.
"""

import re
import argparse
from typing import List
import h5py
import numpy as np
from pathlib import Path
from transforms3d.quaternions import quat2mat
from tqdm import tqdm

# Diff thresholds to determine if the robot has moved
TRANSLATION_THRESHOLD = 0.001
ROTATION_THRESHOLD = 0.05


def _diff_action(action1, action2):
    def _diff_translation(t1, t2, delta):
        dist = np.linalg.norm(t1 - t2)
        return dist > delta
    def _diff_rotation(q1, q2, delta):
        mat1, mat2 = quat2mat(q1), quat2mat(q2)
        rot_diff = np.matmul(mat1, mat2.T)
        rot_diff = (np.diag(rot_diff).sum() - 1) / 2
        rot_diff = min(max(rot_diff, -1), 1)
        rot_diff = np.arccos(rot_diff)
        return rot_diff > delta

    tcp1 = action1[0:7]
    tcp2 = action2[0:7]
    if _diff_translation(tcp1[:3], tcp2[:3], TRANSLATION_THRESHOLD):
        return True
    if _diff_rotation(tcp1[3:7], tcp2[3:7], ROTATION_THRESHOLD):
        return True

    return False


def load_action(lowdim_file, timestamp, tcp_key):
    timestamps = lowdim_file['timestamp']
    idx = np.searchsorted(timestamps, timestamp)
    if idx == len(timestamps):
        idx = idx - 1
    return lowdim_file[tcp_key][idx]


def natural_sort_key(path: Path) -> int:
    """Extract trailing number from a scene folder name for natural sorting."""
    match = re.search(r"(\d+)$", path.name)
    return int(match.group(1)) if match else 0


def collect_scenes(data_dir: Path) -> List[Path]:
    """Auto-detects scene layout and returns a sorted list of scene paths."""
    train_dir = data_dir / "train"
    scene_root = train_dir if train_dir.is_dir() else data_dir

    if not scene_root.exists():
        return []

    scenes = sorted(
        [p for p in scene_root.iterdir() if p.is_dir() and p.name.startswith("scene_")],
        key=natural_sort_key,
    )
    return scenes


def process_scene(path: Path, tcp_key: str):
    lowdim_path = path / "lowdim" / "lowdim.h5"
    if not lowdim_path.exists():
        return None

    # Find all camera folders inside the scene dynamically
    cam_dirs = [d for d in path.iterdir() if d.is_dir() and d.name.startswith("cam_")]
    if not cam_dirs:
        return None

    with h5py.File(lowdim_path, "r") as lowdim_file:
        if tcp_key not in lowdim_file:
            print(f"\n[Warning] {tcp_key} not found in {lowdim_path}")
            return None

        cam_deleted_info = {}

        # Process each camera directory independently
        for cam_dir in cam_dirs:
            color_dir = cam_dir / "color"
            depth_dir = cam_dir / "depth"
            if not color_dir.exists():
                continue

            ts_list = sorted([int(x.stem) for x in color_dir.glob("*.png")])
            if not ts_list:
                continue

            # Trim begin
            start = 0
            while start < len(ts_list) - 1 and not _diff_action(
                load_action(lowdim_file, ts_list[start], tcp_key),
                load_action(lowdim_file, ts_list[start + 1], tcp_key)
            ):
                color_img = color_dir / f"{ts_list[start]}.png"
                depth_img = depth_dir / f"{ts_list[start]}.png"
                if color_img.exists(): color_img.unlink()
                if depth_img.exists(): depth_img.unlink()
                start += 1

            # Trim end
            end = -1
            while abs(end) < len(ts_list) and not _diff_action(
                load_action(lowdim_file, ts_list[end - 1], tcp_key),
                load_action(lowdim_file, ts_list[end], tcp_key)
            ):
                color_img = color_dir / f"{ts_list[end]}.png"
                depth_img = depth_dir / f"{ts_list[end]}.png"
                if color_img.exists(): color_img.unlink()
                if depth_img.exists(): depth_img.unlink()
                end -= 1

            cam_deleted_info[cam_dir.name] = (start, abs(end) - 1)

        return cam_deleted_info


def main():
    parser = argparse.ArgumentParser(description="Remove static early & late frames in scenes.")
    parser.add_argument("--data_dir", type=Path, required=True, help="Data directory containing scenes (either directly or matching data_dir/train/scene_*)")
    parser.add_argument("--tcp_key", type=str, default="tcp_pose", help="Key for tcp pose in lowdim.h5")
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Directory {args.data_dir} does not exist.")
        return

    scenes = collect_scenes(args.data_dir)
    print(f"Found {len(scenes)} scenes in {args.data_dir}")

    pbar = tqdm(scenes)
    for scene_path in pbar:
        result = process_scene(scene_path, args.tcp_key)
        if result:
            info_strs = []
            for cam_name, (d_start, d_end) in result.items():
                info_strs.append(f"{cam_name} (-{d_start}, -{d_end})")
            info_str = " | ".join(info_strs)
            tqdm.write(f"{scene_path.name}: {info_str}")


if __name__ == "__main__":
    main()
