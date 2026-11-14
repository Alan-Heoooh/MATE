#!/usr/bin/env python3
"""Launch the full MATE training and deployment pipelines."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.runtime_config import local_variables, resolve_path, variable

CONFIGS = {
    "vision3d": "configs/components/vision_3d.yaml",
    "pnp3d": "configs/components/pnp_vision_3d.yaml",
    "vision2d": "configs/components/pnp_vision_2d_h64.yaml",
    "force_flip": "configs/components/force_flip.yaml",
    "force_polish": "configs/components/force_polish.yaml",
}


def build_commands(args, extra, values):
    def setting(option, legacy, default):
        return option if option is not None else os.environ.get(legacy, default)
    def path(value):
        return str(resolve_path(value, values))
    task = setting(args.task, "TASK", "pnp" if args.mode == "3d-2d" else "flip")
    if task not in {"flip", "polish", "pnp"}:
        raise ValueError("TASK must be flip, polish, or pnp.")
    checkpoints = resolve_path(variable("MATE_CHECKPOINT_ROOT", "checkpoints", values), values) / task
    python = os.environ.get("PYTHON_BIN", sys.executable)
    env = os.environ.copy()
    gpu_ids = setting(args.gpus, "GPU_IDS", os.environ.get("GPU", "0"))
    gpus = [gpu.strip() for gpu in gpu_ids.split(",")]
    if not all(gpus):
        raise ValueError("GPU_IDS must be a nonempty comma-separated device list.")
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    if args.mode == "train":
        component = setting(args.component, "COMPONENT", "vision3d")
        if component not in {"vision3d", "vision2d", "force"}:
            raise ValueError("COMPONENT must be vision3d, vision2d, or force.")
        if component == "vision2d" and task != "pnp":
            raise ValueError("The supplied vision2d configuration is for TASK=pnp.")
        if component == "force" and task == "pnp":
            raise ValueError("PnP composes vision components; choose vision3d or vision2d.")
        key = "force_" + task if component == "force" else "pnp3d" if component == "vision3d" and task == "pnp" else component
        data_root = resolve_path(variable("MATE_DATA_ROOT", "data", values), values)
        data = setting(args.data_path, "DATA_PATH", variable("MATE_" + task.upper() + "_DATA_PATH", str(data_root / task), values))
        config = setting(args.config, "CONFIG", CONFIGS[key])
        output = setting(args.checkpoint_dir, "CKPT_DIR", str(checkpoints / component))
        if not Path(path(data)).is_dir():
            raise FileNotFoundError("Dataset does not exist: {}".format(path(data)))
        command = [os.environ.get("TORCHRUN_BIN", "torchrun"), "--master_addr", os.environ.get("MASTER_ADDR", "127.0.0.1"),
                   "--master_port", os.environ.get("MASTER_PORT", "17999"), "--nproc_per_node", str(len(gpus)),
                   "train_force.py" if component == "force" else "train.py", "--data_path", path(data),
                   "--config", path(config), "--ckpt_dir", path(output)] + extra
        return [command], env
    if args.mode == "vision-force" and task == "pnp":
        raise ValueError("Use the 3d-2d entrypoint for TASK=pnp.")
    slow_kind = "vision3d"
    fast_kind = "force" if args.mode == "vision-force" else "vision2d"
    slow_ckpt = setting(args.slow_ckpt, "VISION_CKPT" if args.mode == "vision-force" else "SLOW_CKPT", str(checkpoints / slow_kind / "policy_last.ckpt"))
    fast_ckpt = setting(args.fast_ckpt, "FT_CKPT" if args.mode == "vision-force" else "FAST_CKPT", str(checkpoints / fast_kind / "policy_last.ckpt"))
    slow_default = CONFIGS["pnp3d" if args.mode == "3d-2d" else "vision3d"]
    fast_default = CONFIGS["vision2d" if args.mode == "3d-2d" else "force_" + task]
    # Saved training metadata is preferred when supplied beside a checkpoint.
    def checkpoint_config(checkpoint, fallback):
        saved = Path(path(checkpoint)).parent / "hpara.yaml"
        return str(saved) if saved.is_file() else fallback
    slow_config = setting(args.slow_config, "VISION_CONFIG" if args.mode == "vision-force" else "SLOW_CONFIG", checkpoint_config(slow_ckpt, slow_default))
    fast_config = setting(args.fast_config, "FT_CONFIG" if args.mode == "vision-force" else "FAST_CONFIG", checkpoint_config(fast_ckpt, fast_default))
    calib = path(setting(args.calib, "CALIB", variable("MATE_CALIB", "calib", values)))
    if args.mode == "vision-force":
        command = [python, "eval_mate_vision_force.py", "--vision_ckpt", path(slow_ckpt), "--ft_ckpt", path(fast_ckpt),
                   "--vision_config", path(slow_config), "--ft_config", path(fast_config), "--calib", calib]
    else:
        command = [python, "eval_mate_3d_2d.py", "--slow-ckpt", path(slow_ckpt), "--fast-ckpt", path(fast_ckpt),
                   "--slow-config", path(slow_config), "--fast-config", path(fast_config), "--calib", calib,
                   "--slow-hz", "2", "--fast-hz", "5", "--slow-weight-start", "0.5", "--slow-weight-end", "0.5"]
    return [command + extra], env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "vision-force", "3d-2d"])
    for name in ["task", "component", "data-path", "config", "checkpoint-dir", "gpus", "slow-config", "fast-config", "slow-ckpt", "fast-ckpt", "calib"]:
        parser.add_argument("--" + name)
    args, extra = parser.parse_known_args()
    try:
        commands, env = build_commands(args, extra, local_variables())
        for command in commands:
            print(shlex.join(command), flush=True)
            subprocess.run(command, cwd=str(ROOT), env=env, check=True)
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        parser.exit(2, "{}\n".format(error))


if __name__ == "__main__":
    main()
