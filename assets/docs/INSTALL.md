# Installation

Run commands from the repository root. Use the source trees included under `dependencies/` for native builds.

## Reference environment

Fresh installation was validated on Ubuntu 20.04.6, Python 3.8.20, PyTorch 1.13.0+cu117, torchvision 0.14.0+cu117, CUDA Toolkit 11.7, GCC 9.4, and NVIDIA A100 GPUs. MinkowskiEngine 0.5.4 and PyTorch3D 0.7.9 were compiled from the bundled source trees. Short component training, checkpoint reload/resume, and offline composition checks use this environment; hardware deployment additionally requires the SDKs and site adapter below.

## Python and CUDA dependencies

```bash
conda create -n mate python=3.8 -y
conda activate mate
conda install -c conda-forge openblas-devel -y
python -m pip install pip==24.2
python -m pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
```

The CUDA wheel pair follows [PyTorch's previous-version instructions](https://pytorch.org/get-started/previous-versions/#v1130). The base Torch/torchvision pins in [requirements.txt](../../requirements.txt) accept the `+cu117` wheels. Pip 24.2 supports Python 3.8. On Linux x86_64 / Python 3.8, the PyPI wheels `torch==1.13.0` and `torchvision==0.14.0` were also validated as CUDA 11.7 builds. If the PyTorch wheel index is slow or unavailable, replace its install command with `python -m pip install torch==1.13.0 torchvision==0.14.0 --index-url https://pypi.org/simple`; keep the remaining installation steps unchanged.

Install the Conda prerequisites before the pip packages so Conda does not replace the pinned Python build tools afterward. The requirements file pins direct runtime dependencies, compatibility dependencies, and setuptools/wheel/Ninja build tools; pip resolves their declared transitive dependencies. In particular:

- The Hub pin preserves [Diffusers 0.27.2's `cached_download` import](https://github.com/huggingface/diffusers/blob/v0.27.2/src/diffusers/utils/dynamic_modules_utils.py); that API was removed in [Hub 0.26.0](https://github.com/huggingface/huggingface_hub/releases/tag/v0.26.0).
- SciPy is required through Open3D's scikit-learn dependency, and transforms3d is used by the dataset trimming tool.
- iopath satisfies the bundled PyTorch3D package declaration. Ninja is required by the bundled MinkowskiEngine build; version 1.11.1.3 avoids the [malformed wheel metadata in 1.11.1.1](https://github.com/scikit-build/ninja-python-distributions/issues/256) that fails `pip check` with pip 24.2.

## Native extensions

Install a CUDA toolkit/compiler and driver compatible with the Torch build. PyTorch's runtime does not provide `nvcc`. Replace `CUDA_HOME` below with your toolkit directory.

```bash
export CUDA_HOME=/path/to/cuda-11.7
export PATH="${CUDA_HOME}/bin:${PATH}"
export CUB_HOME="${CUDA_HOME}/include"
export MAX_JOBS=4
export OMP_NUM_THREADS=8
# Optional: compile only for your GPU architecture (A100: 8.0).
# export TORCH_CUDA_ARCH_LIST=8.0
cd dependencies/MinkowskiEngine
python setup.py install --blas=openblas \
  --blas_include_dirs="${CONDA_PREFIX}/include" \
  --blas_library_dirs="${CONDA_PREFIX}/lib"
cd ../..
python -m pip install --no-build-isolation ./dependencies/pytorch3d
```

The MinkowskiEngine snapshot uses its setup entrypoint. Build flags and upstream requirements are documented in the bundled [MinkowskiEngine README](../../dependencies/MinkowskiEngine/README.md) and [PyTorch3D guide](../../dependencies/pytorch3d/INSTALL.md). Rebuild native extensions when changing Python, Torch, or CUDA; retain the bundled source snapshots when reproducing this environment.

## Deployment SDKs and site adapter

Install these separately from the Python requirements:

- `pyrealsense2`, matching the RealSense/librealsense installation: [Intel Python wrapper](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python).
- `flexivrdk`, matching the robot/controller and the API calls in [flexiv_api.py](../../device/robot/flexiv_api.py): [Flexiv RDK](https://github.com/flexivrobotics/flexiv_rdk). No SDK version is validated in the recorded training environment.
- A site module implementing `zero_ft_sensor(robot)` and `move_l_pose(robot, pose, is_blocking)`. Here `robot` is the underlying `flexivrdk.Robot` object. The default module name is `flexiv_robot_interface.flexiv_rizon_interface`; install it on `PYTHONPATH` or set `MATE_ROBOT_INTERFACE_MODULE` in the local configuration.

The site adapter, data, checkpoints, and measured calibration are supplied separately. Continue with [DATA.md](DATA.md) for training inputs or [DEPLOY.md](DEPLOY.md) for hardware setup.
