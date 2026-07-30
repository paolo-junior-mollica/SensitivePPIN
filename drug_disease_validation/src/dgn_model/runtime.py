import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    device_type: str
    lightning_accelerator: str
    uses_ray_gpu: bool
    cuda_device_count: int = 0


def _mps_is_available(torch_module) -> bool:
    backends = getattr(torch_module, "backends", None)
    mps_backend = getattr(backends, "mps", None)
    return bool(mps_backend and mps_backend.is_available())


def get_runtime_config(torch_module) -> RuntimeConfig:
    if torch_module.cuda.is_available():
        return RuntimeConfig(
            device_type="cuda",
            lightning_accelerator="gpu",
            uses_ray_gpu=True,
            cuda_device_count=torch_module.cuda.device_count(),
        )

    if _mps_is_available(torch_module):
        return RuntimeConfig(
            device_type="mps",
            lightning_accelerator="mps",
            uses_ray_gpu=False,
        )

    return RuntimeConfig(
        device_type="cpu",
        lightning_accelerator="cpu",
        uses_ray_gpu=False,
    )


def get_trainer_kwargs(runtime: RuntimeConfig) -> dict:
    return {"accelerator": runtime.lightning_accelerator, "devices": 1}


def get_ray_tune_resources(runtime: RuntimeConfig, workers: int, gpus_per_trial: float) -> dict:
    resources = {"cpu": workers}
    if runtime.uses_ray_gpu and gpus_per_trial > 0:
        resources["gpu"] = gpus_per_trial
    return resources


def get_max_concurrent_trials(runtime: RuntimeConfig, gpus_per_trial: float, cpu_default: int = 8) -> int:
    if runtime.uses_ray_gpu and gpus_per_trial > 0:
        return max(1, int(runtime.cuda_device_count / gpus_per_trial))

    if runtime.device_type == "mps":
        return 1

    return cpu_default


def get_ray_init_kwargs(base_dir=None) -> dict:
    base_path = Path(base_dir) if base_dir else Path(tempfile.gettempdir()) / "digitalhealthlab-sensitive-ppin"
    temp_dir = base_path / "ray_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return {"ignore_reinit_error": True, "_temp_dir": str(temp_dir)}


def get_ray_storage_path(base_dir=None) -> str:
    base_path = Path(base_dir) if base_dir else Path.cwd()
    storage_path = base_path / "ray_results"
    storage_path.mkdir(parents=True, exist_ok=True)
    return str(storage_path)


def get_wandb_start_method(platform=None) -> str:
    current_platform = platform or sys.platform
    return "thread" if current_platform == "darwin" else "fork"


def configure_cuda_visible_devices(gpu_arg, runtime: RuntimeConfig, environ=None) -> None:
    if not runtime.uses_ray_gpu or gpu_arg in (None, ""):
        return

    target_environ = environ if environ is not None else os.environ
    if isinstance(gpu_arg, (list, tuple)):
        target_environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_arg))
    else:
        target_environ["CUDA_VISIBLE_DEVICES"] = str(gpu_arg)
