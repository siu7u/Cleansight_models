"""随机种子、设备与运行环境记录（framework 层）。

需求 §6.1/§12.1：记录足以解释本次运行的设备与环境信息。
"""

from __future__ import annotations

import platform
import random
import shlex
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import torch


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def now_iso() -> str:
    """返回带本地时区偏移的 ISO 8601 时间，供 schema v2 结果记录。"""

    return datetime.now().astimezone().isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(preferred: str = "auto") -> torch.device:
    """解析显式设备；``auto`` 按 CUDA、MPS、CPU 顺序选择。"""

    if preferred != "auto":
        device = torch.device(preferred)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("请求了 CUDA，但当前环境 torch.cuda.is_available() 为 False")
        if device.type == "mps" and (
            getattr(torch.backends, "mps", None) is None
            or not torch.backends.mps.is_available()
        ):
            raise ValueError("请求了 MPS，但当前环境不可用")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _git_snapshot(repo_root: Path) -> dict:
    """记录代码版本和 dirty 状态，仅作为事实，不执行门禁。"""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.splitlines()
        return {"commit": commit, "dirty": bool(status), "changed_paths": status}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}


def _dependency_snapshot() -> dict[str, str]:
    """以包名到版本的稳定映射记录当前 Python 环境。"""

    packages = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            packages[str(name).lower()] = dist.version
    return dict(sorted(packages.items()))


def capture_env(device: torch.device, seed: int | None = None) -> dict:
    """捕获运行命令、代码、依赖、CUDA/cuDNN 与精度模式。"""

    info = {
        "timestamp": now_stamp(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "seed": seed,
        "command": shlex.join(sys.argv),
        "precision_mode": "fp32",
        "default_dtype": str(torch.get_default_dtype()),
        "git": _git_snapshot(Path(__file__).resolve().parents[3]),
        "dependencies": _dependency_snapshot(),
    }
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        info["cuda_device"] = torch.cuda.get_device_name(idx)
        info["cuda_runtime"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
    return info
