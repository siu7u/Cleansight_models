"""随机种子、设备与运行环境记录（framework 层）。

需求 §6.1/§12.1：记录足以解释本次运行的设备与环境信息。
"""

from __future__ import annotations

import platform
import random
import sys
from datetime import datetime

import numpy as np
import torch


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    """选择训练/评估设备：CUDA > MPS(Apple) > CPU。"""

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def capture_env(device: torch.device, seed: int | None = None) -> dict:
    """捕获可复现所需的环境快照。"""

    info = {
        "timestamp": now_stamp(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "seed": seed,
    }
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        info["cuda_device"] = torch.cuda.get_device_name(idx)
    return info
