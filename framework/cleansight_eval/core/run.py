"""训练/评估运行的统一组织（framework 层）。

一个 ``RunContext`` 负责：分配 run_id、组织运行目录、落盘解析后的配置与环境
快照。训练与评估共享同一套目录语义，便于矩阵层后续汇总（需求 §5.5/§6.1）。
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import torch

from .environment import capture_env, now_stamp


class RunContext:
    def __init__(self, root: str | Path, label: str, run_id: str | None = None):
        # label 只是 run 目录名前缀（通常取 model.type），与模型语义无关。
        root = Path(root)
        if run_id is None:
            base_run_id = f"{label}-{now_stamp()}"
            candidate = root / base_run_id
            suffix = 1
            while candidate.exists():
                candidate = root / f"{base_run_id}-{suffix}"
                suffix += 1
            self.run_id = candidate.name
            self.dir = candidate
        else:
            self.run_id = run_id
            self.dir = root / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.label = label

    @property
    def checkpoints_dir(self) -> Path:
        d = self.dir / "checkpoints"
        d.mkdir(exist_ok=True)
        return d

    @property
    def evals_dir(self) -> Path:
        d = self.dir / "evals"
        d.mkdir(exist_ok=True)
        return d

    def save_config(self, cfg: dict) -> None:
        (self.dir / "config.resolved.json").write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def save_env(self, device: torch.device, seed: int | None = None) -> None:
        (self.dir / "env.json").write_text(
            json.dumps(capture_env(device, seed), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @property
    def history_path(self) -> Path:
        return self.dir / "history.csv"

    @property
    def status_path(self) -> Path:
        return self.dir / "status.json"

    def write_status(self, state: str, **fields: Any) -> None:
        """写训练运行状态，异常中断时也保留可诊断事实。"""

        payload = {
            "schema_version": 1,
            "run_id": self.run_id,
            "label": self.label,
            "state": state,
            "updated_at": now_stamp(),
            **fields,
        }
        self.status_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def write_exception_status(self, exc: BaseException, **fields: Any) -> None:
        """把异常类型、消息和 traceback 写入 status.json 后再由调用方抛出。"""

        self.write_status(
            "failed",
            error={
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            **fields,
        )
