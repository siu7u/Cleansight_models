"""实验配置加载、覆盖与有效性检查（framework 层）。

配置驱动同架构变体（需求 §4.3）：族、规模、任务、执行模式、数据、特征、
训练与评估参数、指标都由 YAML 表达。本模块只做与模型语义无关的加载与结构
校验，不理解具体模型。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# 框架层只校验与模型语义无关的通用字段（§4.2）。feature_schema、train、
# model.input_dim/num_classes 等是**任务专属**要求，下沉到各 Task.validate_config，
# 否则检测这类无特征向量的任务连配置都过不了。
# feeding：本实验的喂入模式，**训练与评估共用同一个**（训练怎么喂，评估就怎么喂）。
REQUIRED_TOP_KEYS = ("family", "model", "task", "feeding", "data")


def load_config(path: str | Path) -> dict:
    """读取 YAML 实验配置：先做框架层通用校验，再委托任务层校验专属字段。"""

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是映射: {path}")
    validate_config(data)
    # 任务专属校验（延迟 import 避免与 tasks 层的循环依赖）。
    from ..tasks import get_task

    get_task(data["task"]).validate_config(data)
    return data


def validate_config(cfg: dict) -> None:
    """框架层通用结构校验（不含任何任务专属字段）。"""

    missing = [k for k in REQUIRED_TOP_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"配置缺少必要字段: {missing}")
    if not isinstance(cfg["feeding"], str) or not cfg["feeding"]:
        raise ValueError("feeding 必须是非空字符串（训练与评估共用的喂入模式），如 windowed_causal")


def apply_overrides(cfg: dict, overrides: dict[str, Any]) -> dict:
    """把 CLI 传入的覆盖项应用到 train 段（如 epochs/lr/batch_size/window）。"""

    out = {**cfg, "train": {**cfg.get("train", {})}}
    for key, value in overrides.items():
        if value is not None:
            out["train"][key] = value
    return out
