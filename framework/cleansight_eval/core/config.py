"""实验配置加载、覆盖与有效性检查（framework 层）。

配置驱动同架构变体（需求 §4.3）：族、规模、任务、执行模式、数据、特征、
训练与评估参数、指标都由 YAML 表达。本模块只做与模型语义无关的加载与结构
校验，不理解具体模型。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# 框架层只校验与模型语义无关的通用字段。feature_schema、train、model.input_dim/
# num_classes 等是**流水线专属**要求，下沉到各 Pipeline.validate_config，否则检测这类
# 无特征向量的流水线连配置都过不了。
# pipeline：本实验属于哪条流水线（detection / full_sequence_temporal /
# sliding_window_temporal）；训练与评估同属一条，输入构造与输出语义一致。
REQUIRED_TOP_KEYS = ("pipeline", "model", "data")


def load_config(path: str | Path) -> dict:
    """读取 YAML 实验配置，只做**格式中立**的框架层通用校验。

    流水线专属校验（feature_schema、input_dim、data_yaml…）由各流水线的
    ``validate_config`` 负责，在 CLI 分派器里于本函数之后调用。core 因此**不 import
    任何流水线**，脊柱不反依赖 temporal/detection。
    """

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是映射: {path}")
    validate_config(data)
    return data


def validate_config(cfg: dict) -> None:
    """框架层通用结构校验（不含任何流水线专属字段）。"""

    missing = [k for k in REQUIRED_TOP_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"配置缺少必要字段: {missing}")
    if not isinstance(cfg["pipeline"], str) or not cfg["pipeline"]:
        raise ValueError(
            "pipeline 必须是非空字符串，如 sliding_window_temporal / full_sequence_temporal / detection"
        )


def apply_overrides(cfg: dict, overrides: dict[str, Any]) -> dict:
    """把 CLI 传入的覆盖项应用到 train 段（如 epochs/lr/batch_size/window）。"""

    out = {**cfg, "train": {**cfg.get("train", {})}}
    for key, value in overrides.items():
        if value is not None:
            out["train"][key] = value
    return out
