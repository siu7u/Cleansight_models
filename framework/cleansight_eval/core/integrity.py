"""framework 模型执行兼容性检查。

本模块只检查 checkpoint、模型配置和输入特征是否能够安全执行。评测 profile 与
EvaluationResult 完整性由 ``benchmark.core.integrity`` 独占。
"""

from __future__ import annotations

class CompatibilityError(Exception):
    """checkpoint / 特征 / 配置出现硬性不兼容时抛出。"""


def check_checkpoint_config(meta: dict, expected: dict | None) -> list[str]:
    """比对 checkpoint 元信息与期望配置，返回不兼容项列表。

    只比较会改变输入输出契约或权重形状的字段：type、input_dim、num_classes。
    ``window`` 等不改变权重形状的超参数只提示、不算硬性不兼容。
    """

    if expected is None:
        return []

    problems: list[str] = []
    for key in ("type", "input_dim", "num_classes"):
        if key in expected and key in meta and meta[key] != expected[key]:
            problems.append(f"{key}: checkpoint={meta[key]!r} != expected={expected[key]!r}")
    return problems


def assert_checkpoint_config(meta: dict, expected: dict | None) -> None:
    """硬性校验 checkpoint 配置，不兼容立即抛 ``CompatibilityError``。"""

    problems = check_checkpoint_config(meta, expected)
    if problems:
        raise CompatibilityError(
            "checkpoint 与期望配置不兼容，拒绝静默加载:\n  - " + "\n  - ".join(problems)
        )


def check_feature_schema(actual_dim: int, expected: dict | None) -> list[str]:
    """检查实际特征维度与期望 feature schema 是否一致。"""

    if not expected:
        return []
    problems: list[str] = []
    exp_dim = expected.get("dim")
    if exp_dim is not None and exp_dim != actual_dim:
        problems.append(f"feature dim: actual={actual_dim} != expected={exp_dim}")
    return problems
