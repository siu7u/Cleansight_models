"""时序训练期 validation 摘要。

训练循环只需要普通数值选择 best checkpoint，不构造 EvaluationResult。这里复用 framework
core 唯一的纯时序指标内核，**并按 ``core/metrics.py`` 的口径注册表**产出验证指标：训练侧键名
（``val_*``）与正式评测 summary 键名（``acc`` / ``edit`` / ``f1@0.5`` …）由注册表一一绑定，
聚合方式、单位与 spec 版本只有一处定义，因此训练选点用的数字与 ``benchmark.cli.eval`` 报出的
同名指标必然同口径。

单位约定：注册表 ``unit=percent`` 的指标统一输出 0..100、四舍五入 2 位；内核原始值恒为 0..1。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..core.metrics import TEMPORAL_METRIC_SPECS, detail_value, temporal_metrics


def _percent(value) -> float | None:
    """0..1 比率 → 0..100 百分数；``None``（样本不足）保持 ``None``，不用 0 冒充。"""

    return None if value is None else round(float(value) * 100.0, 2)


def summarize_training_metrics(
    pred_by_item: Mapping[str, Sequence[str]],
    truth_by_item: Mapping[str, Sequence[str]],
    labels: Sequence[str],
) -> dict[str, float | None]:
    """返回训练历史使用的百分制验证指标，键名与口径全部取自指标注册表。

    返回键 = 注册表中 ``training_key`` 非空的项（当前为
    ``val_acc / val_edit / val_f1_0.1 / val_f1_0.25 / val_f1_0.5``），它们同时也是
    ``train.best_metric`` 的合法取值集合。
    """

    raw = temporal_metrics(pred_by_item, truth_by_item, labels)
    summary: dict[str, float | None] = {}
    for spec in TEMPORAL_METRIC_SPECS.values():
        training_key = spec.get("training_key")
        if not training_key:
            continue
        summary[training_key] = _percent(detail_value(raw, spec["detail_path"]))
    return summary
