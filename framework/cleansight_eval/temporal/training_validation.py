"""时序训练期 validation 摘要。

训练循环只需要普通数值选择 best checkpoint，不构造 EvaluationResult。这里复用 framework
core 唯一的纯时序指标内核，避免在 framework 复制 Edit/F1 算法或引入评测结果类型。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..core.metrics import temporal_metrics


def summarize_training_metrics(
    pred_by_item: Mapping[str, Sequence[str]],
    truth_by_item: Mapping[str, Sequence[str]],
    labels: Sequence[str],
) -> dict[str, float | None]:
    """返回训练历史使用的百分制 acc/edit/F1@0.5，不生成正式评测对象。"""

    raw = temporal_metrics(pred_by_item, truth_by_item, labels)
    accuracy = raw["frame"].get("accuracy")
    edit = raw["segment"].get("edit")
    f1 = raw["segment"]["details_at_iou"]["0.50"].get("f1")

    def percent(value) -> float | None:
        return None if value is None else round(float(value) * 100.0, 2)

    return {
        "val_acc": percent(accuracy),
        "val_edit": percent(edit),
        "val_f1_0.5": percent(f1),
    }
