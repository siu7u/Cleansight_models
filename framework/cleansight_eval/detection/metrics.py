"""检测任务指标组装（任务层，与 ultralytics 解耦）。

输入是一份已从 ultralytics 结果中抽出的**普通 dict**（见 ``YoloAdapter.val``），
本模块只负责把它翻译成三态 ``MetricValue``。这样单元测试无需安装 ultralytics
即可覆盖最关键的对齐逻辑（§8.2 / §10）：

- 已计算的指标 → ``COMPUTED``，并声明口径 ``spec``；
- 验证集**无样本**的类别 → ``MISSING``（不是 0，也不是 N/A）；
- 全程不含任何业务门槛或 PASS/FAIL 判断（§4.5 / §13.11）。
"""

from __future__ import annotations

from ..core.envelope import MetricValue

SPEC_MAP50 = "map/coco-0.5/v1"
SPEC_MAP50_95 = "map/coco-0.5:0.95/v1"
SPEC_PRECISION = "precision/detection-iou0.5/v1"
SPEC_RECALL = "recall/detection-iou0.5/v1"

_NO_SAMPLE = "验证集无该类样本，无法评估"


def build_detection_metrics(val: dict) -> dict[str, MetricValue]:
    """把 ``YoloAdapter.val`` 的输出翻译为三态指标字典。

    ``val`` 约定字段：
      - ``map50`` / ``map50_95`` / ``precision`` / ``recall``：整体标量；
      - ``names``：``{class_id: name}``，data.yaml 里声明的全部类别；
      - ``per_class``：``{name: {precision, recall, map50}}``，仅含验证集里
        有样本、被评估到的类别。
    """

    metrics: dict[str, MetricValue] = {
        "mAP@0.5": MetricValue.computed(round(float(val["map50"]), 4), spec=SPEC_MAP50),
        "mAP@0.5:0.95": MetricValue.computed(round(float(val["map50_95"]), 4), spec=SPEC_MAP50_95),
        "precision": MetricValue.computed(round(float(val["precision"]), 4), spec=SPEC_PRECISION),
        "recall": MetricValue.computed(round(float(val["recall"]), 4), spec=SPEC_RECALL),
    }

    per_class = val.get("per_class", {})
    for cid, name in sorted(val.get("names", {}).items(), key=lambda kv: int(kv[0])):
        if name in per_class:
            pc = per_class[name]
            metrics[f"precision:{name}"] = MetricValue.computed(round(float(pc["precision"]), 4), spec=SPEC_PRECISION)
            metrics[f"recall:{name}"] = MetricValue.computed(round(float(pc["recall"]), 4), spec=SPEC_RECALL)
        else:
            metrics[f"precision:{name}"] = MetricValue.missing(reason=_NO_SAMPLE, spec=SPEC_PRECISION)
            metrics[f"recall:{name}"] = MetricValue.missing(reason=_NO_SAMPLE, spec=SPEC_RECALL)

    return metrics
