"""指标口径一致性：训练选点 / 正式评测 / history / 矩阵工具 必须读同一套定义。

背景：同一批指标出现在四个面上（训练期 validation、benchmark 正式评测、history.csv 与曲线、
矩阵汇总工具）。过去四处各自拼名字与单位，容易出现"训练侧 val_f1_0.5、评测侧 f1@0.5、一个
百分数一个 0..1 比率、选点词表只覆盖三个指标"这类隐性分歧。

现在唯一真源是 ``framework/cleansight_eval/core/metrics.py`` 的两个注册表（``TEMPORAL_METRIC_SPECS`` / ``CLASSIFICATION_METRIC_SPECS``），
本测试把它钉死：任何一面新增/改名/换单位而没同步注册表，这里就会红。
"""

from __future__ import annotations

import json

from benchmark.evaluators.temporal import compute_temporal_metrics_by_item
from framework.cleansight_eval.core.history import temporal_history_columns
from framework.cleansight_eval.core.metrics import (
    TEMPORAL_METRIC_SPECS,
    detail_value,
    metric_spec,
    temporal_metrics,
    training_metric_keys,
    training_metric_name,
)
from framework.cleansight_eval.temporal.training_validation import summarize_training_metrics
from framework.cleansight_eval.temporal.util import VALID_BEST_METRICS

LABELS = ["idle", "brush", "flush"]
TRUTH = {
    "video-a": ["idle"] * 6 + ["brush"] * 6 + ["flush"] * 4 + ["idle"] * 8,
    "video-b": ["brush"] * 5 + ["idle"] * 10 + ["flush"] * 3,
}
PRED = {
    # video-a：段边界偏 1 帧 + 多出一段短 brush（制造 FP）
    "video-a": ["idle"] * 7 + ["brush"] * 5 + ["flush"] * 3 + ["brush"] * 2 + ["idle"] * 7,
    # video-b：整段正确，只差首段时间
    "video-b": ["brush"] * 4 + ["idle"] * 11 + ["flush"] * 3,
}


def _raw():
    return temporal_metrics(PRED, TRUTH, LABELS)


def test_registry_specs_match_evaluator_output():
    """注册表声明的 spec 必须与评测器真正落盘的口径字符串逐一相同。"""

    metrics = compute_temporal_metrics_by_item(PRED, TRUTH, LABELS)
    assert set(metrics) == set(TEMPORAL_METRIC_SPECS), "评测器与注册表的指标名集合不一致"
    for name, metric in metrics.items():
        assert metric.spec == metric_spec(name), f"{name} 的 spec 与注册表不一致"


def test_summary_is_percent_and_details_are_ratio():
    """单位约定：对外报出 percent(0..100)，details 里的原始值恒为 0..1 比率。"""

    metrics = compute_temporal_metrics_by_item(PRED, TRUTH, LABELS)
    raw = _raw()
    for name, entry in TEMPORAL_METRIC_SPECS.items():
        detail = detail_value(raw, entry["detail_path"])
        assert detail is not None, f"{name} 的 detail_path 取不到值: {entry['detail_path']}"
        if entry["unit"] == "percent":
            assert 0.0 <= float(detail) <= 1.0, f"{name} 的 details 值应为 0..1 比率"
            assert metrics[name].value == round(float(detail) * 100.0, 2), f"{name} 换算口径不一致"
        else:
            assert metrics[name].value == int(detail), f"{name} 计数口径不一致"


def test_training_validation_same_numbers_as_benchmark():
    """同一个输入：训练期 validation 与正式评测的同名指标必须给出同一个数。"""

    training = summarize_training_metrics(PRED, TRUTH, LABELS)
    metrics = compute_temporal_metrics_by_item(PRED, TRUTH, LABELS)
    assert tuple(training) == training_metric_keys()
    for training_key, value in training.items():
        name = training_metric_name(training_key)
        assert value == metrics[name].value, f"{training_key} 与 {name} 数值不一致"


def test_best_metric_vocabulary_is_derived_from_registry():
    """选点词表 = 注册表里声明了 training_key 的指标；不存在第二份枚举。"""

    assert set(VALID_BEST_METRICS) == set(training_metric_keys())
    for training_key in VALID_BEST_METRICS:
        # 每个可选指标都必须能被「列名 → 评测指标名」解析，且已在注册表登记。
        assert training_metric_name(training_key) in TEMPORAL_METRIC_SPECS


def test_history_columns_cover_every_selectable_metric():
    """两条时序流水线共用的 history 列必须覆盖全部可选指标（含新增的 F1@0.1/0.25）。"""

    columns = temporal_history_columns()
    for training_key in training_metric_keys():
        assert training_key in columns, f"history 缺少可选指标列: {training_key}"
    assert columns[:3] == ["epoch", "train_loss", "val_loss"]
    assert {"lr", "epoch_sec", "checkpoint_best", "checkpoint_last", "status"} <= set(columns)


def test_pipelines_use_shared_history_columns():
    """两条流水线都从 core.history 取列定义，不允许各自内联一份列表。"""

    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "framework" / "cleansight_eval" / "temporal"
    for name in ("full_sequence_pipeline.py", "sliding_window_pipeline.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "temporal_history_columns()" in text, f"{name} 未使用共享 history 列定义"
        assert '"val_f1_0.5"' not in text, f"{name} 仍内联指标列名"


def test_matrix_tools_read_registered_paths():
    """矩阵工具从 evaluation.json 读的路径必须都在注册表登记（单位换算由注册表声明）。"""

    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    tools = [root / "tools" / "run_capacity_matrix.py", root / "tools" / "run_strategy_matrix.py"]
    registered = set(TEMPORAL_METRIC_SPECS)
    for tool in tools:
        text = tool.read_text(encoding="utf-8")
        assert '["details"]["temporal"]' in text, f"{tool.name} 未按 details 口径读取指标"
        # 工具直接用 summary 键名时必须用注册名（如 summary["acc"] / summary["edit"]）。
        for used in re.findall(r'summary\["([^"]+)"\]', text):
            assert used in registered, f"{tool.name} 读了未注册的 summary 键: {used}"
    # 工具读的段级 details 阈值键必须与注册表一致（0.10/0.25/0.50，不是 0.1/0.5）。
    detail_keys = {path[-1] for name, entry in TEMPORAL_METRIC_SPECS.items()
                   if (path := entry["detail_path"])[1] in {"f1_at_iou", "details_at_iou"}}
    assert {"0.10", "0.25", "0.50"} <= detail_keys
    json.dumps(_raw())  # 结果必须可序列化（工具直接读落盘的 evaluation.json）


# ---------------------------------------------------------------------------
# 分类（roi_classification）：同一套要求，注册表同样在 core/metrics.py 内核里
# ---------------------------------------------------------------------------


def test_classification_specs_match_registry_and_evaluator():
    """分类注册表的 spec 必须与评测器落盘的口径字符串逐一相同。"""

    from benchmark.evaluators import classification as evaluator
    from framework.cleansight_eval.core.metrics import (
        CLASSIFICATION_METRIC_SPECS,
        classification_metric_spec,
    )

    for name in CLASSIFICATION_METRIC_SPECS:
        assert classification_metric_spec(name) == CLASSIFICATION_METRIC_SPECS[name]["spec"]
    assert evaluator.SPEC_PRECISION == classification_metric_spec("precision")
    assert evaluator.SPEC_RECALL == classification_metric_spec("recall")
    assert evaluator.SPEC_F1 == classification_metric_spec("f1")
    assert evaluator.SPEC_EXACT_MATCH == classification_metric_spec("exact_match")
    # 未注册名字必须报错，避免下游自造口径。
    for bad in ("accuracy", "map@0.5"):
        try:
            classification_metric_spec(bad)
        except KeyError:
            continue
        raise AssertionError(f"未注册的分类指标 {bad} 竟然被接受")


def test_classification_numbers_unchanged_vs_legacy_formula():
    """共享实现必须与历史内联公式同值（回归锚点：统一口径不改数字）。"""

    import numpy as np

    from framework.cleansight_eval.core.metrics import (
        DECISION_THRESHOLD,
        decide,
        multilabel_metrics,
    )

    rng = np.random.default_rng(11)
    scores = rng.random((200, 3)).astype(np.float32)
    truth = (rng.random((200, 3)) > 0.6).astype(np.float32)
    binary = decide(scores)
    assert np.array_equal(binary, (scores > DECISION_THRESHOLD).astype(np.float32))

    got = multilabel_metrics(binary, truth, ["a", "b", "c"])

    # ↓ 逐字复制自统一前的 classification/pipeline.py::_evaluate
    legacy_per_class = {}
    for index, name in enumerate(["a", "b", "c"]):
        tp = ((binary[:, index] == 1) & (truth[:, index] == 1)).sum()
        fp = ((binary[:, index] == 1) & (truth[:, index] == 0)).sum()
        fn = ((binary[:, index] == 0) & (truth[:, index] == 1)).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        legacy_per_class[name] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(2 * precision * recall / max(precision + recall, 1e-8)), 4),
            "support": int(truth[:, index].sum()),
        }
    tp_all = ((binary == 1) & (truth == 1)).sum()
    fp_all = ((binary == 1) & (truth == 0)).sum()
    fn_all = ((binary == 0) & (truth == 1)).sum()
    micro_p = tp_all / max(tp_all + fp_all, 1)
    micro_r = tp_all / max(tp_all + fn_all, 1)

    assert got["per_class"] == legacy_per_class
    assert got["micro"] == {
        "precision": round(float(micro_p), 4),
        "recall": round(float(micro_r), 4),
        "f1": round(float(2 * micro_p * micro_r / max(micro_p + micro_r, 1e-8)), 4),
    }
    assert got["exact_match"] == round(float(((binary == truth).all(axis=1)).mean()), 4)


def test_classification_batch_merge_equals_oneshot():
    """训练期逐 batch 累加计数必须与评测期一次性计算完全等价。"""

    import numpy as np

    from framework.cleansight_eval.core.metrics import (
        confusion_counts,
        decide,
        merge_counts,
        metrics_from_counts,
    )

    rng = np.random.default_rng(7)
    scores = rng.normal(size=(30, 3)).astype(np.float32) * 2.0
    truth = (rng.random((30, 3)) > 0.55).astype(np.float32)
    classes = ["a", "b", "c"]
    per_batch = [
        confusion_counts(decide(scores[start:start + 8]), truth[start:start + 8])
        for start in range(0, 30, 8)
    ]
    assert metrics_from_counts(merge_counts(per_batch), classes) == metrics_from_counts(
        confusion_counts(decide(scores), truth), classes
    )
    # 一个 batch 都没有 → 明确报错，不用 0 指标掩盖接线错误。
    try:
        metrics_from_counts(merge_counts([]), classes)
    except ValueError:
        return
    raise AssertionError("未累积任何 batch 时应当报错")


def test_classification_threshold_and_history_keys_are_single_source():
    """判正阈值与训练侧指标键都只在注册表声明，流水线不得再内联。"""

    from pathlib import Path

    from framework.cleansight_eval.core.metrics import (
        DECISION_THRESHOLD,
        classification_training_keys,
    )

    root = Path(__file__).resolve().parents[1] / "framework" / "cleansight_eval" / "classification"
    text = (root / "pipeline.py").read_text(encoding="utf-8")
    assert "decide(" in text, "训练/评测路径未走唯一阈值入口 decide()"
    assert "> 0.5" not in text, "仍内联判正阈值 0.5（应使用 DECISION_THRESHOLD / decide()）"
    assert "classification_training_keys()" in text, "history 指标键未从注册表派生"
    # 反向断言：注册表里的键名不得在流水线里再抄一份（键 → 取值位置全部由 value_path 给出）；
    # 唯一例外是文档化的旧别名行 val_acc = epoch_values["val_exact_match"]。
    allowed_literals = {'val_acc = epoch_values["val_exact_match"]'}
    for key in classification_training_keys():
        offenders = [
            line.strip() for line in text.splitlines()
            if f'"{key}"' in line and line.strip() not in allowed_literals
        ]
        assert not offenders, f"流水线里仍内联指标键 {key}: {offenders}"
    assert DECISION_THRESHOLD == 0.5  # 口径值本身：改动需同步文档与历史结论


def test_both_registries_share_structure():
    """两个任务的注册表结构一致，第三个任务照此扩展即可全链路生效。"""

    from framework.cleansight_eval.core.metrics import (
        CLASSIFICATION_METRIC_SPECS,
        classification_training_keys,
    )

    for registry, keys in (
        (TEMPORAL_METRIC_SPECS, training_metric_keys()),
        (CLASSIFICATION_METRIC_SPECS, classification_training_keys()),
    ):
        for name, entry in registry.items():
            assert entry["spec"].endswith("; source=" + entry["spec"].split("source=")[-1])
            assert entry["spec"].split("source=")[-1].startswith("framework.cleansight_eval")
            assert entry["unit"] in {"percent", "ratio", "count"}
        assert len(set(keys)) == len(keys)
        assert set(keys) <= {entry["training_key"] for entry in registry.values() if entry.get("training_key")}


def test_classification_best_metric_vocabulary_is_registry_derived():
    """分类选点词表 = val_loss + 注册表训练键；方向与非法值处理都只有一处。"""

    from framework.cleansight_eval.classification.pipeline import (
        CLASSIFICATION_BEST_METRICS,
        ClassificationPipeline,
        best_metric_mode,
        metric_improved,
    )
    from framework.cleansight_eval.core.metrics import classification_training_keys

    assert CLASSIFICATION_BEST_METRICS == ("val_loss", *classification_training_keys())
    assert best_metric_mode("val_loss") == "min"
    for key in classification_training_keys():
        assert best_metric_mode(key) == "max"
    # 方向语义：min 越小越好、max 越大越好；None（样本不足）永不视为改善。
    assert metric_improved(0.5, 0.6, "min") and not metric_improved(0.7, 0.6, "min")
    assert metric_improved(0.7, 0.6, "max") and not metric_improved(0.5, 0.6, "max")
    assert not metric_improved(None, 0.6, "max")
    assert metric_improved(0.1, None, "max")

    cfg = {
        "pipeline": "roi_classification",
        "model": {"type": "feature_fusion", "backbone": "resnet18"},
        "data": {"classes": ["a", "b"], "group_dir": "tmp/nonexistent"},
        "train": {},
    }
    pipeline = ClassificationPipeline()
    pipeline.validate_config({**cfg, "train": {"best_metric": "val_f1"}})  # 合法
    pipeline.validate_config(cfg)  # 缺省允许
    try:
        pipeline.validate_config({**cfg, "train": {"best_metric": "val_map"}})
    except ValueError as exc:
        assert "best_metric" in str(exc)
        return
    raise AssertionError("未注册的 best_metric 必须被拒绝")


def test_classification_training_values_cover_registry():
    """每个训练侧键都必须能由注册表的 value_path 从指标结果里取到值。"""

    import numpy as np

    from framework.cleansight_eval.core.metrics import (
        classification_training_keys,
        classification_training_values,
        decide,
        multilabel_metrics,
    )

    rng = np.random.default_rng(3)
    metrics = multilabel_metrics(
        decide(rng.random((40, 2))), (rng.random((40, 2)) > 0.5).astype(np.float32), ["a", "b"]
    )
    values = classification_training_values(metrics)
    assert set(values) == set(classification_training_keys())
    for key, value in values.items():
        assert value is not None, f"{key} 的 value_path 取不到值"
        assert 0.0 <= float(value) <= 1.0


def test_classification_loop_records_and_selects_by_registry():
    """训练循环必须按键记录注册表指标，并按 epoch_values[best_metric] 选点。"""

    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "framework" / "cleansight_eval"
            / "classification" / "pipeline.py").read_text(encoding="utf-8")
    assert "classification_training_values(val_metrics)" in text, "未用注册表 value_path 取训练侧指标"
    assert "for key in classification_training_keys()" in text, "history 未按注册表键写入"
    assert "metric_improved(epoch_values[best_metric]" in text, "未按 best_metric 选点"
    assert "best_metric_mode(best_metric)" in text, "未使用注册表派生的方向"
