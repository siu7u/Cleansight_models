"""ROI 分类流水线（roi_classification）冒烟测试。

用小假数据（随机张量）跑通 FeatureFusionModel 构建 → 分类 evaluator 三态翻译；
分类流水线的完整 train/predict 在无 torch 环境不执行（由 classification/pipeline 的
重依赖延迟 import 保证可导入）。
"""

from __future__ import annotations

import pytest

from benchmark.evaluators.classification import evaluate
from benchmark.evaluators.registry import get_evaluator
from benchmark.core.result import MetricState


def test_classification_evaluator_registered():
    assert get_evaluator("roi_classification") is evaluate


def test_classification_evaluator_translates_metrics():
    """用与 pipeline.predict 同形的普通 dict 验证 evaluator 翻译。"""

    output = {
        "model_type": "feature_fusion",
        "model_id": "feature_fusion-resnet50",
        "pipeline": "roi_classification",
        "checkpoint": "/tmp/fake.pt",
        "dataset": "fusion_small_eliminated",
        "feature_schema": {"modality": "roi_image", "roi_size": 224},
        "inference_semantics": {"mode": "single_roi", "stateless": True},
        "num_params": 24_000_000,
        "native_metrics": {
            "per_class": {
                "air_gun": {"precision": 0.8, "recall": 0.7, "f1": 0.75, "support": 120},
                "brush_tip_out": {"precision": 0.6, "recall": 0.5, "f1": 0.55, "support": 40},
            },
            "micro": {"precision": 0.75, "recall": 0.65, "f1": 0.70},
            "exact_match": 0.62,
            "labels": {0: "air_gun", 1: "brush_tip_out"},
        },
        "errors": [],
    }

    result = evaluate(output, {"save_predictions": False})

    assert result.pipeline == "roi_classification"
    assert result.metrics["precision"].state is MetricState.COMPUTED
    assert result.metrics["precision"].value == 0.75
    assert result.metrics["recall"].value == 0.65
    assert result.metrics["f1"].value == 0.70
    assert result.metrics["exact_match"].value == 0.62
    assert result.metric_details["per_class"]["air_gun"]["recall"] == 0.7
    assert result.metric_details["per_class"]["brush_tip_out"]["f1"] == 0.55
    assert result.performance["model_forward_mean_ms"].state is MetricState.NOT_APPLICABLE


def test_classification_evaluator_missing_class_marked_missing():
    output = {
        "pipeline": "roi_classification",
        "model_type": "feature_fusion",
        "model_id": "feature_fusion-resnet50",
        "checkpoint": "/tmp/fake.pt",
        "dataset": "d",
        "feature_schema": {},
        "inference_semantics": {},
        "num_params": 1,
        "native_metrics": {
            "per_class": {},
            "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
            "exact_match": 0.0,
            "labels": {0: "air_gun"},
        },
        "errors": [],
    }
    result = evaluate(output, {"save_predictions": False})
    detail = result.metric_details["per_class"]["air_gun"]
    assert detail["precision"] == {"state": "missing"}


def test_roi_dataset_accepts_both_names_forms(tmp_path):
    """data.yaml 的 names 无论 list 还是 dict{id: name}，ROI 提取结果都必须一致。"""

    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    from framework.cleansight_eval.classification.data import build_roi_dataset

    def make(root, names_block):
        (root / "images" / "train").mkdir(parents=True)
        (root / "labels" / "train").mkdir(parents=True)
        (root / "data.yaml").write_text(
            "train: images/train\nval: images/train\nnc: 2\n" + names_block, encoding="utf-8"
        )
        for index in range(4):
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[10:30, 10:30] = (0, 0, 255)
            cv2.imwrite(str(root / "images" / "train" / f"img{index}.jpg"), image)
            cls = index % 2
            (root / "labels" / "train" / f"img{index}.txt").write_text(
                f"{cls} 0.3 0.3 0.3 0.3\n", encoding="utf-8"
            )

    list_root = tmp_path / "list_form"
    dict_root = tmp_path / "dict_form"
    make(list_root, "names:\n  - syringe\n  - air_gun\n")
    make(dict_root, "names:\n  0: syringe\n  1: air_gun\n")

    xs, ys, names_a, _ = build_roi_dataset(list_root, ["air_gun"], roi_size=32, neg_ratio=0.0)
    xd, yd, names_b, _ = build_roi_dataset(dict_root, ["air_gun"], roi_size=32, neg_ratio=0.0)
    assert names_a == names_b == ["air_gun"]
    assert xs.shape == xd.shape and ys.shape == yd.shape
    assert np.array_equal(xs, xd) and np.array_equal(ys, yd)
