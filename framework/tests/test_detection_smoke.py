"""检测流水线 evaluate 端到端冒烟（注入假 adapter，免 ultralytics）。

验证检测流水线仍能：读 sidecar 校验 → 取 adapter.val 结果 → 组装规范三态信封 → 被同一
core 矩阵收纳。ultralytics 不可用（重依赖），故 monkeypatch adapter。
"""

import sys
from types import SimpleNamespace

import torch

from benchmark.evaluators import evaluate_prediction

from benchmark.core.integrity import check_result_complete
from cleansight_eval.core.checkpoint import write_meta
from benchmark.core.result import MetricState
from cleansight_eval.core.execution import PredictionOutput
from benchmark.core.matrix import build_matrix
from cleansight_eval.detection import pipeline as det
from cleansight_eval.detection.yolo import YoloAdapter


class _FakeAdapter:
    """假 YOLO 适配器：val 返回与 YoloAdapter.val 同形的普通 dict。"""

    def val(self, **_kwargs):
        return {
            "map50": 0.612345,
            "map50_95": 0.401111,
            "precision": 0.5,
            "recall": 0.45,
            "num_params": 2600000,
            "names": {0: "hand", 1: "scope_control_body", 2: "scope_mid_section"},
            "per_class": {
                "hand": {"precision": 0.8, "recall": 0.7, "map50": 0.75},
                "scope_mid_section": {"precision": 0.0, "recall": 0.0, "map50": 0.0},
            },
        }

    def predict(self, **_kwargs):
        return {
            "split": "val",
            "labels": {"0": "hand", "1": "scope_control_body", "2": "scope_mid_section"},
            "items": {"frame-0001.jpg": {"predictions": []}},
        }


class _Parameter:
    """只暴露 numel 的轻量参数，用于验证 fuse 前后的统计时机。"""

    def __init__(self, count):
        self.count = count

    def numel(self):
        return self.count


class _ParameterContainer:
    def __init__(self, count):
        self.count = count

    def parameters(self):
        return [_Parameter(self.count)]


class _FusingFakeYOLO:
    """模拟 val 后替换为较小 fused 模型的 Ultralytics YOLO。"""

    def __init__(self, _weights):
        self.model = _ParameterContainer(11)
        self.names = {0: "hand"}

    def val(self, **_kwargs):
        self.model = _ParameterContainer(7)
        box = SimpleNamespace(
            map50=0.5,
            map=0.4,
            mp=0.3,
            mr=0.2,
            ap_class_index=[],
        )
        return SimpleNamespace(box=box)

def _write_ckpt_with_meta(tmp_path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"not-a-real-weight")  # 权重内容无关（评估走假 adapter）
    meta = {"type": "yolo", "pipeline": "detection", "num_params": 2600000, "nc": 3}
    write_meta(ckpt, meta)
    return str(ckpt)


def _write_ckpt_without_meta(tmp_path):
    ckpt = tmp_path / "external.pt"
    ckpt.write_bytes(b"not-a-real-weight")
    return str(ckpt)


def _cfg():
    return {
        "pipeline": "detection",
        "model": {"type": "yolo", "weights": "yolo11n.pt", "imgsz": 640},
        "data": {"name": "group1_large", "data_yaml": "dummy/data.yaml", "eval_split": "val"},
        "evaluation": {"mode": "formal", "save_predictions": True},
    }


def _evaluate(cfg, ckpt):
    """测试组合根：pipeline 只预测，benchmark evaluator 负责指标和结果。"""

    output = det.DetectionPipeline().predict(cfg, ckpt, torch.device("cpu"))
    result = evaluate_prediction(output, cfg.get("evaluation"))
    result.integrity = check_result_complete(result)
    return result


def test_yolo_adapter_counts_checkpoint_parameters_before_validation_fusion(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=_FusingFakeYOLO))

    result = YoloAdapter().val(
        weights="fake.pt",
        data_yaml="fake.yaml",
        split="val",
        imgsz=640,
        device=torch.device("cpu"),
        conf=0.001,
        iou=0.7,
        max_det=300,
        agnostic_nms=False,
    )

    assert result["num_params"] == 11


def test_evaluate_produces_wellformed_envelope(tmp_path, monkeypatch):
    monkeypatch.setattr(det, "get_adapter", lambda mt: _FakeAdapter())
    ckpt = _write_ckpt_with_meta(tmp_path)

    env = _evaluate(_cfg(), ckpt)

    # 基本字段 + 完整性
    assert env.model_type == "yolo" and env.pipeline == "detection"
    assert env.integrity["ok"] is True
    assert env.num_params == 2600000
    # 单帧语义常量挂上了
    assert env.inference_semantics["mode"] == "single_frame"
    # 主指标只保留整体值；逐类三态放到 details，避免 summary 横向膨胀。
    assert env.metrics["mAP@0.5"].state is MetricState.COMPUTED
    assert env.metrics["mAP@0.5:0.95"].state is MetricState.COMPUTED
    assert "0.95" not in env.metric_details["per_class"]
    assert "recall:scope_control_body" not in env.metrics
    assert env.metric_details["per_class"]["scope_mid_section"]["recall"] == 0.0
    assert env.metric_details["per_class"]["scope_control_body"]["recall"]["state"] == "missing"
    assert env.performance["model_forward_mean_ms"].state is MetricState.NOT_APPLICABLE
    assert set(env.performance) == {"model_forward_mean_ms"}
    assert env.metric_details["effective_parameters"]["conf"] == 0.001
    assert env.pending_artifacts["predictions"]["task_type"] == "detection"
    assert env.pending_artifacts["predictions"]["prediction_format"] == "class_confidence_xywhn"


def test_predict_returns_native_facts_without_framework_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(det, "get_adapter", lambda mt: _FakeAdapter())
    ckpt = _write_ckpt_with_meta(tmp_path)

    output = det.DetectionPipeline().predict(_cfg(), ckpt, torch.device("cpu"))

    assert isinstance(output, PredictionOutput)
    assert output.model_id == "yolo-2.6M"
    assert output.native_metrics["map50"] == 0.612345
    assert list(output.predictions) == ["frame-0001.jpg"]
    assert output.metadata["effective_parameters"] == {
        "conf": 0.001,
        "iou": 0.7,
        "imgsz": 640,
        "split": "val",
        "max_det": 300,
        "agnostic_nms": False,
    }
    assert "metrics" not in output.to_dict()


def test_evaluate_rejects_type_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(det, "get_adapter", lambda mt: _FakeAdapter())
    ckpt = _write_ckpt_with_meta(tmp_path)
    cfg = _cfg()
    cfg["model"]["type"] = "detr"  # 与 sidecar 的 yolo 不符 → 拒绝盲加载
    import pytest

    from cleansight_eval.core.integrity import CompatibilityError

    with pytest.raises(CompatibilityError):
        det.DetectionPipeline().predict(cfg, ckpt, torch.device("cpu"))


def test_evaluate_allows_missing_meta_when_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(det, "get_adapter", lambda mt: _FakeAdapter())
    ckpt = _write_ckpt_without_meta(tmp_path)
    cfg = _cfg()
    cfg["model"]["allow_missing_meta"] = True
    cfg["evaluation"]["mode"] = "exploratory"

    env = _evaluate(cfg, ckpt)

    assert env.model_type == "yolo"
    assert env.model_id == "yolo-2.6M"
    assert env.num_params == 2600000
    assert env.integrity["ok"] is True


def test_evaluate_rejects_missing_meta_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(det, "get_adapter", lambda mt: _FakeAdapter())
    ckpt = _write_ckpt_without_meta(tmp_path)
    import pytest

    with pytest.raises(FileNotFoundError):
        det.DetectionPipeline().predict(_cfg(), ckpt, torch.device("cpu"))


def test_detection_envelope_folds_into_matrix(tmp_path, monkeypatch):
    monkeypatch.setattr(det, "get_adapter", lambda mt: _FakeAdapter())
    ckpt = _write_ckpt_with_meta(tmp_path)
    env = _evaluate(_cfg(), ckpt)
    matrix = build_matrix([env])
    assert "mAP@0.5" in matrix["metric_columns"]
    assert matrix["rows"][0]["model_type"] == "yolo"
