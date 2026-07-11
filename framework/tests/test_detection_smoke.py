"""检测纵 orchestration.evaluate 端到端冒烟（注入假 adapter，免 ultralytics）。

验证拆分后检测纵仍能：读 sidecar 校验 → 取 adapter.val 结果 → 组装规范三态信封 →
被同一 core 矩阵收纳。ultralytics 不可用（重依赖），故 monkeypatch adapter。
"""

import json

import torch

from cleansight_eval.core.checkpoint import meta_path_for
from cleansight_eval.core.envelope import MetricState
from cleansight_eval.core.matrix import build_matrix
from cleansight_eval.detection import orchestration as det


class _FakeAdapter:
    """假 YOLO 适配器：val 返回与 YoloAdapter.val 同形的普通 dict。"""

    def val(self, **_kwargs):
        return {
            "map50": 0.612345,
            "map50_95": 0.401111,
            "precision": 0.5,
            "recall": 0.45,
            "names": {0: "hand", 1: "scope_control_body", 2: "scope_mid_section"},
            "per_class": {
                "hand": {"precision": 0.8, "recall": 0.7, "map50": 0.75},
                "scope_mid_section": {"precision": 0.0, "recall": 0.0, "map50": 0.0},
            },
        }


def _write_ckpt_with_meta(tmp_path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"not-a-real-weight")  # 权重内容无关（评估走假 adapter）
    meta = {"family": "yolo", "task": "detection", "num_params": 2600000, "nc": 3}
    meta_path_for(ckpt).write_text(json.dumps(meta), encoding="utf-8")
    return str(ckpt)


def _cfg():
    return {
        "family": "yolo",
        "task": "detection",
        "feeding": "single_frame",
        "model": {"weights": "yolo11n.pt", "imgsz": 640},
        "data": {"name": "group1_large", "data_yaml": "dummy/data.yaml", "eval_split": "val"},
    }


def test_evaluate_produces_wellformed_envelope(tmp_path, monkeypatch):
    monkeypatch.setattr(det, "get_adapter", lambda fid: _FakeAdapter())
    ckpt = _write_ckpt_with_meta(tmp_path)

    env = det.DetectionOrchestrator().evaluate(_cfg(), ckpt, "single_frame", torch.device("cpu"))

    # 基本字段 + 完整性
    assert env.family == "yolo" and env.task == "detection" and env.feeding == "single_frame"
    assert env.integrity["ok"] is True
    assert env.num_params == 2600000
    # 单帧语义常量挂上了
    assert env.feeding_semantics["mode"] == "single_frame"
    # 指标三态：mAP COMPUTED、无样本类 MISSING、延迟 N/A
    assert env.metrics["mAP@0.5"].state is MetricState.COMPUTED
    assert env.metrics["recall:scope_control_body"].state is MetricState.MISSING
    assert env.performance["latency_mean_ms"].state is MetricState.NOT_APPLICABLE


def test_evaluate_rejects_family_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(det, "get_adapter", lambda fid: _FakeAdapter())
    ckpt = _write_ckpt_with_meta(tmp_path)
    cfg = _cfg()
    cfg["family"] = "detr"  # 与 sidecar 的 yolo 不符 → 拒绝盲加载
    import pytest

    from cleansight_eval.core.integrity import CompatibilityError

    with pytest.raises(CompatibilityError):
        det.DetectionOrchestrator().evaluate(cfg, ckpt, "single_frame", torch.device("cpu"))


def test_detection_envelope_folds_into_matrix(tmp_path, monkeypatch):
    monkeypatch.setattr(det, "get_adapter", lambda fid: _FakeAdapter())
    ckpt = _write_ckpt_with_meta(tmp_path)
    env = det.DetectionOrchestrator().evaluate(_cfg(), ckpt, "single_frame", torch.device("cpu"))
    matrix = build_matrix([env])
    assert "mAP@0.5" in matrix["metric_columns"]
    assert matrix["rows"][0]["family"] == "yolo"
