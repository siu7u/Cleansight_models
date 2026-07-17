"""benchmark EvaluationResult v2 唯一真源与历史转换测试。"""

from benchmark.core.result import (
    EvaluationResult,
    MetricState,
    MetricValue,
    build_result,
    upgrade_legacy_result,
    validate_result,
)
from framework.cleansight_eval.core.envelope import EvalEnvelope


def test_framework_envelope_is_only_compatibility_alias():
    assert EvalEnvelope is EvaluationResult


def test_model_evaluation_v2_roundtrip_and_optional_decision(tmp_path):
    result = EvaluationResult(
        model_type="gru",
        model_id="gru-51k",
        pipeline="sliding_window_temporal",
        checkpoint="runs/gru/checkpoints/best.pt",
        dataset="actionmixed-v1",
        feature_schema={"dim": 40, "version": "actionmixed-bbox-8cls-v1"},
        metrics={"f1@0.5": MetricValue.computed(75.0, spec="f1/v2")},
        inference_semantics={"mode": "windowed_causal", "window": 64},
    )

    payload = result.to_dict()
    validate_result(payload)
    assert payload["schema_version"] == 2
    assert "decision" not in payload
    assert payload["metrics"]["summary"]["f1@0.5"] == {
        "state": "computed",
        "value": 75.0,
        "spec": "f1/v2",
    }

    path = result.write(tmp_path / "result.evaluation.json")
    restored = EvaluationResult.read(path)
    assert restored.metrics["f1@0.5"].state is MetricState.COMPUTED
    assert restored.feature_schema["dim"] == 40


def test_metric_value_omits_meaningless_null_fields():
    missing = MetricValue.missing("没有样本", spec="recall/v1").to_dict()
    assert missing == {"state": "missing", "spec": "recall/v1", "reason": "没有样本"}
    assert "value" not in missing


def test_model_omits_unknown_parameter_count():
    result = EvaluationResult(
        model_type="yolo",
        model_id="yolo-?",
        pipeline="detection",
        checkpoint="external.pt",
        dataset="fixture-v1",
    )
    assert "num_params" not in result.to_dict()["model"]


def test_benchmark_builder_uses_v2_decision_not_legacy_gates():
    payload = build_result(
        benchmark="single_model_temporal",
        task_type="temporal",
        run_id="run-001",
        model={"type": "gru", "id": "gru-v1"},
        testset={
            "id": "temporal.v1.test",
            "dataset_version": "temporal-v1",
            "split": "test",
            "manifest_sha256": "a" * 64,
        },
        inference={"mode": "windowed_causal"},
        metrics={"accuracy": 0.8},
        status="PASS",
    )

    assert payload["schema_version"] == 2
    assert payload["decision"]["status"] == "PASS"
    assert "gates" not in payload
    assert payload["metrics"]["summary"]["accuracy"]["state"] == "computed"


def test_upgrade_legacy_benchmark_result_preserves_run_time():
    legacy = {
        "schema_version": 1,
        "benchmark": "e2e_3min",
        "task_type": "e2e",
        "run": {"id": "old-run", "created_at": "2026-07-16T00:00:00Z"},
        "model": None,
        "testset": {
            "id": "e2e.clean.test",
            "dataset_version": "clean-v1",
            "split": "test",
            "manifest_sha256": "b" * 64,
        },
        "inference": {"mode": "end_to_end"},
        "metrics": {"success_rate": 1.0},
        "limits": {"is_smoke": False},
        "gates": {"status": "PASS", "reasons": []},
        "artifacts": {},
    }

    upgraded = upgrade_legacy_result(legacy)
    assert upgraded["schema_version"] == 2
    assert upgraded["run"]["created_at"] == "2026-07-16T00:00:00Z"
    assert upgraded["decision"]["status"] == "PASS"
