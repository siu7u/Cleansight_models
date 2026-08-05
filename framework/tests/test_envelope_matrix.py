"""信封三态与矩阵机读/人读（需求 §9 / §10）。"""

from benchmark.core.result import EvaluationResult, MetricState, MetricValue
from benchmark.core.matrix import build_matrix, render_markdown


def _envelope(pipeline, latency_state):
    perf = {
        "latency_mean_ms": MetricValue.computed(3.2, spec="latency/v1")
        if latency_state == "computed"
        else MetricValue.not_applicable("全序列不测延迟", spec="latency/v1")
    }
    return EvaluationResult(
        model_type="gru",
        model_id="gru-128h",
        pipeline=pipeline,
        checkpoint="x.pt",
        dataset="endo",
        metrics={
            "acc": MetricValue.computed(88.0, spec="acc/v1"),
            "edit": MetricValue.missing("对齐失败", spec="edit/v1"),
        },
        performance=perf,
        num_params=256131,
    )


def test_three_states_distinct():
    na = MetricValue.not_applicable()
    miss = MetricValue.missing()
    comp = MetricValue.computed(1.0)
    assert (na.display(), miss.display(), comp.display()) == ("N/A", "MISSING", "1.0")
    assert na.state is MetricState.NOT_APPLICABLE
    assert miss.state is MetricState.MISSING


def test_matrix_heterogeneous_columns_and_states():
    envs = [_envelope("full_sequence_temporal", "na"), _envelope("sliding_window_temporal", "computed")]
    matrix = build_matrix(envs)
    # 全序列行延迟为 N/A，滑窗行为已计算值 —— 二者不混淆
    rows = {r["pipeline"]: r for r in matrix["rows"]}
    off = rows["full_sequence_temporal"]["cells"]["perf.latency_mean_ms"]
    rt = rows["sliding_window_temporal"]["cells"]["perf.latency_mean_ms"]
    assert off["state"] == MetricState.NOT_APPLICABLE.value
    assert rt["state"] == MetricState.COMPUTED.value
    # 缺失指标不伪装成 N/A
    assert rows["full_sequence_temporal"]["cells"]["edit"]["state"] == MetricState.MISSING.value


def test_markdown_renders_states():
    md = render_markdown(build_matrix([_envelope("full_sequence_temporal", "na")]))
    assert "N/A" in md
    assert "MISSING" in md
    assert "gru" in md


def test_envelope_roundtrip(tmp_path):
    env = _envelope("sliding_window_temporal", "computed")
    p = env.write(tmp_path / "e.envelope.json")
    back = EvaluationResult.read(p)
    assert back.metrics["acc"].value == 88.0
    assert back.performance["latency_mean_ms"].state is MetricState.COMPUTED
    assert back.to_dict()["schema_version"] == 2


def test_read_legacy_v1_envelope():
    legacy = {
        "model_type": "gru",
        "model_id": "gru-old",
        "pipeline": "sliding_window_temporal",
        "checkpoint": "old.pt",
        "dataset": "old-data",
        "metrics": {"acc": MetricValue.computed(50.0, spec="acc/v1").to_dict()},
        "performance": {},
    }
    env = EvaluationResult.from_dict(legacy)
    assert env.model_id == "gru-old"
    assert env.metrics["acc"].value == 50.0
    assert env.to_dict()["schema_version"] == 2
