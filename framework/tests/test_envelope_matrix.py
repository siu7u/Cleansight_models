"""信封三态与矩阵机读/人读（需求 §9 / §10）。"""

from cleansight_eval.core.envelope import EvalEnvelope, MetricState, MetricValue
from cleansight_eval.core.matrix import build_matrix, render_markdown


def _envelope(feeding, latency_state):
    perf = {
        "latency_mean_ms": MetricValue.computed(3.2, spec="latency/v1")
        if latency_state == "computed"
        else MetricValue.not_applicable("全序列不测延迟", spec="latency/v1")
    }
    return EvalEnvelope(
        family="gru",
        model_id="gru-128h",
        task="temporal",
        feeding=feeding,
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
    envs = [_envelope("full_sequence", "na"), _envelope("windowed_causal", "computed")]
    matrix = build_matrix(envs)
    # 全序列行延迟为 N/A，有界因果窗行为已计算值 —— 二者不混淆
    rows = {r["feeding"]: r for r in matrix["rows"]}
    off = rows["full_sequence"]["cells"]["perf.latency_mean_ms"]
    rt = rows["windowed_causal"]["cells"]["perf.latency_mean_ms"]
    assert off["state"] == MetricState.NOT_APPLICABLE.value
    assert rt["state"] == MetricState.COMPUTED.value
    # 缺失指标不伪装成 N/A
    assert rows["full_sequence"]["cells"]["edit"]["state"] == MetricState.MISSING.value


def test_markdown_renders_states():
    md = render_markdown(build_matrix([_envelope("full_sequence", "na")]))
    assert "N/A" in md
    assert "MISSING" in md
    assert "gru" in md


def test_envelope_roundtrip(tmp_path):
    env = _envelope("windowed_causal", "computed")
    p = env.write(tmp_path / "e.envelope.json")
    back = EvalEnvelope.read(p)
    assert back.metrics["acc"].value == 88.0
    assert back.performance["latency_mean_ms"].state is MetricState.COMPUTED
