import argparse
import json

from benchmark.release_gate import build_result


def test_legacy_release_tool_only_builds_manual_review_facts(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps([{"status": "PASS"}, {"status": "FAIL"}]), encoding="utf-8")
    args = argparse.Namespace(
        version="review-v1",
        summary=[str(summary)],
        card=None,
        latency_ms=None,
        causality=None,
        num_params=None,
    )

    result = build_result(args)

    assert result["review_state"] == "MANUAL_REVIEW_REQUIRED"
    assert "status" not in result
    assert result["summaries"][0]["observed_status"] == "MIXED"
