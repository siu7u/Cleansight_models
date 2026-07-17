import pytest

from types import SimpleNamespace

from cleansight_eval.core.envelope import MetricValue
from cleansight_eval.core.integrity import (
    CompatibilityError,
    assert_evaluation_profile,
    check_result_complete,
)


def test_formal_profile_requires_registered_valid_testset():
    cfg = {"evaluation": {"mode": "formal"}}
    with pytest.raises(CompatibilityError, match="已登记"):
        assert_evaluation_profile(cfg, {"registered": False, "validation_errors": []})
    with pytest.raises(CompatibilityError, match="校验失败"):
        assert_evaluation_profile(cfg, {"registered": True, "validation_errors": ["leak"]})


def test_exploratory_profile_allows_degraded_provenance():
    assert_evaluation_profile(
        {"evaluation": {"mode": "exploratory"}},
        {"registered": False, "validation_errors": ["ad-hoc"]},
    )


def test_integrity_does_not_repeat_testset_error_details():
    result = SimpleNamespace(
        model_type="yolo",
        pipeline="detection",
        checkpoint="best.pt",
        dataset="fixture-v1",
        metrics={"mAP@0.5": MetricValue.computed(0.5, spec="map/v1")},
        run={"id": "run-1", "evaluation_mode": "exploratory"},
        checkpoint_info={"sha256": "a" * 64},
        testset={
            "registered": True,
            "fingerprint_sha256": "b" * 64,
            "validation_errors": ["leak-a", "leak-b"],
        },
        artifacts={"predictions": {"path": "predictions.json", "sha256": "c" * 64}},
        metric_details={},
    )

    report = check_result_complete(result)

    assert report["ok"] is False
    assert report["failed_checks"] == ["testset_validation_passed"]
    assert report["issues"] == ["testset 校验未通过"]
