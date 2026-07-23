"""benchmark 评测配置与 EvaluationResult 完整性检查。

本模块只判断评测事实是否可解释，不判断模型是否达到发布门槛。模型 checkpoint、输入维度等
运行前兼容性检查仍属于 framework。
"""

from __future__ import annotations

from typing import Any

from benchmark.core.result import MetricState


class EvaluationProfileError(Exception):
    """正式/探索评测配置与 testset 契约不兼容。"""


def assert_evaluation_profile(cfg: dict, testset: dict) -> None:
    """正式评测必须使用已登记且校验通过的 testset；探索模式允许降级留痕。"""

    mode = (cfg.get("evaluation") or {}).get("mode", "formal")
    if mode not in {"formal", "exploratory"}:
        raise ValueError(f"evaluation.mode 必须是 formal 或 exploratory，当前为 {mode!r}")
    if mode == "exploratory":
        return
    if not testset.get("registered"):
        raise EvaluationProfileError("formal 评估必须声明 benchmark 已登记 testset")
    errors = testset.get("validation_errors") or []
    if errors:
        raise EvaluationProfileError(
            "formal testset 校验失败:\n  - " + "\n  - ".join(map(str, errors))
        )


def check_result_complete(result: Any) -> dict:
    """检查评测结果是否具备最小可解释字段，返回完整性报告（不做达标判断）。"""

    report: dict[str, Any] = {"ok": True, "issues": []}
    required = ("model_type", "pipeline", "checkpoint", "dataset")
    for key in required:
        if not getattr(result, key, None):
            report["ok"] = False
            report["issues"].append(f"缺少必要字段: {key}")

    if not result.metrics:
        report["ok"] = False
        report["issues"].append("没有任何指标")

    for name, metric in result.metrics.items():
        if metric.state is MetricState.COMPUTED and not metric.spec:
            report["ok"] = False
            report["issues"].append(f"指标 {name} 已计算但未声明口径(spec)")

    if result.run:
        checks: dict[str, bool] = {}
        checks["run_context_present"] = bool(result.run.get("id"))
        checks["checkpoint_hash_present"] = bool(result.checkpoint_info.get("sha256"))
        if result.run.get("evaluation_mode") == "formal":
            meta = result.checkpoint_info.get("meta") or {}
            checks["checkpoint_metadata_bound"] = bool(
                meta.get("schema_version") and meta.get("checkpoint_bound")
            )
        checks["testset_registered"] = bool(result.testset.get("registered"))
        checks["testset_fingerprint_present"] = bool(result.testset.get("fingerprint_sha256"))
        validation_errors = result.testset.get("validation_errors") or []
        checks["testset_validation_passed"] = not validation_errors
        prediction_ref = result.artifacts.get("predictions") or {}
        checks["prediction_artifact_present"] = bool(prediction_ref.get("path"))
        checks["prediction_artifact_hashed"] = bool(prediction_ref.get("sha256"))
        if result.pipeline in {"sliding_window_temporal", "full_sequence_temporal"}:
            checks["metric_details_present"] = bool(result.metric_details.get("temporal"))
            checks["prediction_artifact_recomputable"] = prediction_ref.get("recomputable") is True

        messages = {
            "run_context_present": "缺少 run id",
            "checkpoint_hash_present": "缺少 checkpoint SHA-256",
            "checkpoint_metadata_bound": "formal 评估的 checkpoint metadata 未绑定权重内容",
            "testset_registered": "评估未使用 benchmark 已登记 testset",
            "testset_fingerprint_present": "缺少 testset fingerprint",
            "testset_validation_passed": "testset 校验未通过",
            "prediction_artifact_present": "缺少逐视频/逐图 prediction artifact",
            "prediction_artifact_hashed": "prediction artifact 缺少 SHA-256",
            "metric_details_present": "缺少可复算的时序详细指标",
            "prediction_artifact_recomputable": "时序 prediction artifact 无法复算指标",
        }
        failed_checks = [name for name, passed in checks.items() if not passed]
        if failed_checks:
            report["failed_checks"] = failed_checks
            report["issues"].extend(messages[name] for name in failed_checks)
        report["ok"] = report["ok"] and all(checks.values())
    if not report["issues"]:
        report.pop("issues")
    return report


def check_envelope_complete(envelope: Any) -> dict:
    """历史兼容名称；正式实现由 ``check_result_complete`` 提供。"""

    return check_result_complete(envelope)


__all__ = [
    "EvaluationProfileError",
    "assert_evaluation_profile",
    "check_envelope_complete",
    "check_result_complete",
]
