"""通用完整性检查（framework 层）。

需求 §10 要求完整性检查只描述"评估是否可解释"，不描述"模型是否达标"。
本模块提供与模型语义无关的检查：checkpoint 与配置是否兼容、特征 schema 是否
兼容、正式结果字段是否齐全。硬性不兼容（例如错配 checkpoint）应直接抛错，避免
静默加载错误配置（§7.2 / §8.1）。
"""

from __future__ import annotations

from typing import Any


class CompatibilityError(Exception):
    """checkpoint / 特征 / 配置出现硬性不兼容时抛出。"""


def assert_evaluation_profile(cfg: dict, testset: dict) -> None:
    """正式评估必须使用已登记且校验通过的 testset；探索模式允许降级留痕。"""

    mode = (cfg.get("evaluation") or {}).get("mode", "formal")
    if mode not in {"formal", "exploratory"}:
        raise ValueError(f"evaluation.mode 必须是 formal 或 exploratory，当前为 {mode!r}")
    if mode == "exploratory":
        return
    if not testset.get("registered"):
        raise CompatibilityError("formal 评估必须声明 benchmark 已登记 testset")
    errors = testset.get("validation_errors") or []
    if errors:
        raise CompatibilityError("formal testset 校验失败:\n  - " + "\n  - ".join(map(str, errors)))


def check_checkpoint_config(meta: dict, expected: dict | None) -> list[str]:
    """比对 checkpoint 元信息与期望配置，返回不兼容项列表。

    只比较会改变输入输出契约或权重形状的字段：type、input_dim、num_classes。
    ``window`` 等不改变权重形状的超参数只提示、不算硬性不兼容。
    """

    if expected is None:
        return []

    problems: list[str] = []
    for key in ("type", "input_dim", "num_classes"):
        if key in expected and key in meta and meta[key] != expected[key]:
            problems.append(f"{key}: checkpoint={meta[key]!r} != expected={expected[key]!r}")
    return problems


def assert_checkpoint_config(meta: dict, expected: dict | None) -> None:
    """硬性校验 checkpoint 配置，不兼容立即抛 ``CompatibilityError``。"""

    problems = check_checkpoint_config(meta, expected)
    if problems:
        raise CompatibilityError(
            "checkpoint 与期望配置不兼容，拒绝静默加载:\n  - " + "\n  - ".join(problems)
        )


def check_feature_schema(actual_dim: int, expected: dict | None) -> list[str]:
    """检查实际特征维度与期望 feature schema 是否一致。"""

    if not expected:
        return []
    problems: list[str] = []
    exp_dim = expected.get("dim")
    if exp_dim is not None and exp_dim != actual_dim:
        problems.append(f"feature dim: actual={actual_dim} != expected={exp_dim}")
    return problems


def check_result_complete(result: Any) -> dict:
    """检查评估结果是否具备最小可解释字段，返回完整性报告（不做达标判断）。"""

    report: dict[str, Any] = {"ok": True, "issues": []}
    required = ("model_type", "pipeline", "checkpoint", "dataset")
    for key in required:
        if not getattr(result, key, None):
            report["ok"] = False
            report["issues"].append(f"缺少必要字段: {key}")

    if not result.metrics:
        report["ok"] = False
        report["issues"].append("没有任何指标")

    # 指标口径必须声明（§8.2）：已计算的指标应带 spec。
    from .envelope import MetricState

    for name, mv in result.metrics.items():
        if mv.state is MetricState.COMPUTED and not mv.spec:
            report["ok"] = False
            report["issues"].append(f"指标 {name} 已计算但未声明口径(spec)")

    # CLI 完成 schema v2 溯源信息注入后才执行这些增强检查；pipeline 单测直接调用时
    # run 为空，仍只检查上面的模型事实字段。
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
