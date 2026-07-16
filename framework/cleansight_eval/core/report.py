"""checkpoint 级评估报告写入。

每次 framework 评估除了结构化 EvaluationResult JSON，也在 checkpoint 旁边生成一份
面向该 ``.pt`` 的 Markdown 报告，并向同目录的单一版本管理报告追加本次记录。
"""

from __future__ import annotations

from pathlib import Path

from .envelope import EvaluationResult, MetricValue
from .environment import now_stamp


VERSION_REPORT_NAME = "EVALUATION_REPORT.md"


def _report_category(result: EvaluationResult) -> str:
    """按流水线把公共版本报告归类为时序或 YOLO 探测模型。"""

    if result.pipeline in {"sliding_window_temporal", "full_sequence_temporal"}:
        return "时序模型"
    if result.pipeline == "detection" or result.model_type == "yolo":
        return "YOLO 探测模型"
    return "其他模型"


def _display_metric(value: MetricValue) -> str:
    """渲染三态指标，避免把 N/A、MISSING 与真实 0 混淆。"""

    rendered = value.display()
    if value.spec:
        rendered += f" (`{value.spec}`)"
    if value.reason:
        rendered += f" — {value.reason}"
    return rendered


def render_checkpoint_report(result: EvaluationResult, result_path: str | Path) -> str:
    """把单次正式评估结果渲染成 checkpoint 专属 Markdown。"""

    lines = [
        f"# Checkpoint 评估报告：{Path(result.checkpoint).name}",
        "",
        "## 基本信息",
        "",
        f"- 评估时间：`{result.timestamp or now_stamp()}`",
        f"- 模型类型：`{result.model_type}`",
        f"- 模型 ID：`{result.model_id}`",
        f"- 流水线：`{result.pipeline}`",
        f"- checkpoint：`{result.checkpoint}`",
        f"- evaluation result：`{result_path}`",
        f"- dataset：`{result.dataset}`",
        f"- testset：`{result.testset.get('id', '未登记')}`",
        f"- split：`{result.testset.get('split', '未知')}`",
        f"- testset fingerprint：`{result.testset.get('fingerprint_sha256', 'MISSING')}`",
        f"- device：`{result.run.get('device', '未知')}`",
        f"- checkpoint SHA-256：`{result.checkpoint_info.get('sha256', 'MISSING')}`",
        f"- 参数量：`{result.num_params if result.num_params is not None else '未知'}`",
        "",
        "## Feature Schema",
        "",
    ]
    if result.feature_schema:
        for key, value in result.feature_schema.items():
            lines.append(f"- {key}: `{value}`")
    else:
        lines.append("- N/A")

    lines += [
        "",
        "## 指标",
        "",
        "| 指标 | 结果 |",
        "| --- | --- |",
    ]
    if result.metrics:
        for name, metric in result.metrics.items():
            lines.append(f"| {name} | {_display_metric(metric)} |")
    else:
        lines.append("| 无 | MISSING |")

    lines += [
        "",
        "## 性能",
        "",
        "| 指标 | 结果 |",
        "| --- | --- |",
    ]
    if result.performance:
        for name, metric in result.performance.items():
            lines.append(f"| {name} | {_display_metric(metric)} |")
    else:
        lines.append("| 无 | N/A |")

    lines += [
        "",
        "## 推理语义",
        "",
    ]
    if result.inference_semantics:
        for key, value in result.inference_semantics.items():
            lines.append(f"- {key}: `{value}`")
    else:
        lines.append("- N/A")

    integrity = result.integrity or {}
    lines += [
        "",
        "## 完整性检查",
        "",
        f"- ok：`{integrity.get('ok')}`",
    ]
    for name, passed in (integrity.get("checks") or {}).items():
        lines.append(f"- check `{name}`：`{passed}`")
    for issue in integrity.get("issues", []) or []:
        lines.append(f"- issue：{issue}")

    lines += ["", "## Artifacts", ""]
    if result.artifacts:
        for name, value in result.artifacts.items():
            lines.append(f"- {name}：`{value}`")
    else:
        lines.append("- MISSING")

    lines += [
        "",
        "## 人工维护区（评估后填写）",
        "",
        "- 结论：",
        "- 主要问题：",
        "- 是否进入版本/ModelScope：",
        "- 下一步动作：",
        "",
    ]
    return "\n".join(lines)


def write_checkpoint_reports(result: EvaluationResult, result_path: str | Path) -> tuple[Path, Path]:
    """写 checkpoint 旁路报告，并向同目录单文件版本报告追加本次评估记录。

    返回 ``(checkpoint_report, version_report)``。``checkpoint_report`` 是当前
    ``.pt`` 的专属报告，固定为 ``<checkpoint>.eval.md``；``version_report`` 是
    checkpoint 所在目录下唯一的 ``EVALUATION_REPORT.md``，每次评估 append 一段。
    """

    ckpt = Path(result.checkpoint)
    report_path = ckpt.with_suffix(".eval.md")
    version_report = ckpt.parent / VERSION_REPORT_NAME
    report_text = render_checkpoint_report(result, result_path)
    report_path.write_text(report_text + "\n", encoding="utf-8")

    category = _report_category(result)
    entry = [
        "",
        "---",
        "",
        f"### {result.timestamp or now_stamp()} · {ckpt.name}",
        "",
        f"- checkpoint 专属报告：`{report_path.name}`",
        "",
        report_text,
    ]
    if version_report.exists():
        with version_report.open("a", encoding="utf-8") as f:
            existing = version_report.read_text(encoding="utf-8")
            if f"## {category}" not in existing:
                f.write(f"\n## {category}\n")
            f.write("\n".join(entry) + "\n")
    else:
        header = [
            "# 版本化评估报告",
            "",
            "本文件由 framework eval 追加维护；同目录下每个 `.pt` 仍保留自己的 `*.eval.md`。",
            "",
            f"## {category}",
        ]
        version_report.write_text("\n".join(header + entry) + "\n", encoding="utf-8")
    return report_path, version_report
