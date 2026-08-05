"""checkpoint 级 benchmark 评估报告写入。

每次 benchmark 评估除了结构化 EvaluationResult JSON，也在 checkpoint 旁边生成一份
面向该 ``.pt`` 的 Markdown 报告，并向同目录的单一版本管理报告追加本次记录。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from benchmark.core.result import EvaluationResult, MetricValue
from framework.cleansight_eval.core.environment import now_stamp


VERSION_REPORT_NAME = "EVALUATION_REPORT.md"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _markdown_file_link(
    path: str | Path,
    *,
    report_dir: Path,
    base_dir: Path | None = None,
) -> str:
    """把文件引用渲染为相对当前报告的 Markdown 链接。

    ``base_dir`` 用于解释 artifact 等以 run 为根的相对路径；其他仓库相对路径默认
    从当前工作目录解释。链接目标始终使用相对路径，保证整个 run 目录移动后仍可用。
    """

    display = str(path)
    target = Path(path)
    if not target.is_absolute():
        target = (base_dir or Path.cwd()) / target
    relative = os.path.relpath(target.resolve(), start=report_dir.resolve())
    return f"[`{display}`](<{Path(relative).as_posix()}>)"


def _run_dir_for_report(report_path: Path, result_path: Path) -> Path:
    """根据 checkpoint/evals 的标准布局定位 artifact 相对路径所使用的 run 根。"""

    if report_path.parent.name == "checkpoints":
        return report_path.parent.parent
    if result_path.parent.name == "evals":
        return result_path.parent.parent
    return result_path.parent


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


def _display_detail(value) -> str:
    """渲染逐类详情中的紧凑数值或 missing/not_applicable 状态。"""

    if not isinstance(value, Mapping):
        return str(value)
    state = value.get("state")
    if state == "missing":
        rendered = "MISSING"
    elif state == "not_applicable":
        rendered = "N/A"
    else:
        rendered = str(value.get("value", state or "—"))
    if value.get("reason"):
        rendered += f" — {value['reason']}"
    return rendered


def _append_per_class_metrics(lines: list[str], result: EvaluationResult) -> None:
    """把检测逐类 P/R 详情渲染成独立表格，避免污染主指标列表。"""

    details = result.metric_details or {}
    per_class = details.get("per_class")
    if not isinstance(per_class, Mapping) or not per_class:
        return
    lines += [
        "",
        "## 逐类指标",
        "",
        "| 类别 | Precision | Recall |",
        "| --- | --- | --- |",
    ]
    for class_name, values in per_class.items():
        values = values if isinstance(values, Mapping) else {}
        lines.append(
            f"| {class_name} | {_display_detail(values.get('precision', '—'))} | "
            f"{_display_detail(values.get('recall', '—'))} |"
        )
    specs = details.get("per_class_specs") or {}
    if specs:
        lines += ["", f"- 口径：precision `{specs.get('precision', '未声明')}`；recall `{specs.get('recall', '未声明')}`"]


def _append_artifacts(
    lines: list[str],
    result: EvaluationResult,
    *,
    report_dir: Path,
    run_dir: Path,
) -> None:
    """用表格展示 artifact 引用，避免直接打印 Python dict。"""

    lines += [
        "",
        "## Artifacts",
        "",
        "| 产物 | 路径 | Schema | SHA-256 | 状态 |",
        "| --- | --- | --- | --- | --- |",
    ]
    if not result.artifacts:
        lines.append("| 无 | MISSING | — | — | missing |")
        return
    for name, raw_reference in result.artifacts.items():
        references = raw_reference if isinstance(raw_reference, list) else [raw_reference]
        for index, reference in enumerate(references, start=1):
            label = name if len(references) == 1 else f"{name}[{index}]"
            if not isinstance(reference, Mapping):
                link = _markdown_file_link(reference, report_dir=report_dir, base_dir=run_dir)
                lines.append(f"| {label} | {link} | — | — | — |")
                continue
            if reference.get("recomputable") is True:
                state = "可复算"
            elif reference.get("recomputable") is False:
                state = "不可复算"
            elif result.pipeline == "detection" and name == "predictions":
                state = "需结合 testset 真值"
            else:
                state = str(reference.get("state") or "—")
            artifact_path = reference.get("path")
            rendered_path = (
                _markdown_file_link(artifact_path, report_dir=report_dir, base_dir=run_dir)
                if artifact_path
                else "—"
            )
            lines.append(
                f"| {label} | {rendered_path} | "
                f"{reference.get('schema_version', '—')} | `{reference.get('sha256', '—')}` | {state} |"
            )


def render_checkpoint_report(
    result: EvaluationResult,
    result_path: str | Path,
    report_path: str | Path | None = None,
) -> str:
    """把单次正式评估结果渲染成 checkpoint 专属 Markdown。"""

    result_path = Path(result_path)
    report_path = (
        Path(report_path)
        if report_path is not None
        else Path(result.checkpoint).with_suffix(".eval.md")
    )
    report_dir = report_path.parent
    run_dir = _run_dir_for_report(report_path, result_path)
    checkpoint_link = _markdown_file_link(result.checkpoint, report_dir=report_dir)
    result_link = _markdown_file_link(result_path, report_dir=report_dir)

    lines = [
        f"# Checkpoint 评估报告：{Path(result.checkpoint).name}",
        "",
        "## 基本信息",
        "",
        f"- 评估时间：`{result.timestamp or now_stamp()}`",
        f"- 模型类型：`{result.model_type}`",
        f"- 模型 ID：`{result.model_id}`",
        f"- 流水线：`{result.pipeline}`",
        f"- checkpoint：{checkpoint_link}",
        f"- evaluation result：{result_link}",
        f"- dataset：`{result.dataset}`",
        f"- testset：`{result.testset.get('id', '未登记')}`",
        f"- split：`{result.testset.get('split', '未知')}`",
        f"- testset fingerprint：`{result.testset.get('fingerprint_sha256', 'MISSING')}`",
        f"- checkpoint SHA-256：`{result.checkpoint_info.get('sha256', 'MISSING')}`",
        f"- 参数量：`{result.num_params if result.num_params is not None else '未知'}`",
    ]
    config_path = result.run.get("config")
    if config_path:
        lines.append(
            f"- config：{_markdown_file_link(config_path, report_dir=report_dir, base_dir=_REPO_ROOT)}"
        )
    lines += ["", "## Feature Schema", ""]
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

    _append_per_class_metrics(lines, result)

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
    for name in integrity.get("failed_checks", []) or []:
        lines.append(f"- failed check：`{name}`")
    for issue in integrity.get("issues", []) or []:
        lines.append(f"- issue：{issue}")

    _append_artifacts(lines, result, report_dir=report_dir, run_dir=run_dir)

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
    report_text = render_checkpoint_report(result, result_path, report_path)
    report_path.write_text(report_text + "\n", encoding="utf-8")

    category = _report_category(result)
    entry = [
        "",
        "---",
        "",
        f"### {result.timestamp or now_stamp()} · {ckpt.name}",
        "",
        f"- checkpoint 专属报告：[{report_path.name}](<{report_path.name}>)",
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
            "本文件由 benchmark eval 追加维护；同目录下每个 `.pt` 仍保留自己的 `*.eval.md`。",
            "",
            f"## {category}",
        ]
        version_report.write_text("\n".join(header + entry) + "\n", encoding="utf-8")
    return report_path, version_report
