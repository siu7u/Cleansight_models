#!/usr/bin/env python3
"""通过 framework 比较同一时序 checkpoint 的完整序列和滑窗推理。

本 benchmark 只定义比较口径、裁剪范围、指标和报告，不构造模型、不加载 checkpoint、
不执行 forward。两种 ``PredictionOutput`` 均由 framework Pipeline 产生。
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

from framework.cleansight_eval.core.metrics import temporal_metrics
from framework.cleansight_eval.core.config import load_config
from framework.cleansight_eval.core.environment import pick_device
from framework.cleansight_eval.core.registry import get_pipeline


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "benchmark" / "temporal_feed_mode"
LATEST_DIR = OUT_DIR / "latest"

DEFAULT_MODELS = {
    "gru": {
        "config": "framework/experiments/legacy-gru-v1.yaml",
        "checkpoint": "registry/temporal/gru-v1/gru-final-20260704-150629.pt",
    },
    "tcn": {
        "config": "framework/experiments/legacy-causal-tcn-v1.yaml",
        "checkpoint": "registry/temporal/causal-tcn-v1/tcn-final-20260704-160652.pt",
    },
    "transformer": {
        "config": "framework/experiments/legacy-causal-transformer-v1.yaml",
        "checkpoint": (
            "registry/temporal/causal-transformer-v1/"
            "transformer-final-20260704-161653.pt"
        ),
    },
}


def build_run_id(version: str | None) -> str:
    """生成用于归档 benchmark summary 的版本化运行编号。"""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not version:
        return timestamp
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", version.strip()).strip("-")
    return f"{slug or 'version'}-{timestamp}"


def _limited_items(
    values: dict[str, list],
    *,
    max_videos: int | None,
    max_frames: int | None,
) -> dict[str, list]:
    """按稳定 item 顺序应用 smoke 限制，不改变 framework 的推理实现。"""

    names = list(values)
    if max_videos is not None:
        names = names[:max_videos]
    return {
        name: list(values[name][:max_frames] if max_frames is not None else values[name])
        for name in names
    }


def _aligned_outputs(
    full_output,
    stream_output,
    *,
    window: int,
    max_videos: int | None,
    max_frames: int | None,
) -> tuple[dict[str, list], dict[str, list], dict[str, list]]:
    """把两种输出裁剪到相同 item 和帧范围，并移除滑窗冷启动帧。"""

    full_predictions = _limited_items(
        full_output.predictions,
        max_videos=max_videos,
        max_frames=max_frames,
    )
    stream_predictions = _limited_items(
        stream_output.predictions,
        max_videos=max_videos,
        max_frames=max_frames,
    )
    truths = _limited_items(
        stream_output.targets,
        max_videos=max_videos,
        max_frames=max_frames,
    )
    names = [
        name
        for name in full_predictions
        if name in stream_predictions and name in truths
    ]
    start = max(window - 1, 0)
    return (
        {name: full_predictions[name][start:] for name in names},
        {name: stream_predictions[name][start:] for name in names},
        {name: truths[name][start:] for name in names},
    )


def score_predictions(
    pred_by_video: dict[str, list],
    truth_by_video: dict[str, list],
    labels: list[str],
) -> dict:
    """按 benchmark 唯一指标内核汇总帧级和片段级结果。"""

    raw = temporal_metrics(
        pred_by_video,
        truth_by_video,
        labels=labels,
        thresholds=(0.1, 0.25, 0.5),
    )
    frame = raw["frame"]
    segment = raw["segment"]
    recalls = {
        label: frame["per_class"].get(str(label), {}).get("recall")
        for label in labels
    }
    return {
        "num_frames": int(frame["num_frames"]),
        "acc": round(float(frame["accuracy"] or 0.0) * 100, 2),
        "edit": round(float(segment["edit"] or 0.0) * 100, 2),
        "f1": {
            str(threshold): round(
                float(segment["details_at_iou"][f"{threshold:.2f}"]["f1"]) * 100,
                2,
            )
            for threshold in (0.1, 0.25, 0.5)
        },
        "per_class_recall": recalls,
        "confusion_matrix_rows_gt_cols_pred": frame[
            "confusion_matrix_rows_truth_cols_prediction"
        ],
    }


def _latency_summary(timing: dict) -> dict:
    """从 framework 原始 tick 样本计算 benchmark 延迟统计。"""

    samples = [float(value) for value in timing.get("samples_ms", [])]
    if not samples:
        return {}
    ordered = sorted(samples)
    return {
        "mean_ms": round(float(statistics.mean(samples)), 4),
        "median_ms": round(float(statistics.median(samples)), 4),
        "p95_ms": round(float(ordered[int(0.95 * (len(ordered) - 1))]), 4),
        "num_ticks": len(samples),
        "scope": timing.get("scope"),
    }


def run_one(model_name: str, args: argparse.Namespace) -> dict:
    """调用 framework 两条 Pipeline，并返回 feed-mode 对比事实。"""

    item = DEFAULT_MODELS[model_name]
    config_path = ROOT / item["config"]
    checkpoint = ROOT / item["checkpoint"]
    base_cfg = load_config(config_path)
    if args.data_dir:
        base_cfg["data"]["root"] = str(Path(args.data_dir).expanduser().resolve())
    base_cfg["data"]["split_eval"] = args.split
    limits = dict((base_cfg.get("evaluation") or {}).get("limits") or {})
    if args.max_videos is not None:
        limits["max_videos"] = args.max_videos
    if args.max_frames is not None:
        limits["max_frames"] = args.max_frames
    if limits:
        limits["is_smoke"] = bool(args.max_videos or args.max_frames)
        base_cfg.setdefault("evaluation", {})["limits"] = limits
    device = pick_device(args.device)

    stream_cfg = copy.deepcopy(base_cfg)
    stream_cfg["pipeline"] = "sliding_window_temporal"
    stream_pipeline = get_pipeline(stream_cfg["pipeline"])
    stream_pipeline.validate_config(stream_cfg)
    stream_output = stream_pipeline.predict(stream_cfg, str(checkpoint), device)

    full_cfg = copy.deepcopy(base_cfg)
    full_cfg["pipeline"] = "full_sequence_temporal"
    full_pipeline = get_pipeline(full_cfg["pipeline"])
    full_pipeline.validate_config(full_cfg)
    full_output = full_pipeline.predict(full_cfg, str(checkpoint), device)

    window = int(base_cfg["train"]["window"])
    full_pred, stream_pred, truths = _aligned_outputs(
        full_output,
        stream_output,
        window=window,
        max_videos=args.max_videos,
        max_frames=args.max_frames,
    )
    labels = list(stream_output.labels)
    full_score = score_predictions(full_pred, truths, labels)
    stream_score = score_predictions(stream_pred, truths, labels)
    agreements = [
        sum(left == right for left, right in zip(full_pred[name], stream_pred[name]))
        / len(full_pred[name])
        for name in full_pred
        if full_pred[name]
    ]

    return {
        "model": model_name,
        "model_type": base_cfg["model"]["type"],
        "config": item["config"],
        "checkpoint": item["checkpoint"],
        "feature_mapping": base_cfg["feature_schema"]["version"],
        "input_dim": base_cfg["model"]["input_dim"],
        "window": window,
        "data_dir": base_cfg["data"]["root"],
        "split": args.split,
        "num_videos": len(full_pred),
        "max_frames": args.max_frames,
        "device": str(device),
        "labels": labels,
        "full_sequence": full_score,
        "streaming": {
            **stream_score,
            "latency": _latency_summary(stream_output.timing),
        },
        "full_vs_streaming_agreement": (
            round(float(statistics.mean(agreements)) * 100, 2)
            if agreements
            else 0.0
        ),
        "note": (
            "两种预测均由 framework Pipeline 产生；评分裁掉 window-1 冷启动帧。"
            "streaming 包含 framework 定义的因果平滑语义。"
        ),
    }


def write_outputs(results: list[dict], version: str | None) -> tuple[Path, Path]:
    """写 JSON 和 Markdown feed-mode 报告。"""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_DIR.mkdir(parents=True, exist_ok=True)
    archive_dir = OUT_DIR / "reports"
    archive_dir.mkdir(parents=True, exist_ok=True)
    run_id = build_run_id(version)
    payload = {
        "benchmark": "temporal_feed_mode",
        "version": version,
        "run_id": run_id,
        "models": results,
    }
    json_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    latest_json = LATEST_DIR / "feed_mode_summary.json"
    archive_json = archive_dir / f"feed_mode_summary_{run_id}.json"
    latest_json.write_text(json_text, encoding="utf-8")
    archive_json.write_text(json_text, encoding="utf-8")

    lines = [
        "# 时序喂法 Benchmark：整段喂 vs 流式喂",
        "",
        f"- 版本：`{version or run_id}`",
        f"- 归档编号：`{run_id}`",
        "- 模型加载与推理：`framework`",
        "- 指标与报告：`benchmark`",
        "",
        "| 模型 | 视频数 | 输入维度 | Full Acc | Stream Acc | Full Edit | Stream Edit | Full F1@0.5 | Stream F1@0.5 | 一致率 | Stream p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        full = item["full_sequence"]
        stream = item["streaming"]
        latency = stream.get("latency", {})
        lines.append(
            "| {model} | {videos} | {dim} | {fa:.2f} | {sa:.2f} | {fe:.2f} | {se:.2f} | {ff} | {sf} | {agree:.2f}% | {p95} |".format(
                model=item["model"],
                videos=item["num_videos"],
                dim=item["input_dim"],
                fa=full["acc"],
                sa=stream["acc"],
                fe=full["edit"],
                se=stream["edit"],
                ff=full["f1"].get("0.5", "NA"),
                sf=stream["f1"].get("0.5", "NA"),
                agree=item["full_vs_streaming_agreement"],
                p95=(
                    f"{latency['p95_ms']:.4f} ms"
                    if latency.get("p95_ms") is not None
                    else "NA"
                ),
            )
        )
    lines += ["", "## 逐类召回", ""]
    for item in results:
        lines += [
            f"### {item['model']}",
            "",
            "| 类别 | Full Recall | Stream Recall |",
            "| --- | ---: | ---: |",
        ]
        full_recalls = item["full_sequence"]["per_class_recall"]
        stream_recalls = item["streaming"]["per_class_recall"]
        for label in item["labels"]:
            full_value = full_recalls.get(label)
            stream_value = stream_recalls.get(label)
            lines.append(
                "| {label} | {full} | {stream} |".format(
                    label=label,
                    full=(
                        f"{full_value * 100:.2f}%"
                        if full_value is not None
                        else "NA"
                    ),
                    stream=(
                        f"{stream_value * 100:.2f}%"
                        if stream_value is not None
                        else "NA"
                    ),
                )
            )
        lines.append("")
    latest_md = LATEST_DIR / "feed_mode_summary.md"
    archive_md = archive_dir / f"feed_mode_summary_{run_id}.md"
    md_text = "\n".join(lines) + "\n"
    latest_md.write_text(md_text, encoding="utf-8")
    archive_md.write_text(md_text, encoding="utf-8")
    return latest_md, archive_md


def main() -> int:
    """解析参数并运行选定模型的 feed-mode benchmark。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(DEFAULT_MODELS), help="只跑一个模型")
    parser.add_argument("--data-dir", help="覆盖 catalog 的 Endo Project 本地挂载")
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/mps")
    parser.add_argument("--max-videos", type=int, help="只推理并评分前 N 个视频")
    parser.add_argument("--max-frames", type=int, help="每个视频只推理并评分前 N 帧")
    parser.add_argument("--version", help="本次 benchmark 版本名")
    args = parser.parse_args()
    if args.max_videos is not None and args.max_videos <= 0:
        parser.error("--max-videos 必须大于 0")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames 必须大于 0")
    selected = [args.model] if args.model else list(DEFAULT_MODELS)
    results = [run_one(model_name, args) for model_name in selected]
    latest_md, archive_md = write_outputs(results, args.version)
    print(f"已写入 {latest_md}")
    print(f"已归档 {archive_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
