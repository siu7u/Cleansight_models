#!/usr/bin/env python3
"""比较整段喂与流式滑窗两种时序推理方式。

这个 benchmark 只评估同一 checkpoint 的输入方式差异，不引入新的模型族：

- full_sequence：一次输入完整特征序列 `[1, T, F]`。
- streaming：维护因果滑窗，每次输入 `[1, window, F]` 并预测最新帧。

两种模式会裁剪到同一帧范围，并使用相同的逐帧指标和片段指标评分。
当前 v1 时序 checkpoint 默认使用 CleanSightBackend 下旧版 20 维
Endo_Project 特征集。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT.parent / "CleanSightBackend" / "MS-TCN2" / "data" / "Endo_Project"
OUT_DIR = ROOT / "benchmark" / "temporal_feed_mode"
LATEST_DIR = OUT_DIR / "latest"

DEFAULT_MODELS = {
    "gru": {
        "repo": "temporal-gru",
        "checkpoint": "registry/gru-v1/gru-final-20260704-150629.pt",
        "input_dim": 20,
        "window": 64,
    },
    "tcn": {
        "repo": "temporal-causal-tcn",
        "checkpoint": "registry/tcn-v1/tcn-final-20260704-160652.pt",
        "input_dim": 20,
        "window": 64,
    },
    "transformer": {
        "repo": "temporal-transformer",
        "checkpoint": "registry/transformer-v1/transformer-final-20260704-161653.pt",
        "input_dim": 20,
        "window": 64,
    },
}


def build_run_id(version: str | None) -> str:
    """生成用于归档 benchmark summary 的版本化运行编号。"""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not version:
        return timestamp
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", version.strip()).strip("-")
    return f"{slug or 'version'}-{timestamp}"


@dataclass
class EvalItem:
    """从 split 中加载的一段时序评测序列。

    `features` 形状为 `[T, F]`，`labels` 形状为 `[T]`；两者会在评分前
    裁剪到相同帧数。
    """

    name: str
    features: np.ndarray
    labels: np.ndarray


def load_mapping(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    """读取 MS-TCN 风格的 `mapping.txt`，返回动作到编号和编号到动作的映射。"""

    action_to_idx: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("._"):
            continue
        idx, action = line.split()
        action_to_idx[action] = int(idx)
    return action_to_idx, {v: k for k, v in action_to_idx.items()}


def load_split(path: Path) -> list[str]:
    """读取 split bundle 文件中的视频 id 列表。"""

    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_features(features_dir: Path, name: str) -> np.ndarray:
    """读取单个视频的旧版特征，并返回 `[T, F]` 的 float32 数组。"""

    raw = np.load(features_dir / f"{name}.npy")
    # 旧版 MS-TCN 数据按 [F, T] 存储；当前时序仓库统一消费 [T, F]。
    if raw.ndim != 2:
        raise ValueError(f"expected 2-D features for {name}, got shape={raw.shape}")
    return raw.T.astype(np.float32)


def load_labels(truths_dir: Path, name: str, action_to_idx: dict[str, int]) -> np.ndarray:
    """读取单个视频的逐帧文本标签，并转换为类别编号。"""

    labels = []
    for line in (truths_dir / f"{name}.txt").read_text(encoding="utf-8").splitlines():
        label = line.strip()
        if label:
            labels.append(action_to_idx[label])
    return np.asarray(labels, dtype=np.int64)


def load_eval_items(data_dir: Path, split_name: str) -> tuple[list[EvalItem], dict[int, str]]:
    """读取一个 split 内全部视频，并对齐特征长度和标签长度。"""

    action_to_idx, idx_to_action = load_mapping(data_dir / "mapping.txt")
    names = load_split(data_dir / "splits" / split_name)
    items = []
    for name in names:
        features = load_features(data_dir / "features", name)
        labels = load_labels(data_dir / "groundTruth", name, action_to_idx)
        common = min(len(features), len(labels))
        items.append(EvalItem(name=name, features=features[:common], labels=labels[:common]))
    return items, idx_to_action


def import_class(module_path: Path, class_name: str):
    """从仓库本地文件导入模型类，避免要求把子仓库安装成 Python 包。"""

    spec = importlib.util.spec_from_file_location(f"_feed_mode_{module_path.stem}_{class_name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def build_model(model_name: str, repo: Path, input_dim: int, num_classes: int):
    """按 checkpoint 约定的维度构造指定时序分类器。"""

    if model_name == "gru":
        cls = import_class(repo / "model" / "gru.py", "GRUClassifier")
        return cls(input_dim, num_classes)
    if model_name == "tcn":
        sys.path.insert(0, str(repo))
        try:
            cls = import_class(repo / "model" / "tcn.py", "TCNClassifier")
            return cls(input_dim, num_classes)
        finally:
            try:
                sys.path.remove(str(repo))
            except ValueError:
                pass
    if model_name == "transformer":
        cls = import_class(repo / "model" / "transformer.py", "TransformerClassifier")
        return cls(input_dim, num_classes)
    raise ValueError(f"unknown model: {model_name}")


def predict_full_sequence(model: torch.nn.Module, features: np.ndarray, device: torch.device) -> np.ndarray:
    """用 `[1, T, F]` 做一次整段前向推理，并返回 `[T]` 类别编号。"""

    x = torch.from_numpy(features).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits[0], dim=-1).cpu().numpy()
    return pred.astype(np.int64)


def predict_streaming(
    model: torch.nn.Module,
    features: np.ndarray,
    window: int,
    device: torch.device,
) -> tuple[np.ndarray, list[float]]:
    """执行因果滑窗推理，并记录每个在线 tick 的耗时。

    每次预测只看到最近 `[window, F]` 帧。前 `window - 1` 帧保留为 `-1`，
    会在公平对比时排除。
    """

    preds = np.full(len(features), fill_value=-1, dtype=np.int64)
    latencies_ms: list[float] = []

    with torch.no_grad():
        for end in range(window, len(features) + 1):
            x_np = features[end - window : end]
            x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            logits = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            preds[end - 1] = int(torch.argmax(logits[0, -1], dim=-1).cpu().item())

    return preds, latencies_ms


def segments(labels: Iterable[int], bg_class: set[int] | None = None) -> tuple[list[int], list[int], list[int]]:
    """把逐帧标签折叠为片段标签、起点索引和终点索引。"""

    values = list(labels)
    if not values:
        return [], [], []
    bg_class = bg_class or set()
    out_labels: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    last = values[0]
    if last not in bg_class:
        out_labels.append(last)
        starts.append(0)
    for i, value in enumerate(values):
        if value == last:
            continue
        if value not in bg_class:
            out_labels.append(value)
            starts.append(i)
        if last not in bg_class:
            ends.append(i)
        last = value
    if last not in bg_class:
        ends.append(len(values))
    return out_labels, starts, ends


def levenstein(pred: list[int], truth: list[int], norm: bool = True) -> float:
    """基于 Levenshtein 距离计算片段级 edit score。"""

    rows, cols = len(pred), len(truth)
    dp = np.zeros((rows + 1, cols + 1), dtype=np.float64)
    for i in range(rows + 1):
        dp[i, 0] = i
    for j in range(cols + 1):
        dp[0, j] = j
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if pred[i - 1] == truth[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + 1)
    if not norm:
        return float(dp[-1, -1])
    return float((1 - dp[-1, -1] / max(rows, cols, 1)) * 100)


def edit_score(pred: np.ndarray, truth: np.ndarray) -> float:
    """将逐帧标签折叠为片段后计算归一化 edit score。"""

    p_labels, _, _ = segments(pred)
    y_labels, _, _ = segments(truth)
    return levenstein(p_labels, y_labels, norm=True)


def f_score(pred: np.ndarray, truth: np.ndarray, overlap: float) -> tuple[float, float, float]:
    """返回片段级 true positive、false positive 和 false negative。"""

    p_label, p_start, p_end = segments(pred)
    y_label, y_start, y_end = segments(truth)
    if not p_label and not y_label:
        return 0.0, 0.0, 0.0
    if not p_label:
        return 0.0, 0.0, float(len(y_label))
    if not y_label:
        return 0.0, float(len(p_label)), 0.0

    tp = 0.0
    fp = 0.0
    hits = np.zeros(len(y_label), dtype=np.float32)
    y_start_np = np.asarray(y_start)
    y_end_np = np.asarray(y_end)
    for j, label in enumerate(p_label):
        intersection = np.minimum(p_end[j], y_end_np) - np.maximum(p_start[j], y_start_np)
        union = np.maximum(p_end[j], y_end_np) - np.minimum(p_start[j], y_start_np)
        iou = (intersection / np.maximum(union, 1)) * np.asarray([label == item for item in y_label])
        idx = int(np.argmax(iou))
        if iou[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = float(len(y_label) - hits.sum())
    return tp, fp, fn


def score_predictions(
    pred_by_video: dict[str, np.ndarray],
    truth_by_video: dict[str, np.ndarray],
    num_classes: int,
    idx_to_action: dict[int, str],
    start_frame: int,
) -> dict:
    """在共同裁剪后的帧范围上评分，保证两种喂法公平对比。"""

    all_pred = []
    all_truth = []
    edit_scores = []
    f_totals = {0.1: [0.0, 0.0, 0.0], 0.25: [0.0, 0.0, 0.0], 0.5: [0.0, 0.0, 0.0]}

    for name, pred in pred_by_video.items():
        truth = truth_by_video[name]
        pred_crop = pred[start_frame:]
        truth_crop = truth[start_frame:]
        valid = pred_crop >= 0
        pred_crop = pred_crop[valid]
        truth_crop = truth_crop[valid]
        if len(pred_crop) == 0:
            continue
        all_pred.append(pred_crop)
        all_truth.append(truth_crop)
        edit_scores.append(edit_score(pred_crop, truth_crop))
        for overlap in f_totals:
            tp, fp, fn = f_score(pred_crop, truth_crop, overlap)
            f_totals[overlap][0] += tp
            f_totals[overlap][1] += fp
            f_totals[overlap][2] += fn

    all_pred_np = np.concatenate(all_pred)
    all_truth_np = np.concatenate(all_truth)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for gt, pred in zip(all_truth_np, all_pred_np):
        if 0 <= gt < num_classes and 0 <= pred < num_classes:
            confusion[int(gt), int(pred)] += 1

    recalls = {}
    for i in range(num_classes):
        denom = int(confusion[i].sum())
        recalls[idx_to_action.get(i, str(i))] = None if denom == 0 else round(float(confusion[i, i] / denom), 4)

    f1 = {}
    for overlap, (tp, fp, fn) in f_totals.items():
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        value = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1[str(overlap)] = round(value * 100, 2)

    return {
        "num_frames": int(len(all_truth_np)),
        "acc": round(float((all_pred_np == all_truth_np).mean() * 100), 2),
        "edit": round(float(statistics.mean(edit_scores)) if edit_scores else 0.0, 2),
        "f1": f1,
        "per_class_recall": recalls,
        "confusion_matrix_rows_gt_cols_pred": confusion.tolist(),
    }


def summarize_latency(samples: list[float]) -> dict:
    """汇总流式推理每个 tick 的毫秒级延迟样本。"""

    if not samples:
        return {}
    sorted_samples = sorted(samples)
    return {
        "mean_ms": round(float(statistics.mean(samples)), 4),
        "median_ms": round(float(statistics.median(samples)), 4),
        "p95_ms": round(float(sorted_samples[int(0.95 * (len(sorted_samples) - 1))]), 4),
        "num_ticks": len(samples),
    }


def run_one(model_name: str, args: argparse.Namespace) -> dict:
    """加载一个 checkpoint，运行两种喂法，并返回可直接写报告的指标。"""

    cfg = DEFAULT_MODELS[model_name]
    repo = ROOT / cfg["repo"]
    data_dir = Path(args.data_dir).resolve()
    items, idx_to_action = load_eval_items(data_dir, args.split)
    if args.max_videos is not None:
        items = items[: args.max_videos]
    if args.max_frames is not None:
        limited_items = []
        for item in items:
            max_frames = min(args.max_frames, len(item.features), len(item.labels))
            limited_items.append(
                EvalItem(
                    name=item.name,
                    features=item.features[:max_frames],
                    labels=item.labels[:max_frames],
                )
            )
        items = limited_items
    num_classes = len(idx_to_action)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(model_name, repo, int(cfg["input_dim"]), num_classes).to(device)
    model.load_state_dict(torch.load(repo / cfg["checkpoint"], map_location=device))
    model.eval()

    full_pred: dict[str, np.ndarray] = {}
    stream_pred: dict[str, np.ndarray] = {}
    truth_by_video: dict[str, np.ndarray] = {}
    latencies: list[float] = []
    window = int(cfg["window"])

    for item in items:
        if item.features.shape[1] != int(cfg["input_dim"]):
            raise ValueError(
                f"{item.name}: feature dim {item.features.shape[1]} != expected {cfg['input_dim']}"
            )
        full_pred[item.name] = predict_full_sequence(model, item.features, device)
        pred, samples = predict_streaming(model, item.features, window, device)
        stream_pred[item.name] = pred
        truth_by_video[item.name] = item.labels
        latencies.extend(samples)

    start_frame = window - 1
    full_score = score_predictions(full_pred, truth_by_video, num_classes, idx_to_action, start_frame)
    stream_score = score_predictions(stream_pred, truth_by_video, num_classes, idx_to_action, start_frame)

    agreement_values = []
    for name in full_pred:
        full_crop = full_pred[name][start_frame:]
        stream_crop = stream_pred[name][start_frame:]
        valid = stream_crop >= 0
        agreement_values.append(float((full_crop[valid] == stream_crop[valid]).mean()))

    return {
        "model": model_name,
        "repo": cfg["repo"],
        "checkpoint": cfg["checkpoint"],
        "feature_mapping": "legacy-20d-v1",
        "input_dim": cfg["input_dim"],
        "window": window,
        "data_dir": str(data_dir),
        "split": args.split,
        "num_videos": len(items),
        "max_frames": args.max_frames,
        "device": str(device),
        "labels": [idx_to_action[i] for i in range(num_classes)],
        "full_sequence": full_score,
        "streaming": {**stream_score, "latency": summarize_latency(latencies)},
        "full_vs_streaming_agreement": round(float(statistics.mean(agreement_values)) * 100, 2),
        "note": (
            "Scores crop the first window-1 frames so both modes are evaluated on the same frames. "
            "For v1 temporal checkpoints this compares full causal sequence inference with "
            "online rolling-window inference on legacy 20-D features."
        ),
    }


def write_outputs(results: list[dict], version: str | None) -> tuple[Path, Path]:
    """在 `benchmark/temporal_feed_mode` 下写入 JSON 和 Markdown 报告。"""

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
        "",
        "本报告比较同一 checkpoint 在两种输入方式下的结果：",
        "",
        "- 整段喂：一次输入完整特征序列 `[1, T, F]`。",
        "- 流式喂：每 tick 输入最近 `window` 帧 `[1, window, F]`，只取最后一帧预测。",
        "",
        "评估时裁掉前 `window - 1` 帧，使两种模式在同一帧范围上对比。",
        "",
        "| 模型 | 视频数 | 最多帧数 | 输入维度 | Full Acc | Stream Acc | Full Edit | Stream Edit | Full F1@0.5 | Stream F1@0.5 | 一致率 | Stream p95 延迟 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        full = item["full_sequence"]
        stream = item["streaming"]
        latency = stream.get("latency", {})
        lines.append(
            "| {model} | {videos} | {frames} | {dim} | {fa:.2f} | {sa:.2f} | {fe:.2f} | {se:.2f} | {ff} | {sf} | {agree:.2f}% | {p95} |".format(
                model=item["model"],
                videos=item["num_videos"],
                frames=item["max_frames"] if item["max_frames"] is not None else "全量",
                dim=item["input_dim"],
                fa=full["acc"],
                sa=stream["acc"],
                fe=full["edit"],
                se=stream["edit"],
                ff=full["f1"].get("0.5", "NA"),
                sf=stream["f1"].get("0.5", "NA"),
                agree=item["full_vs_streaming_agreement"],
                p95=f"{latency.get('p95_ms'):.4f} ms" if latency.get("p95_ms") is not None else "NA",
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
                    full=f"{full_value * 100:.2f}%" if full_value is not None else "NA",
                    stream=f"{stream_value * 100:.2f}%" if stream_value is not None else "NA",
                )
            )
        lines.append("")

    md_text = "\n".join(lines) + "\n"
    latest_md = LATEST_DIR / "feed_mode_summary.md"
    archive_md = archive_dir / f"feed_mode_summary_{run_id}.md"
    latest_md.write_text(md_text, encoding="utf-8")
    archive_md.write_text(md_text, encoding="utf-8")
    return latest_md, archive_md


def main() -> int:
    """解析命令行参数，并运行选定的时序喂法 benchmark。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(DEFAULT_MODELS), help="只跑一个模型")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Endo_Project 数据目录")
    parser.add_argument("--split", default="test.split1.bundle", help="splits/ 下的 split 文件名")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--max-videos", type=int, help="只评测前 N 个视频；不传则全量")
    parser.add_argument("--max-frames", type=int, help="每个视频最多取前 N 帧；不传则全量")
    parser.add_argument("--version", help="为本次 benchmark summary 指定版本名，例如 temporal-v2")
    args = parser.parse_args()

    selected = [args.model] if args.model else list(DEFAULT_MODELS)
    results = [run_one(model_name, args) for model_name in selected]
    latest_md, archive_md = write_outputs(results, args.version)
    print(f"已写入 {latest_md}")
    print(f"已归档 {archive_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
