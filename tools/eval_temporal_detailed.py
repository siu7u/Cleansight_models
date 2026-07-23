#!/usr/bin/env python3
"""时序模型评估：保留视频边界，输出逐视频预测与统一 envelope。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.core.metrics import temporal_metrics  # noqa: E402
from benchmark.core.result import build_result, make_run_id, write_result  # noqa: E402
from benchmark.core.temporal_data import TemporalItem, load_temporal_items  # noqa: E402
from benchmark.core.artifacts import build_temporal_prediction_artifact  # noqa: E402
from benchmark.core.testsets import get_testset, manifest_fingerprint  # noqa: E402
from tools.card_history import append_evaluation_record  # noqa: E402


VIDEO_NAMES = [
    "export1", "export2", "export3", "export4",
    "export5", "export6", "export7-480p", "export8",
    "export9", "export10", "export11", "export12",
    "export13", "export14", "export15-480P", "export16-480P",
    "export17", "export18", "export19", "export20",
]
TEST_IDX = list(range(16, 20))


def build_model(model_name: str, input_dim: int, num_classes: int):
    """按旧模型目录中的公开类名构造 `[B,T,F] -> [B,T,C]` 分类器。"""

    if model_name == "gru":
        from model import GRUClassifier

        return GRUClassifier(input_dim, num_classes)
    if model_name == "tcn":
        from model import TCNClassifier

        return TCNClassifier(input_dim, num_classes)
    if model_name == "transformer":
        from model import TransformerClassifier

        return TransformerClassifier(input_dim, num_classes)
    raise ValueError(f"unknown model: {model_name}")


def parse_args() -> argparse.Namespace:
    """解析 legacy 时序评测参数；模型定位显式传入，不再依赖集中 catalog。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="legacy 时序模型目录")
    parser.add_argument(
        "--model",
        required=True,
        choices=["gru", "tcn", "transformer"],
        help="legacy 模型类名",
    )
    parser.add_argument("--checkpoint", required=True, help="相对 repo 或绝对 checkpoint 路径")
    parser.add_argument("--model-id", help="只写入结果的稳定模型 ID，不再用于 catalog 查询")
    parser.add_argument("--testset", help="可选：benchmark/testsets.yaml 中的 temporal testset id")
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda")
    parser.add_argument("--max-videos", type=int, help="smoke test: 最多评估多少个视频")
    parser.add_argument("--max-frames", type=int, help="smoke test: 每个视频最多评估多少帧")
    parser.add_argument("--output-dir", help="写 predictions/envelope 的目录")
    parser.add_argument("--card", help="可选：向 CARD.md 追加评估记录")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--input-dim", type=int, default=20)
    parser.add_argument("--num-classes", type=int, default=3)
    return parser.parse_args()


def _device_from_arg(value: str) -> torch.device:
    """解析 device 参数；auto 优先使用 CUDA。"""

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _checkpoint_path(repo: Path, checkpoint: str | Path) -> Path:
    """旧脚本允许 checkpoint 写相对 repo 或绝对路径。"""

    path = Path(checkpoint).expanduser()
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _load_legacy_items(repo: Path, max_videos: int | None, max_frames: int | None) -> tuple[list[TemporalItem], dict[int, str]]:
    """按旧目录约定读取固定 test split，并返回独立视频序列。"""

    sys.path.insert(0, str(repo))

    from util import load_features, load_mappings, load_truths

    mapping_path = repo / "data" / "Endo_Project" / "mapping.txt"
    features_dir = repo / "data" / "Endo_Project" / "features"
    truths_dir = repo / "data" / "Endo_Project" / "groundTruth"
    mappings = load_mappings(str(mapping_path))
    idx_to_action = {v: k for k, v in mappings.items()}

    names = [VIDEO_NAMES[index] for index in TEST_IDX]
    if max_videos is not None:
        names = names[:max_videos]
    items: list[TemporalItem] = []
    for name in names:
        features = load_features(str(features_dir), name)
        labels = np.asarray(load_truths(str(truths_dir), name, mappings=mappings), dtype=np.int64)
        common = min(len(features), len(labels))
        if max_frames is not None:
            common = min(common, max_frames)
        if common <= 0:
            raise ValueError(f"{name}: 没有可评估帧")
        items.append(
            TemporalItem(
                name=name,
                features=np.asarray(features[:common], dtype=np.float32),
                labels=labels[:common],
            )
        )
    return items, idx_to_action


def _predict_item_last_frame(
    model: torch.nn.Module,
    item: TemporalItem,
    *,
    window: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """对单个视频按滑窗末帧语义预测，返回与真值等长的 `[T-window+1]` 序列。"""

    if len(item.features) < window:
        return [], []
    predictions: list[int] = []
    truth = item.labels[window - 1 :].astype(np.int64).tolist()
    starts = list(range(0, len(item.features) - window + 1))
    with torch.no_grad():
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            windows = np.stack(
                [item.features[start : start + window] for start in batch_starts],
                axis=0,
            )
            logits = model(torch.from_numpy(windows).float().to(device))
            pred = torch.argmax(logits[:, -1, :], dim=-1).cpu().numpy()
            predictions.extend(int(value) for value in pred)
    return predictions, [int(value) for value in truth]


def build_predictions_artifact(
    *,
    pred_by_item: dict[str, list[int]],
    truth_by_item: dict[str, list[int]],
    index_to_action: dict[int, str],
    window: int,
    inference_mode: str,
) -> dict:
    """向后兼容的本地包装；真实 schema 定义在 benchmark.core.artifacts。"""

    return build_temporal_prediction_artifact(
        pred_by_item=pred_by_item,
        truth_by_item=truth_by_item,
        index_to_action=index_to_action,
        window=window,
        inference_mode=inference_mode,
    )


def _write_json(path: Path, payload: dict) -> Path:
    """写出 UTF-8 JSON artifact，并自动创建父目录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _confusion_from_metrics(metrics: dict) -> list[list[int]]:
    return metrics["frame"]["confusion_matrix_rows_truth_cols_prediction"]


def _legacy_summary(
    *,
    model_name: str,
    checkpoint: Path,
    device: torch.device,
    pred_by_item: dict[str, list[int]],
    truth_by_item: dict[str, list[int]],
    metrics: dict,
    index_to_action: dict[int, str],
) -> dict:
    """保留旧脚本 stdout 的关键字段，同时增加边界感知指标。"""

    per_class = metrics["frame"]["per_class"]
    labels = [index_to_action.get(index, str(index)) for index in sorted(index_to_action)]
    recalls = {
        label: (
            None
            if per_class.get(str(index), {}).get("recall") is None
            else round(float(per_class[str(index)]["recall"]), 4)
        )
        for index, label in (
            (index, index_to_action.get(index, str(index)))
            for index in sorted(index_to_action)
        )
    }
    return {
        "model": model_name,
        "checkpoint": str(checkpoint),
        "device": str(device),
        "num_windows": int(sum(len(values) for values in truth_by_item.values())),
        "note": "Classification metrics use per-video sliding windows and last-frame logits, without causal_decision smoothing.",
        "labels": labels,
        "per_class_recall": recalls,
        "confusion_matrix_rows_gt_cols_pred": _confusion_from_metrics(metrics),
        "metrics": metrics,
    }


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).resolve()
    model_name = args.model
    checkpoint = _checkpoint_path(repo, args.checkpoint)
    input_dim = args.input_dim
    num_classes = args.num_classes
    testset = get_testset(args.testset) if args.testset else None
    labels = list(testset.labels) if testset is not None else []

    if testset is not None:
        if testset.family != "temporal":
            raise SystemExit(f"testset {testset.id!r} 不是 temporal 数据")
        if testset.input_dim is not None and testset.input_dim != input_dim:
            raise SystemExit(
                f"--input-dim={input_dim} 与 testset input_dim={testset.input_dim} 不一致"
            )
        if labels and len(labels) != num_classes:
            raise SystemExit(
                f"--num-classes={num_classes} 与 testset labels={len(labels)} 不一致"
            )
        items, index_to_action = load_temporal_items(
            testset,
            max_videos=args.max_videos,
            max_frames=args.max_frames,
        )
    else:
        items, index_to_action = _load_legacy_items(repo, args.max_videos, args.max_frames)
    sys.path.insert(0, str(repo))

    device = _device_from_arg(args.device)
    model = build_model(model_name, input_dim, num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    pred_by_item: dict[str, list[int]] = {}
    truth_by_item: dict[str, list[int]] = {}
    for item in items:
        predictions, truths = _predict_item_last_frame(
            model,
            item,
            window=args.window,
            batch_size=args.batch_size,
            device=device,
        )
        if predictions:
            pred_by_item[item.name] = predictions
            truth_by_item[item.name] = truths

    if not pred_by_item:
        raise SystemExit("没有可评估窗口；请检查 window/max_frames 是否过大")

    label_ids = list(range(num_classes))
    metrics = temporal_metrics(
        pred_by_item,
        truth_by_item,
        labels=label_ids,
        start_frame=0,
        thresholds=(0.1, 0.25, 0.5),
    )
    result = _legacy_summary(
        model_name=model_name,
        checkpoint=checkpoint,
        device=device,
        pred_by_item=pred_by_item,
        truth_by_item=truth_by_item,
        metrics=metrics,
        index_to_action=index_to_action,
    )

    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    if args.model_id and output_dir is None:
        output_dir = ROOT / "benchmark" / "single_model" / "latest" / args.model_id
    if output_dir is not None:
        run_id = make_run_id(args.model_id or f"temporal-{model_name}")
        predictions_path = _write_json(
            output_dir / "predictions_by_video.json",
            build_predictions_artifact(
                pred_by_item=pred_by_item,
                truth_by_item=truth_by_item,
                index_to_action=index_to_action,
                window=args.window,
                inference_mode="raw_last_frame",
            ),
        )
        result["artifacts"] = {"predictions": str(predictions_path)}
        if testset is not None:
            model_id = args.model_id or f"legacy.{model_name}"
            envelope = build_result(
                benchmark="single_model_temporal",
                task_type="temporal",
                run_id=run_id,
                model={
                    "id": model_id,
                    "family": "temporal",
                    "adapter": "legacy_explicit",
                    "checkpoint": str(checkpoint),
                    "input_dim": input_dim,
                    "window": args.window,
                    "labels": labels,
                },
                testset={
                    "id": testset.id,
                    "dataset_version": testset.dataset_version,
                    "split": testset.split,
                    "manifest_sha256": manifest_fingerprint(testset),
                    "feature_mapping": testset.feature_mapping,
                    "input_dim": testset.input_dim,
                    "labels": list(testset.labels),
                },
                inference={
                    "mode": "raw_last_frame",
                    "device": str(device),
                    "window": args.window,
                    "batch_size": args.batch_size,
                    "video_boundaries_preserved": True,
                },
                metrics=metrics,
                status="PASS",
                limits={
                    "is_smoke": args.max_videos is not None or args.max_frames is not None,
                    "max_videos": args.max_videos,
                    "max_frames": args.max_frames,
                },
                artifacts={"predictions": str(predictions_path)},
            )
            envelope_path = write_result(output_dir / "eval.envelope.json", envelope)
            result["artifacts"]["envelope"] = str(envelope_path)
            if args.card:
                append_evaluation_record(
                    Path(args.card),
                    {
                        "run_id": run_id,
                        "model": model_id,
                        "split": testset.split,
                        "checkpoint": str(checkpoint),
                        "report": str(envelope_path),
                        "metrics": {
                            "accuracy": metrics["frame"]["accuracy"],
                            "edit": metrics["segment"]["edit"],
                            "f1@0.5": metrics["segment"]["f1_at_iou"]["0.50"],
                        },
                    },
                )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
