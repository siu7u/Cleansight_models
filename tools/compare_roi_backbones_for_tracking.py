"""比较不同 ROI backbone 作为 BoT-SORT ReID 外观特征的效果。

实验固定 detector、tracker 和评测集，只替换 per-detection embedding backbone，
同时记录速度、显存和 Label Studio GT 上的 tracking 指标。tracker 结果仅用于工程筛选，
不能替代动作分类/时序分割指标。
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_labelstudio_trackers import (  # noqa: E402
    Box,
    evaluate_matches,
    frame_number,
    greedy_match,
    gt_for_frame,
    load_labelstudio_tasks,
    summarize_tasks,
    task_frames,
    task_tracks,
)


@dataclass
class FrameRecord:
    task_id: int
    frame: int
    path: Path
    xyxy: np.ndarray
    xywh: np.ndarray
    conf: np.ndarray
    cls: np.ndarray


class DetectionArray:
    def __init__(self, xyxy: np.ndarray, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.xywh = np.asarray(xywh, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)

    def __len__(self) -> int:
        return int(len(self.conf))

    def __getitem__(self, index):
        return DetectionArray(self.xyxy[index], self.xywh[index], self.conf[index], self.cls[index])


BACKBONES = {
    "resnet18": {
        "input_size": 224,
        "feat_dim": 512,
        "loader": "torchvision.resnet18",
        "cached_weight": "resnet18-f37072fd.pth",
    },
    "mobilenet_v3_small": {
        "input_size": 224,
        "feat_dim": 576,
        "loader": "torchvision.mobilenet_v3_small",
        "cached_weight": "mobilenet_v3_small-047dcff4.pth",
    },
    "efficientnet_b0": {
        "input_size": 224,
        "feat_dim": 1280,
        "loader": "torchvision.efficientnet_b0",
        "cached_weight": "efficientnet_b0_rwightman-7f5810bc.pth",
    },
    "convnext_tiny": {
        "input_size": 224,
        "feat_dim": 768,
        "loader": "torchvision.convnext_tiny",
        "cached_weight": "convnext_tiny-983f1562.pth",
    },
    "dinov2_vits14": {
        "input_size": 224,
        "feat_dim": 384,
        "loader": "torch.hub facebookresearch/dinov2 dinov2_vits14",
        "cached_weight": "not bundled; attempted via torch.hub",
    },
}


# ============================ 集中参数区 ============================
DEFAULT_LABELSTUDIO = ROOT / "datasets/labelstudio-yolo-test"
DEFAULT_IMAGE_DIR = ROOT / "datasets/cleansight-yolo/group1_large/images/test"
DEFAULT_CHECKPOINT = ROOT / "runs/yolo-20260806-135408/checkpoints/group1_large_yolo11s_strong_cos/weights/best.pt"
DEFAULT_TRACKER_CONFIG = ROOT / "runs/labelstudio_tracker_controlled/configs/botsort_high_clean.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "runs/roi_backbone_tracker_compare"
DEFAULT_BACKBONES = "resnet18,mobilenet_v3_small,efficientnet_b0,convnext_tiny,dinov2_vits14"
DEFAULT_DEVICE = "0"
DEFAULT_IMAGE_SIZE = 640
DEFAULT_CONF = 0.25
DEFAULT_NMS_IOU = 0.55
DEFAULT_MAX_DET = 20
DEFAULT_IOU_MATCH = 0.5
DEFAULT_YOLO_BATCH = 16
DEFAULT_ROI_BATCH = 64
DEFAULT_ROI_PADDING = 0.2


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def run_command(args: list[str]) -> str | None:
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        return result.stdout.strip()
    except Exception:
        return None


def nvidia_smi() -> dict[str, Any]:
    query = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if not query:
        return {}
    parts = [item.strip() for item in query.splitlines()[0].split(",")]
    keys = ["name", "driver_version", "memory_total_mib", "memory_used_mib", "gpu_util_percent"]
    return dict(zip(keys, parts))


def environment_info() -> dict[str, Any]:
    import torchvision
    import ultralytics

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_capability": ".".join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else None,
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_smi_start": nvidia_smi(),
    }


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_tracker_args(path: Path, *, with_reid: bool):
    from ultralytics.utils import IterableSimpleNamespace

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    payload["with_reid"] = bool(with_reid)
    payload["model"] = "auto"
    payload["device"] = "0"
    return IterableSimpleNamespace(**payload)


def build_backbone(name: str, device: torch.device, fp16: bool) -> tuple[nn.Module, dict[str, Any]]:
    if name == "resnet18":
        weights = tv_models.ResNet18_Weights.DEFAULT
        model = tv_models.resnet18(weights=weights)
        model.fc = nn.Identity()
    elif name == "mobilenet_v3_small":
        weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT
        model = tv_models.mobilenet_v3_small(weights=weights)
        model.classifier = nn.Identity()
    elif name == "efficientnet_b0":
        weights = tv_models.EfficientNet_B0_Weights.DEFAULT
        model = tv_models.efficientnet_b0(weights=weights)
        model.classifier = nn.Identity()
    elif name == "convnext_tiny":
        weights = tv_models.ConvNeXt_Tiny_Weights.DEFAULT
        model = tv_models.convnext_tiny(weights=weights)
        model.classifier = nn.Identity()
    elif name == "dinov2_vits14":
        # May download code/weights if not already cached.
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        weights = "torch.hub:facebookresearch/dinov2:dinov2_vits14"
    else:
        raise ValueError(f"未知 ROI backbone: {name}")

    model.eval().to(device)
    if fp16:
        model.half()
    meta = dict(BACKBONES[name])
    meta["weights"] = str(weights)
    return model, meta


def crop_detection(image: np.ndarray, xyxy: np.ndarray, size: int, padding: float) -> np.ndarray | None:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy[:4]]
    bw, bh = x2 - x1, y2 - y1
    x1 = int(max(0, x1 - bw * padding))
    y1 = int(max(0, y1 - bh * padding))
    x2 = int(min(w, x2 + bw * padding))
    y2 = int(min(h, y2 + bh * padding))
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.resize(image[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_LINEAR)


def records_to_detections(record: FrameRecord) -> DetectionArray:
    return DetectionArray(record.xyxy, record.xywh, record.conf, record.cls)


def tracks_to_boxes(tracks: np.ndarray, task_id: int, frame: int) -> list[Box]:
    boxes: list[Box] = []
    if tracks is None or len(tracks) == 0:
        return boxes
    for row in np.asarray(tracks):
        if len(row) < 7:
            continue
        x1, y1, x2, y2 = [float(v) for v in row[:4]]
        track_id = int(row[4])
        conf = float(row[5])
        cls = int(row[6])
        boxes.append(Box(task_id, frame, cls, f"t{task_id}:p{track_id}", conf, (x1, y1, x2, y2)))
    return boxes


def precompute_detections(args: argparse.Namespace, tasks: list[dict]) -> tuple[list[FrameRecord], float, int]:
    from ultralytics import YOLO

    model = YOLO(str(args.ckpt))
    records: list[FrameRecord] = []
    selected = tasks[: args.max_tasks] if args.max_tasks > 0 else tasks
    frame_paths: list[tuple[int, Path]] = []
    for task in selected:
        task_id = int(task["id"])
        for path in task_frames(args.image_dir, task_id):
            frame_paths.append((task_id, path))

    start = time.perf_counter()
    for batch_start in range(0, len(frame_paths), args.yolo_batch):
        batch = frame_paths[batch_start : batch_start + args.yolo_batch]
        results = model.predict(
            source=[str(path) for _task_id, path in batch],
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.nms_iou,
            max_det=args.max_det,
            device=args.device,
            half=args.fp16,
            batch=args.yolo_batch,
            stream=False,
            verbose=False,
        )
        torch.cuda.synchronize()
        for (task_id, path), result in zip(batch, results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                xyxy = np.zeros((0, 4), dtype=np.float32)
                xywh = np.zeros((0, 4), dtype=np.float32)
                conf = np.zeros((0,), dtype=np.float32)
                cls = np.zeros((0,), dtype=np.float32)
            else:
                xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                xywh = boxes.xywh.detach().cpu().numpy().astype(np.float32)
                conf = boxes.conf.detach().cpu().numpy().astype(np.float32)
                cls = boxes.cls.detach().cpu().numpy().astype(np.float32)
            records.append(FrameRecord(task_id, frame_number(path), path, xyxy, xywh, conf, cls))
    elapsed = time.perf_counter() - start
    return records, elapsed, len(frame_paths)


def extract_features_for_records(
    records: list[FrameRecord],
    backbone_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    model, meta = build_backbone(backbone_name, device, args.fp16)
    size = int(meta["input_size"])
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    if args.fp16:
        mean = mean.half()
        std = std.half()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    features_by_record: dict[int, np.ndarray] = {}
    pending: list[np.ndarray] = []
    pending_indexes: list[tuple[int, int]] = []
    crop_resize_time = 0.0
    forward_wall = 0.0
    forward_event = 0.0
    batch_wall_times: list[float] = []
    invalid = 0
    extracted = 0

    def flush() -> None:
        nonlocal pending, pending_indexes, forward_wall, forward_event, extracted
        if not pending:
            return
        arr = np.stack(pending)
        tensor = torch.from_numpy(arr).to(device, non_blocking=False)
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
        tensor = tensor.half() if args.fp16 else tensor.float()
        tensor = tensor / 255.0
        tensor = (tensor - mean) / std
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start = time.perf_counter()
        start_event.record()
        with torch.inference_mode():
            feats = model(tensor)
        if isinstance(feats, (list, tuple)):
            feats = feats[0]
        feats = feats.flatten(1)
        feats = torch.nn.functional.normalize(feats.float(), p=2, dim=1)
        end_event.record()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        event_elapsed = start_event.elapsed_time(end_event) / 1000.0
        out = feats.detach().cpu().numpy().astype(np.float32)
        for (record_idx, det_idx), feat in zip(pending_indexes, out):
            dest = features_by_record.setdefault(record_idx, np.zeros((len(records[record_idx].conf), out.shape[1]), dtype=np.float32))
            dest[det_idx] = feat
        extracted += int(out.shape[0])
        forward_wall += elapsed
        forward_event += event_elapsed
        batch_wall_times.append(elapsed)
        pending = []
        pending_indexes = []

    start_total = time.perf_counter()
    for record_idx, record in enumerate(records):
        if len(record.conf) == 0:
            features_by_record[record_idx] = np.zeros((0, int(meta["feat_dim"])), dtype=np.float32)
            continue
        image = imread_unicode(record.path)
        if image is None:
            invalid += len(record.conf)
            continue
        for det_idx, xyxy in enumerate(record.xyxy):
            t0 = time.perf_counter()
            crop = crop_detection(image, xyxy, size, args.roi_padding)
            crop_resize_time += time.perf_counter() - t0
            if crop is None:
                invalid += 1
                continue
            pending.append(crop)
            pending_indexes.append((record_idx, det_idx))
            if len(pending) >= args.roi_batch:
                flush()
    flush()
    torch.cuda.synchronize(device)
    total = time.perf_counter() - start_total
    peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2

    # Fill missing entries after feature dimension is known.
    feat_dim = 0
    for feats in features_by_record.values():
        if feats.size:
            feat_dim = int(feats.shape[1])
            break
    feat_dim = feat_dim or int(meta["feat_dim"])
    for idx, record in enumerate(records):
        if idx not in features_by_record:
            features_by_record[idx] = np.zeros((len(record.conf), feat_dim), dtype=np.float32)
        elif len(record.conf) == 0 and features_by_record[idx].shape[1] != feat_dim:
            features_by_record[idx] = np.zeros((0, feat_dim), dtype=np.float32)

    stats = {
        "backbone": backbone_name,
        "meta": meta,
        "roi_detections": sum(len(r.conf) for r in records),
        "roi_features_extracted": extracted,
        "invalid_rois": invalid,
        "feature_dim": feat_dim,
        "timing_seconds": {
            "total": total,
            "crop_resize": crop_resize_time,
            "h2d_normalize_forward_wall": forward_wall,
            "gpu_forward_events_sum": forward_event,
            "per_roi_total_ms": total / max(extracted, 1) * 1000.0,
            "per_roi_gpu_event_ms": forward_event / max(extracted, 1) * 1000.0,
            "throughput_roi_per_sec_total": extracted / total if total else 0.0,
            "throughput_roi_per_sec_gpu_event": extracted / forward_event if forward_event else 0.0,
            "batch_wall_p50_ms": percentile(batch_wall_times, 0.5) * 1000.0,
            "batch_wall_p95_ms": percentile(batch_wall_times, 0.95) * 1000.0,
        },
        "memory_mib": {
            "torch_peak_allocated": peak_alloc,
            "torch_peak_reserved": peak_reserved,
            "nvidia_smi_after": nvidia_smi(),
        },
    }
    del model
    torch.cuda.empty_cache()
    return features_by_record, stats


def evaluate_tracker_with_features(
    records: list[FrameRecord],
    tasks: list[dict],
    features_by_record: dict[int, np.ndarray] | None,
    args: argparse.Namespace,
    *,
    with_reid: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from ultralytics.trackers.bot_sort import BOTSORT

    tracker = BOTSORT(load_tracker_args(args.tracker_config, with_reid=with_reid))
    records_by_task: dict[int, list[tuple[int, FrameRecord]]] = {}
    for idx, record in enumerate(records):
        records_by_task.setdefault(record.task_id, []).append((idx, record))

    gt_all: list[Box] = []
    pred_all: list[Box] = []
    matches_by_frame: dict[tuple[int, int], list[tuple[Box, Box, float]]] = {}
    per_task: list[dict[str, Any]] = []
    update_times: list[float] = []
    detections_per_frame: list[int] = []
    start_total = time.perf_counter()

    for task in tasks[: args.max_tasks] if args.max_tasks > 0 else tasks:
        task_id = int(task["id"])
        task_records = records_by_task.get(task_id, [])
        if not task_records:
            continue
        tracks, meta = task_tracks(task, args.image_dir)
        if not tracks:
            continue
        tracker.reset()
        task_gt: list[Box] = []
        task_pred: list[Box] = []
        task_matches: dict[tuple[int, int], list[tuple[Box, Box, float]]] = {}
        for record_idx, record in task_records:
            image = imread_unicode(record.path)
            if image is None:
                continue
            det = records_to_detections(record)
            feats = features_by_record.get(record_idx) if features_by_record is not None else None
            t0 = time.perf_counter()
            tracks_out = tracker.update(det, image, feats=feats)
            update_times.append(time.perf_counter() - t0)
            detections_per_frame.append(len(record.conf))
            gt = gt_for_frame(task_id, record.frame, image.shape, tracks, float(meta["scale"]))
            pred = tracks_to_boxes(tracks_out, task_id, record.frame)
            pairs = [(gt[gi], pred[pi], value) for gi, pi, value in greedy_match(gt, pred, args.iou_match)]
            key = (task_id, record.frame)
            task_matches[key] = pairs
            matches_by_frame[key] = pairs
            task_gt.extend(gt)
            task_pred.extend(pred)
        gt_all.extend(task_gt)
        pred_all.extend(task_pred)
        metrics = evaluate_matches(task_gt, task_pred, task_matches)
        metrics.update(
            {
                "task_id": task_id,
                "frames": len(task_records),
                "duration": meta.get("duration"),
                "video": Path((task.get("data") or {}).get("video", "")).name,
            }
        )
        per_task.append(round_metrics(metrics))

    metrics = round_metrics(evaluate_matches(gt_all, pred_all, matches_by_frame))
    elapsed = time.perf_counter() - start_total
    metrics["tracker_timing_seconds"] = {
        "total_wall": elapsed,
        "update_only_sum": sum(update_times),
        "update_per_frame_ms": sum(update_times) / max(len(update_times), 1) * 1000.0,
        "update_fps": len(update_times) / sum(update_times) if sum(update_times) else 0.0,
        "update_p50_ms": percentile(update_times, 0.5) * 1000.0,
        "update_p95_ms": percentile(update_times, 0.95) * 1000.0,
        "detections_per_frame_mean": statistics.mean(detections_per_frame) if detections_per_frame else 0.0,
        "detections_per_frame_p95": percentile([float(x) for x in detections_per_frame], 0.95),
    }
    return metrics, per_task


def round_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            out[key] = round(value, 4)
        else:
            out[key] = value
    return out


def write_backbone_report(path: Path, payload: dict[str, Any]) -> None:
    env = payload["environment"]
    common = payload["common"]
    result = payload["result"]
    feature = result.get("feature_extraction")
    metrics = result.get("metrics")
    lines = [
        f"# ROI Backbone 评测：{result['backbone']}",
        "",
        "## 1. 公共设置",
        "",
        f"- GPU：`{env['cuda_device']}`",
        f"- 显存总量：`{env.get('nvidia_smi_start', {}).get('memory_total_mib', 'unknown')} MiB`",
        f"- PyTorch：`{env['torch']}`",
        f"- TorchVision：`{env['torchvision']}`",
        f"- Ultralytics：`{env['ultralytics']}`",
        f"- YOLO 权重：`{common['ckpt']}`",
        f"- 图片目录：`{common['image_dir']}`",
        f"- Label Studio：`{common['labelstudio']}`",
        f"- 评测 task 数：`{common['evaluated_task_count']}`",
        f"- 评测帧数：`{common['evaluated_frame_count']}`",
        f"- YOLO 参数：`imgsz={common['imgsz']}, conf={common['conf']}, nms_iou={common['nms_iou']}, max_det={common['max_det']}`",
        f"- ROI batch size：`{common['roi_batch']}`",
        f"- ROI padding：`{common['roi_padding']}`",
        f"- 精度：`{'fp16' if common['fp16'] else 'fp32'}`",
        "",
    ]
    if result.get("status") != "ok":
        lines += [
            "## 2. 状态",
            "",
            f"- 结果：`skipped`",
            f"- 原因：`{result.get('error')}`",
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    if feature:
        lines += [
            "## 2. Backbone 与 ROI 特征",
            "",
            "| 项目 | 数值 |",
            "| --- | --- |",
            f"| loader | `{feature['meta']['loader']}` |",
            f"| weights | `{feature['meta']['weights']}` |",
            f"| input_size | `{feature['meta']['input_size']}` |",
            f"| feature_dim | `{feature['feature_dim']}` |",
            f"| ROI detections | `{feature['roi_detections']}` |",
            f"| features extracted | `{feature['roi_features_extracted']}` |",
            f"| invalid ROIs | `{feature['invalid_rois']}` |",
            "",
            "## 3. 速度和显存",
            "",
            "| 指标 | 数值 |",
            "| --- | ---: |",
            f"| ROI 特征提取总耗时 | {feature['timing_seconds']['total']:.4f} s |",
            f"| ROI crop + resize | {feature['timing_seconds']['crop_resize']:.4f} s |",
            f"| H2D + normalize + forward wall | {feature['timing_seconds']['h2d_normalize_forward_wall']:.4f} s |",
            f"| GPU forward event sum | {feature['timing_seconds']['gpu_forward_events_sum']:.4f} s |",
            f"| 每 ROI 总耗时 | {feature['timing_seconds']['per_roi_total_ms']:.4f} ms |",
            f"| 每 ROI GPU forward | {feature['timing_seconds']['per_roi_gpu_event_ms']:.4f} ms |",
            f"| 总吞吐 | {feature['timing_seconds']['throughput_roi_per_sec_total']:.4f} ROI/s |",
            f"| GPU forward 吞吐 | {feature['timing_seconds']['throughput_roi_per_sec_gpu_event']:.4f} ROI/s |",
            f"| batch p50 | {feature['timing_seconds']['batch_wall_p50_ms']:.4f} ms |",
            f"| batch p95 | {feature['timing_seconds']['batch_wall_p95_ms']:.4f} ms |",
            f"| torch peak allocated | {feature['memory_mib']['torch_peak_allocated']:.4f} MiB |",
            f"| torch peak reserved | {feature['memory_mib']['torch_peak_reserved']:.4f} MiB |",
            f"| tracker update only | {metrics['tracker_timing_seconds']['update_only_sum']:.4f} s |",
            f"| tracker update per frame | {metrics['tracker_timing_seconds']['update_per_frame_ms']:.4f} ms |",
            "",
        ]
    else:
        lines += [
            "## 2. 基线说明",
            "",
            "该项不使用 ROI backbone，也不向 BoT-SORT 传入外观 embedding，只作为判断 ReID 特征是否有收益的 no-ReID 对照组。",
            "",
            "## 3. 速度",
            "",
            "| 指标 | 数值 |",
            "| --- | ---: |",
            f"| tracker update only | {metrics['tracker_timing_seconds']['update_only_sum']:.4f} s |",
            f"| tracker update per frame | {metrics['tracker_timing_seconds']['update_per_frame_ms']:.4f} ms |",
            "",
        ]
    lines += [
        "## 4. Track 效果",
        "",
        "| Precision | Recall | F1 | IDF1 | ID Switches | Fragments | MT | ML | Pred Boxes |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['idf1']:.4f} | {metrics['id_switches']} | {metrics['fragments']} | {metrics['mostly_tracked']} | {metrics['mostly_lost']} | {metrics['pred_boxes']} |",
        "",
        "## 5. 说明",
        "",
        "本报告固定 YOLO 检测和 BoT-SORT 参数，只替换 ROI backbone。ROI embedding 通过 `BOTSORT.update(..., feats=features)` 接入外观关联，评测指标与之前 Label Studio GT track id 实验一致。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    env = payload["environment"]
    common = payload["common"]
    results = payload["results"]
    ok = [r for r in results if r.get("status") == "ok"]
    skipped = [r for r in results if r.get("status") != "ok"]
    baseline = next((r for r in ok if r["backbone"] == "no_reid_baseline"), None)
    best = max(ok, key=lambda r: (r["metrics"]["idf1"], r["metrics"]["f1"])) if ok else None
    fastest = min([r for r in ok if r.get("feature_extraction")], key=lambda r: r["feature_extraction"]["timing_seconds"]["per_roi_total_ms"], default=None)

    lines = [
        "# ROI Backbone 接入动作模型前的 Tracker 对比总报告",
        "",
        "## 1. 公共设置",
        "",
        f"- 测试时间：`{payload['created_at']}`",
        f"- GPU：`{env['cuda_device']}`",
        f"- 显存总量：`{env.get('nvidia_smi_start', {}).get('memory_total_mib', 'unknown')} MiB`",
        f"- PyTorch：`{env['torch']}`",
        f"- TorchVision：`{env['torchvision']}`",
        f"- Ultralytics：`{env['ultralytics']}`",
        f"- YOLO 权重：`{common['ckpt']}`",
        f"- 图片目录：`{common['image_dir']}`",
        f"- Label Studio GT：`{common['labelstudio']}`",
        f"- 评测 task 数：`{common['evaluated_task_count']}`",
        f"- 评测帧数：`{common['evaluated_frame_count']}`",
        f"- YOLO batch size：`{common['yolo_batch']}`",
        f"- ROI batch size：`{common['roi_batch']}`",
        f"- 图片输入尺寸：`imgsz={common['imgsz']}`",
        f"- 检测参数：`conf={common['conf']}, nms_iou={common['nms_iou']}, max_det={common['max_det']}`",
        f"- Tracker：`BoT-SORT high-clean`",
        f"- 评测 IoU match：`{common['iou_match']}`",
        f"- 精度：`{'fp16' if common['fp16'] else 'fp32'}`",
        f"- YOLO 检测预计算耗时：`{payload['detection_precompute_seconds']:.4f} s`",
        "",
        "## 2. 综合对比表",
        "",
        "| Backbone | Feature Dim | ROI/s | ms/ROI | GPU ms/ROI | Peak Alloc MiB | Precision | Recall | F1 | IDF1 | ID Switches | Fragments | Update ms/frame | 状态 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        if result.get("status") != "ok":
            lines.append(f"| `{result['backbone']}` | - | - | - | - | - | - | - | - | - | - | - | - | skipped: {result.get('error', '')} |")
            continue
        metrics = result["metrics"]
        feat = result.get("feature_extraction")
        if feat:
            lines.append(
                f"| `{result['backbone']}` | {feat['feature_dim']} | {feat['timing_seconds']['throughput_roi_per_sec_total']:.2f} | "
                f"{feat['timing_seconds']['per_roi_total_ms']:.4f} | {feat['timing_seconds']['per_roi_gpu_event_ms']:.4f} | "
                f"{feat['memory_mib']['torch_peak_allocated']:.2f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                f"{metrics['f1']:.4f} | {metrics['idf1']:.4f} | {metrics['id_switches']} | {metrics['fragments']} | "
                f"{metrics['tracker_timing_seconds']['update_per_frame_ms']:.4f} | ok |"
            )
        else:
            lines.append(
                f"| `{result['backbone']}` | - | - | - | - | - | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                f"{metrics['f1']:.4f} | {metrics['idf1']:.4f} | {metrics['id_switches']} | {metrics['fragments']} | "
                f"{metrics['tracker_timing_seconds']['update_per_frame_ms']:.4f} | ok |"
            )

    lines += [
        "",
        "## 3. 显存占用对比表",
        "",
        "| Backbone | Feature Dim | Torch Peak Allocated MiB | Torch Peak Reserved MiB | nvidia-smi Used MiB | GPU Util % | 说明 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        if result.get("status") != "ok":
            lines.append(f"| `{result['backbone']}` | - | - | - | - | - | skipped: {result.get('error', '')} |")
            continue
        feat = result.get("feature_extraction")
        if not feat:
            lines.append(f"| `{result['backbone']}` | - | - | - | - | - | 无 ROI backbone，仅作为 tracker 基线 |")
            continue
        memory = feat.get("memory_mib", {})
        smi = memory.get("nvidia_smi_after") or {}
        note = "ROI backbone 显存采样"
        lines.append(
            f"| `{result['backbone']}` | {feat.get('feature_dim', '-')} | "
            f"{memory.get('torch_peak_allocated', 0.0):.2f} | {memory.get('torch_peak_reserved', 0.0):.2f} | "
            f"{smi.get('memory_used_mib', '-')} | {smi.get('gpu_util_percent', '-')} | {note} |"
        )

    lines += [
        "",
        "说明：`Torch Peak Allocated` 是 PyTorch 实际分配峰值，`Torch Peak Reserved` 是 PyTorch CUDA 缓存池保留峰值，`nvidia-smi Used` 是提取结束后进程级显存占用采样值，三者口径不同，不能简单相加。",
        "",
        "## 4. 结果分析",
        "",
    ]
    if best:
        lines.append(f"- 按 IDF1 优先排序，当前最佳是 `{best['backbone']}`，IDF1=`{best['metrics']['idf1']:.4f}`，F1=`{best['metrics']['f1']:.4f}`。")
    if baseline and best and best is not baseline:
        lines.append(
            f"- 相比 `no_reid_baseline`，最佳 backbone 的 IDF1 变化为 `{best['metrics']['idf1'] - baseline['metrics']['idf1']:+.4f}`，"
            f"ID Switches 变化为 `{best['metrics']['id_switches'] - baseline['metrics']['id_switches']:+d}`。"
        )
    if fastest:
        lines.append(f"- ROI 特征提取速度最快的是 `{fastest['backbone']}`，总吞吐约 `{fastest['feature_extraction']['timing_seconds']['throughput_roi_per_sec_total']:.2f} ROI/s`。")
    lines += [
        "- 如果某个 backbone 的 IDF1 没有超过 no-ReID 基线，说明 ImageNet/DINO 通用 ROI embedding 不能直接作为本场景的可靠 ReID 特征，需要在本数据上做对比学习或轨迹监督微调。",
        "- ROI backbone 对后续动作模型仍然有价值：即使它不提升 tracker IDF1，也可能给动作分类/时序分割提供更强的局部视觉表征。最终是否接入动作模型，还需要看动作模型指标。",
        "",
        "## 5. DINO 说明",
        "",
        "`DINOv2 ViT-S/14` 属于自监督 ViT 特征，理论上适合作为 ROI embedding，但本地没有预置权重或依赖时会尝试通过 torch hub 获取。若本轮报告显示 skipped，需要先准备本地 DINOv2 权重后再复测。",
        "",
        "## 6. 单独报告",
        "",
    ]
    for result in results:
        lines.append(f"- `{result['backbone']}`：`{result.get('report_path', '')}`")
    if skipped:
        lines += ["", "## 7. 跳过项", ""]
        for result in skipped:
            lines.append(f"- `{result['backbone']}`：{result.get('error')}")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ROI backbone embeddings for BoT-SORT tracking.")
    parser.add_argument("--labelstudio", type=Path, default=DEFAULT_LABELSTUDIO)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tracker-config", type=Path, default=DEFAULT_TRACKER_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backbones", default=DEFAULT_BACKBONES)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU)
    parser.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET)
    parser.add_argument("--iou-match", type=float, default=DEFAULT_IOU_MATCH)
    parser.add_argument("--yolo-batch", type=int, default=DEFAULT_YOLO_BATCH)
    parser.add_argument("--roi-batch", type=int, default=DEFAULT_ROI_BATCH)
    parser.add_argument("--roi-padding", type=float, default=DEFAULT_ROI_PADDING)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("需要本地 CUDA GPU 才能完成该测试")
    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    env = environment_info()
    tasks_all = load_labelstudio_tasks(args.labelstudio)
    task_summary = summarize_tasks(tasks_all, args.image_dir)
    matched_ids = {item["task_id"] for item in task_summary}
    tasks = [task for task in tasks_all if int(task["id"]) in matched_ids]
    selected_tasks = tasks[: args.max_tasks] if args.max_tasks > 0 else tasks

    print("[1/3] Precompute YOLO detections", flush=True)
    records, detection_time, frame_count = precompute_detections(args, selected_tasks)
    common = {
        "labelstudio": str(args.labelstudio),
        "image_dir": str(args.image_dir),
        "ckpt": str(args.ckpt),
        "tracker_config": str(args.tracker_config),
        "evaluated_task_count": len(selected_tasks),
        "evaluated_frame_count": frame_count,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "nms_iou": args.nms_iou,
        "max_det": args.max_det,
        "iou_match": args.iou_match,
        "yolo_batch": args.yolo_batch,
        "roi_batch": args.roi_batch,
        "roi_padding": args.roi_padding,
        "fp16": args.fp16,
    }

    results: list[dict[str, Any]] = []
    print("[2/3] Evaluate no-ReID baseline", flush=True)
    metrics, per_task = evaluate_tracker_with_features(records, selected_tasks, None, args, with_reid=False)
    baseline = {"backbone": "no_reid_baseline", "status": "ok", "metrics": metrics, "per_task": per_task}
    baseline["report_path"] = str(args.out_dir / "no_reid_baseline_report.md")
    results.append(baseline)
    write_backbone_report(
        args.out_dir / "no_reid_baseline_report.md",
        {
            "created_at": created_at,
            "environment": env,
            "common": common,
            "result": baseline,
        },
    )

    backbone_names = [name.strip() for name in args.backbones.split(",") if name.strip()]
    print("[3/3] Evaluate ROI backbones", flush=True)
    for idx, name in enumerate(backbone_names, start=1):
        print(f"  [{idx}/{len(backbone_names)}] {name}", flush=True)
        result: dict[str, Any] = {"backbone": name}
        try:
            features, feature_stats = extract_features_for_records(records, name, args, device)
            metrics, per_task = evaluate_tracker_with_features(records, selected_tasks, features, args, with_reid=True)
            result.update({"status": "ok", "feature_extraction": feature_stats, "metrics": metrics, "per_task": per_task})
        except Exception as exc:
            result.update({"status": "skipped", "error": repr(exc)})
        report_path = args.out_dir / f"{name}_report.md"
        result["report_path"] = str(report_path)
        results.append(result)
        partial = {
            "created_at": created_at,
            "environment": env,
            "common": common,
            "result": result,
        }
        write_backbone_report(report_path, partial)
        torch.cuda.empty_cache()

    payload = {
        "created_at": created_at,
        "environment": env,
        "common": common,
        "detection_precompute_seconds": detection_time,
        "results": results,
    }
    json_path = args.out_dir / "roi_backbone_tracker_compare.json"
    summary_path = args.out_dir / "ROI_BACKBONE_TRACKER_COMPARE_SUMMARY.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(summary_path, payload)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
