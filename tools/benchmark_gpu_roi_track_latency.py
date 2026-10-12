"""分别测试 GPU ROI 特征提取和 tracker update 延迟。

ROI 测试包含读图、裁剪、resize、batch 和 CNN forward；tracker 测试先完成 YOLO
预计算，再只计 ``BOTSORT.update(det, image)``。两者刻意分开，不能直接当作生产端到端延迟。
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
import torchvision.models as tv_models
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ============================ 集中参数区 ============================
DEFAULT_IMAGE_DIR = ROOT / "datasets/cleansight-yolo/group1_large/images/test"
DEFAULT_LABELS_DIR = ROOT / "datasets/cleansight-yolo/group1_large/labels/test"
DEFAULT_DATA_YAML = ROOT / "datasets/cleansight-yolo/group1_large/data.yaml"
DEFAULT_CHECKPOINT = ROOT / "runs/yolo-20260806-135408/checkpoints/group1_large_yolo11s_strong_cos/weights/best.pt"
DEFAULT_TRACKER_CONFIG = ROOT / "runs/labelstudio_tracker_controlled/configs/botsort_high_clean.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "runs/gpu_latency_bench"
DEFAULT_DEVICE = "0"
DEFAULT_ROI_SIZE = 224
DEFAULT_ROI_PADDING = 0.2
DEFAULT_ROI_BATCH = 64
DEFAULT_TRACK_BATCH = 16
DEFAULT_TRACK_IMAGE_SIZE = 640
DEFAULT_TRACK_CONF = 0.25
DEFAULT_TRACK_IOU = 0.55
DEFAULT_TRACK_MAX_DET = 20


@dataclass
class RoiItem:
    image_path: Path
    class_id: int
    xywhn: tuple[float, float, float, float]


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
            "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    if not query:
        return {}
    parts = [item.strip() for item in query.splitlines()[0].split(",")]
    keys = ["name", "driver_version", "memory_total_mib", "memory_used_mib", "gpu_util_percent", "power_draw_w"]
    return dict(zip(keys, parts))


def environment_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": getattr(__import__("torchvision"), "__version__", "unknown"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_capability": ".".join(map(str, torch.cuda.get_device_capability(0))) if torch.cuda.is_available() else None,
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_smi": nvidia_smi(),
    }


def data_names(data_yaml: Path) -> dict[int, str]:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = payload.get("names") or {}
    return {int(k): str(v) for k, v in dict(names).items()}


def list_images(image_dir: Path, limit: int = 0) -> list[Path]:
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    return images[:limit] if limit and limit > 0 else images


def load_roi_items(image_dir: Path, labels_dir: Path, limit_frames: int = 0) -> tuple[list[RoiItem], int, dict[int, int]]:
    images = list_images(image_dir, limit_frames)
    items: list[RoiItem] = []
    per_class: dict[int, int] = {}
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            xywhn = tuple(float(v) for v in parts[1:5])
            items.append(RoiItem(image_path=image_path, class_id=class_id, xywhn=xywhn))  # type: ignore[arg-type]
            per_class[class_id] = per_class.get(class_id, 0) + 1
    return items, len(images), per_class


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def crop_roi_bgr(image: np.ndarray, xywhn: tuple[float, float, float, float], roi_size: int, padding: float) -> np.ndarray | None:
    h, w = image.shape[:2]
    cx, cy, bw, bh = xywhn
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    pad_w = int((x2 - x1) * padding)
    pad_h = int((y2 - y1) * padding)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = image[y1:y2, x1:x2]
    return cv2.resize(roi, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)


def make_resnet50_backbone(device: torch.device, fp16: bool) -> torch.nn.Module:
    model = tv_models.resnet50(weights=None)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    if fp16:
        model.half()
    return model


def benchmark_roi(args: argparse.Namespace, labels: dict[int, str], device: torch.device) -> dict[str, Any]:
    items, frame_count, per_class = load_roi_items(args.image_dir, args.labels_dir, args.limit_frames)
    if not items:
        raise RuntimeError("没有找到可用于 ROI 测试的标注框")

    backbone = make_resnet50_backbone(device, args.roi_fp16)
    batch_size = args.roi_batch
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    if args.roi_fp16:
        mean = mean.half()
        std = std.half()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    warmup_rois = []
    cache: dict[Path, np.ndarray] = {}
    for item in items[: min(len(items), batch_size)]:
        image = cache.get(item.image_path)
        if image is None:
            image = imread_unicode(item.image_path)
            if image is None:
                continue
            cache[item.image_path] = image
        roi = crop_roi_bgr(image, item.xywhn, args.roi_size, args.roi_padding)
        if roi is not None:
            warmup_rois.append(roi)
    if warmup_rois:
        tensor = torch.from_numpy(np.stack(warmup_rois)).to(device, non_blocking=False)
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
        tensor = tensor.half() if args.roi_fp16 else tensor.float()
        tensor = tensor / 255.0
        tensor = (tensor - mean) / std
        with torch.inference_mode():
            _ = backbone(tensor)
        torch.cuda.synchronize(device)

    batch_wall_times: list[float] = []
    batch_gpu_times: list[float] = []
    preprocess_time = 0.0
    forward_time_wall = 0.0
    total_features = 0
    feature_dim = 0
    start_total = time.perf_counter()
    batch: list[np.ndarray] = []
    cache.clear()

    def flush_batch() -> None:
        nonlocal forward_time_wall, total_features, feature_dim, batch
        if not batch:
            return
        gpu_start = torch.cuda.Event(enable_timing=True)
        gpu_end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        arr = np.stack(batch)
        tensor = torch.from_numpy(arr).to(device, non_blocking=False)
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
        tensor = tensor.half() if args.roi_fp16 else tensor.float()
        tensor = tensor / 255.0
        tensor = (tensor - mean) / std
        gpu_start.record()
        with torch.inference_mode():
            features = backbone(tensor)
        gpu_end.record()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - wall_start
        gpu_ms = gpu_start.elapsed_time(gpu_end)
        forward_time_wall += elapsed
        batch_wall_times.append(elapsed)
        batch_gpu_times.append(gpu_ms / 1000.0)
        total_features += int(features.shape[0])
        feature_dim = int(features.shape[1]) if features.ndim == 2 else int(np.prod(features.shape[1:]))
        batch = []

    last_image_path: Path | None = None
    image = None
    invalid_rois = 0
    for item in items:
        if item.image_path != last_image_path:
            image = imread_unicode(item.image_path)
            last_image_path = item.image_path
        if image is None:
            invalid_rois += 1
            continue
        t0 = time.perf_counter()
        roi = crop_roi_bgr(image, item.xywhn, args.roi_size, args.roi_padding)
        preprocess_time += time.perf_counter() - t0
        if roi is None:
            invalid_rois += 1
            continue
        batch.append(roi)
        if len(batch) >= batch_size:
            flush_batch()
    flush_batch()
    torch.cuda.synchronize(device)
    total_time = time.perf_counter() - start_total
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2

    return {
        "name": "rgb_roi_feature_extraction",
        "parameters": {
            "image_dir": str(args.image_dir),
            "labels_dir": str(args.labels_dir),
            "limit_frames": args.limit_frames,
            "roi_size": args.roi_size,
            "roi_padding": args.roi_padding,
            "roi_batch": batch_size,
            "backbone": "torchvision.resnet50(fc=Identity)",
            "weights": "None (latency only)",
            "precision": "fp16" if args.roi_fp16 else "fp32",
            "normalization": "ImageNet mean/std",
        },
        "data": {
            "frames_scanned": frame_count,
            "roi_boxes_total": len(items),
            "roi_features_extracted": total_features,
            "invalid_rois": invalid_rois,
            "feature_dim": feature_dim,
            "per_class_roi_count": {labels.get(k, str(k)): v for k, v in sorted(per_class.items())},
        },
        "timing_seconds": {
            "total": total_time,
            "cpu_crop_resize": preprocess_time,
            "gpu_h2d_normalize_forward_wall": forward_time_wall,
            "gpu_forward_events_sum": sum(batch_gpu_times),
            "per_roi_total_ms": total_time / max(total_features, 1) * 1000.0,
            "per_roi_gpu_event_ms": sum(batch_gpu_times) / max(total_features, 1) * 1000.0,
            "throughput_roi_per_sec_total": total_features / total_time if total_time else 0.0,
            "throughput_roi_per_sec_gpu_event": total_features / sum(batch_gpu_times) if sum(batch_gpu_times) else 0.0,
            "batch_wall_p50_ms": percentile(batch_wall_times, 0.5) * 1000.0,
            "batch_wall_p95_ms": percentile(batch_wall_times, 0.95) * 1000.0,
        },
        "memory_mib": {
            "torch_peak_allocated": peak_allocated,
            "torch_peak_reserved": peak_reserved,
            "nvidia_smi_after": nvidia_smi(),
        },
    }


def sequence_key(path: Path) -> str:
    stem = path.stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def load_tracker_args(path: Path):
    from ultralytics.utils import IterableSimpleNamespace

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return IterableSimpleNamespace(**payload)


def benchmark_track(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    from ultralytics import YOLO
    from ultralytics.trackers.bot_sort import BOTSORT

    image_paths = list_images(args.image_dir, args.limit_frames)
    if not image_paths:
        raise RuntimeError("没有找到可用于 track 测试的图片")

    model = YOLO(str(args.ckpt))
    tracker = BOTSORT(load_tracker_args(args.tracker_config))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    # Warm up detection only; not included in tracker timing.
    _ = model.predict(
        source=[str(p) for p in image_paths[: min(args.track_batch, len(image_paths))]],
        imgsz=args.track_imgsz,
        conf=args.track_conf,
        iou=args.track_iou,
        max_det=args.track_max_det,
        device=args.device,
        half=args.track_fp16,
        batch=args.track_batch,
        stream=False,
        verbose=False,
    )
    torch.cuda.synchronize(device)

    tracker_times: list[float] = []
    detections_per_frame: list[int] = []
    detection_precompute_time = 0.0
    tracker_total_time = 0.0
    reset_count = 0
    previous_sequence: str | None = None
    processed_frames = 0
    start_total = time.perf_counter()

    for start in range(0, len(image_paths), args.track_batch):
        batch_paths = image_paths[start : start + args.track_batch]
        detect_start = time.perf_counter()
        results = model.predict(
            source=[str(p) for p in batch_paths],
            imgsz=args.track_imgsz,
            conf=args.track_conf,
            iou=args.track_iou,
            max_det=args.track_max_det,
            device=args.device,
            half=args.track_fp16,
            batch=args.track_batch,
            stream=False,
            verbose=False,
        )
        torch.cuda.synchronize(device)
        detection_precompute_time += time.perf_counter() - detect_start

        for path, result in zip(batch_paths, results):
            key = sequence_key(path)
            if previous_sequence is not None and key != previous_sequence:
                tracker.reset()
                reset_count += 1
            previous_sequence = key
            det = result.boxes.cpu().numpy()
            detections_per_frame.append(len(det))
            t0 = time.perf_counter()
            _tracks = tracker.update(det, result.orig_img)
            elapsed = time.perf_counter() - t0
            tracker_times.append(elapsed)
            tracker_total_time += elapsed
            processed_frames += 1
    total_wall_including_detection = time.perf_counter() - start_total
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2

    return {
        "name": "yolo_detection_then_botsort_postprocess",
        "parameters": {
            "image_dir": str(args.image_dir),
            "ckpt": str(args.ckpt),
            "tracker_config": str(args.tracker_config),
            "limit_frames": args.limit_frames,
            "track_batch": args.track_batch,
            "imgsz": args.track_imgsz,
            "conf": args.track_conf,
            "nms_iou": args.track_iou,
            "max_det": args.track_max_det,
            "precision": "fp16" if args.track_fp16 else "fp32",
            "timed_scope": "BOTSORT.update(det, result.orig_img) only; YOLO detection is precomputed and reported separately",
        },
        "data": {
            "frames_total": len(image_paths),
            "frames_processed": processed_frames,
            "sequence_resets": reset_count,
            "detections_total": int(sum(detections_per_frame)),
            "detections_per_frame_mean": statistics.mean(detections_per_frame) if detections_per_frame else 0.0,
            "detections_per_frame_p95": percentile([float(v) for v in detections_per_frame], 0.95),
        },
        "timing_seconds": {
            "track_postprocess_total": tracker_total_time,
            "track_postprocess_per_frame_ms": tracker_total_time / max(processed_frames, 1) * 1000.0,
            "track_postprocess_fps": processed_frames / tracker_total_time if tracker_total_time else 0.0,
            "track_postprocess_frame_p50_ms": percentile(tracker_times, 0.5) * 1000.0,
            "track_postprocess_frame_p95_ms": percentile(tracker_times, 0.95) * 1000.0,
            "yolo_detection_precompute_not_in_track_total": detection_precompute_time,
            "wall_including_detection_for_context": total_wall_including_detection,
        },
        "memory_mib": {
            "torch_peak_allocated_while_yolo_loaded": peak_allocated,
            "torch_peak_reserved_while_yolo_loaded": peak_reserved,
            "nvidia_smi_after": nvidia_smi(),
            "note": "BoT-SORT update itself runs mainly on CPU/OpenCV in this config; GPU memory is dominated by loaded YOLO model and detection precompute.",
        },
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    env = payload["environment"]
    roi = payload["roi"]
    track = payload["track"]

    def fmt(value: Any, digits: int = 4) -> str:
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    lines = [
        "# GPU ROI 特征提取与 YOLO Track 后处理测速报告",
        "",
        "## 1. 测试环境",
        "",
        f"- 测试时间：`{payload['created_at']}`",
        f"- Python：`{env['python']}`",
        f"- PyTorch：`{env['torch']}`",
        f"- TorchVision：`{env['torchvision']}`",
        f"- CUDA 可用：`{env['cuda_available']}`",
        f"- GPU：`{env['cuda_device']}`",
        f"- CUDA capability：`{env['cuda_capability']}`",
        f"- cuDNN：`{env['cudnn']}`",
        f"- nvidia-smi：`{json.dumps(env.get('nvidia_smi', {}), ensure_ascii=False)}`",
        "",
        "## 2. 测试一：RGB 图片 ROI 特征提取",
        "",
        "### 参数",
        "",
    ]
    for key, value in roi["parameters"].items():
        lines.append(f"- `{key}`：`{value}`")
    lines += [
        "",
        "### 数据量",
        "",
        f"- 扫描图片帧数：`{roi['data']['frames_scanned']}`",
        f"- ROI 框总数：`{roi['data']['roi_boxes_total']}`",
        f"- 成功提取特征数：`{roi['data']['roi_features_extracted']}`",
        f"- 无效 ROI：`{roi['data']['invalid_rois']}`",
        f"- 特征维度：`{roi['data']['feature_dim']}`",
        f"- 逐类 ROI 数：`{json.dumps(roi['data']['per_class_roi_count'], ensure_ascii=False)}`",
        "",
        "### 耗时和显存",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 总耗时 | {fmt(roi['timing_seconds']['total'])} s |",
        f"| CPU 裁剪 + resize | {fmt(roi['timing_seconds']['cpu_crop_resize'])} s |",
        f"| H2D + normalize + GPU forward wall time | {fmt(roi['timing_seconds']['gpu_h2d_normalize_forward_wall'])} s |",
        f"| GPU forward events 累计 | {fmt(roi['timing_seconds']['gpu_forward_events_sum'])} s |",
        f"| 平均每个 ROI 总耗时 | {fmt(roi['timing_seconds']['per_roi_total_ms'])} ms |",
        f"| 平均每个 ROI GPU forward | {fmt(roi['timing_seconds']['per_roi_gpu_event_ms'])} ms |",
        f"| 总吞吐 | {fmt(roi['timing_seconds']['throughput_roi_per_sec_total'])} ROI/s |",
        f"| GPU forward 吞吐 | {fmt(roi['timing_seconds']['throughput_roi_per_sec_gpu_event'])} ROI/s |",
        f"| batch wall p50 | {fmt(roi['timing_seconds']['batch_wall_p50_ms'])} ms |",
        f"| batch wall p95 | {fmt(roi['timing_seconds']['batch_wall_p95_ms'])} ms |",
        f"| torch peak allocated | {fmt(roi['memory_mib']['torch_peak_allocated'])} MiB |",
        f"| torch peak reserved | {fmt(roi['memory_mib']['torch_peak_reserved'])} MiB |",
        "",
        "## 3. 测试二：YOLO 检测后 Track 后处理",
        "",
        "### 参数",
        "",
    ]
    for key, value in track["parameters"].items():
        lines.append(f"- `{key}`：`{value}`")
    lines += [
        "",
        "### 数据量",
        "",
        f"- 图片帧总数：`{track['data']['frames_total']}`",
        f"- 实际处理帧数：`{track['data']['frames_processed']}`",
        f"- 视频/片段切换 reset 次数：`{track['data']['sequence_resets']}`",
        f"- YOLO 检测框总数：`{track['data']['detections_total']}`",
        f"- 平均每帧检测框：`{fmt(track['data']['detections_per_frame_mean'])}`",
        f"- 每帧检测框 p95：`{fmt(track['data']['detections_per_frame_p95'])}`",
        "",
        "### 耗时和显存",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| Track 后处理总耗时 | {fmt(track['timing_seconds']['track_postprocess_total'])} s |",
        f"| Track 后处理平均每帧 | {fmt(track['timing_seconds']['track_postprocess_per_frame_ms'])} ms |",
        f"| Track 后处理吞吐 | {fmt(track['timing_seconds']['track_postprocess_fps'])} FPS |",
        f"| Track 单帧 p50 | {fmt(track['timing_seconds']['track_postprocess_frame_p50_ms'])} ms |",
        f"| Track 单帧 p95 | {fmt(track['timing_seconds']['track_postprocess_frame_p95_ms'])} ms |",
        f"| YOLO 检测预计算耗时（不计入 track） | {fmt(track['timing_seconds']['yolo_detection_precompute_not_in_track_total'])} s |",
        f"| 包含检测预计算的上下文总 wall time | {fmt(track['timing_seconds']['wall_including_detection_for_context'])} s |",
        f"| torch peak allocated while YOLO loaded | {fmt(track['memory_mib']['torch_peak_allocated_while_yolo_loaded'])} MiB |",
        f"| torch peak reserved while YOLO loaded | {fmt(track['memory_mib']['torch_peak_reserved_while_yolo_loaded'])} MiB |",
        "",
        "## 4. 说明和分析",
        "",
        "1. ROI 测试使用 ResNet50 结构输出 2048 维特征，权重为随机初始化；这是延迟测试，权重内容不影响算子耗时。若未来正式 ROI 模型使用 ResNet18、EfficientNet 或 MobileNet，需要按最终 backbone 重新测速。",
        "2. ROI 总耗时包含 OpenCV 读图、裁剪、resize、CPU 到 GPU 拷贝、归一化和 backbone forward；其中 GPU event 只覆盖 backbone forward。",
        "3. Track 测试严格只计 `BOTSORT.update(det, image)`；YOLO 检测已经完成后的 tracker 关联耗时很低。报告中的 YOLO 检测预计算耗时只作为上下文，不计入 track 后处理时间。",
        "4. 当前 `BoT-SORT high-clean` 配置 `with_reid=False`，tracker update 主要是 CPU/OpenCV/Kalman/匹配计算，GPU 显存主要由 YOLO 模型和检测预计算占用。",
        "5. Tracker 有时间状态依赖，不能像 CNN ROI forward 那样真正把多帧并行更新；这里的 batch 表示 YOLO 检测结果按批生成，tracker 后处理仍按帧顺序逐帧 update。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ROI feature extraction and tracker post-processing latency.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR)
    parser.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--tracker-config", type=Path, default=DEFAULT_TRACKER_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--limit-frames", type=int, default=0)
    parser.add_argument("--roi-size", type=int, default=DEFAULT_ROI_SIZE)
    parser.add_argument("--roi-padding", type=float, default=DEFAULT_ROI_PADDING)
    parser.add_argument("--roi-batch", type=int, default=DEFAULT_ROI_BATCH)
    parser.add_argument("--roi-fp16", action="store_true")
    parser.add_argument("--track-batch", type=int, default=DEFAULT_TRACK_BATCH)
    parser.add_argument("--track-imgsz", type=int, default=DEFAULT_TRACK_IMAGE_SIZE)
    parser.add_argument("--track-conf", type=float, default=DEFAULT_TRACK_CONF)
    parser.add_argument("--track-iou", type=float, default=DEFAULT_TRACK_IOU)
    parser.add_argument("--track-max-det", type=int, default=DEFAULT_TRACK_MAX_DET)
    parser.add_argument("--track-fp16", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("本测试要求 CUDA GPU，但当前 torch.cuda.is_available() 为 False")
    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    labels = data_names(args.data_yaml)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    created_at = time.strftime("%Y-%m-%d %H:%M:%S")
    env = environment_info()
    print("[1/2] ROI feature extraction benchmark", flush=True)
    roi = benchmark_roi(args, labels, device)
    print("[2/2] YOLO detection then BOTSORT postprocess benchmark", flush=True)
    track = benchmark_track(args, device)

    payload = {
        "created_at": created_at,
        "environment": env,
        "roi": roi,
        "track": track,
    }
    json_path = args.out_dir / "gpu_roi_track_latency_report.json"
    md_path = args.out_dir / "gpu_roi_track_latency_report.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, payload)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
