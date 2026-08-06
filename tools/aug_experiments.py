#!/usr/bin/env python3
"""YOLO 数据增强对比实验（CPU 友好版）。

在 WSL ext4 子数据集（~/cleansight-yolo-sub）上训练不同增强配置的
yolo11s/yolo11n，训练后做 val 评测，输出整体与逐类指标汇总。

用法:
    python tools/aug_experiments.py --group group1_large --model yolo11s \
        --presets default,strong,mosaic_off --epochs 12 --imgsz 480

增强预设定义与 detection/sweep.py 的 get_augment_params 对齐。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HOME = Path.home()
# 默认数据集/输出目录（可用 --data-dir / --runs-dir 覆盖；
# Windows 上建议直接用 E:\曦源\Cleansight_models\datasets\cleansight-yolo 全量数据）
DATASET_BASE = Path(os.environ.get("AUG_DATA_DIR", str(HOME / "cleansight-yolo-sub")))
RUNS_BASE = Path(os.environ.get("AUG_RUNS_DIR", str(HOME / "cleansight-runs")))

# 增强预设（与 sweep.get_augment_params 对齐）
AUGMENT_PRESETS = {
    # ultralytics 官方默认（mosaic 全开、fliplr 0.5、轻度 HSV/缩放）
    "default": {
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0,
        "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
        "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0, "erasing": 0.0,
    },
    # 仓库 sweep 的 strong：更强的 HSV/几何 + mixup
    "strong": {
        "hsv_h": 0.02, "hsv_s": 0.8, "hsv_v": 0.5,
        "degrees": 0.0, "translate": 0.2, "scale": 0.7, "shear": 2.0,
        "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
        "mosaic": 1.0, "mixup": 0.15, "copy_paste": 0.0, "erasing": 0.0,
    },
    # 关闭 mosaic（小数据集/局部目标场景常显著影响）
    "mosaic_off": {
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0,
        "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
        "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0, "erasing": 0.0,
    },
    # 轻度：仅水平翻转 + 微缩放，其余关闭（近似"干净"训练）
    "mild": {
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
        "degrees": 0.0, "translate": 0.0, "scale": 0.2, "shear": 0.0,
        "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
        "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0, "erasing": 0.0,
    },
    # copy_paste（稀有类增强；仅检测框时依赖 ultralytics 行为，失败则跳过）
    "copy_paste": {
        "hsv_h": 0.02, "hsv_s": 0.8, "hsv_v": 0.5,
        "degrees": 0.0, "translate": 0.2, "scale": 0.7, "shear": 2.0,
        "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5,
        "mosaic": 1.0, "mixup": 0.1, "copy_paste": 0.3, "erasing": 0.0,
    },
}


def pick_device() -> str:
    """自动选择设备：优先 cuda，其次 mps，最后 cpu。"""
    import torch

    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_absolute_data_yaml(group_dir: Path) -> Path:
    """若 data.yaml 的 path 为相对路径（'.'），原地改为绝对路径，避免 ultralytics
    按 cwd 解析出错。返回 data.yaml 路径。"""
    data_yaml = group_dir / "data.yaml"
    if not data_yaml.is_file():
        return data_yaml
    import yaml

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    path_val = cfg.get("path")
    if isinstance(path_val, str) and not Path(path_val).is_absolute():
        cfg["path"] = str(group_dir)
        data_yaml.write_text(
            yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        print(f"[data] 已将 data.yaml path 改为绝对路径: {cfg['path']}")
    return data_yaml


def run_experiment(
    group: str, model: str, preset: str, imgsz: int, epochs: int,
    batch: int, device: str, dry_run: bool = False,
) -> dict:
    group_dir = DATASET_BASE / group
    data_yaml = ensure_absolute_data_yaml(group_dir)
    if not data_yaml.is_file():
        return {"error": f"数据集缺失: {data_yaml}", "group": group,
                "model": model, "preset": preset, "imgsz": imgsz,
                "epochs": epochs, "batch": batch, "name": f"{model}-{preset}"}

    name = f"{model}-{preset}-{imgsz}-{epochs}e-{datetime.now().strftime('%H%M%S')}"
    result = {"group": group, "model": model, "preset": preset,
              "imgsz": imgsz, "epochs": epochs, "batch": batch,
              "name": name, "augment": AUGMENT_PRESETS[preset]}
    print(f"\n{'='*70}\n实验: {name}\n  model={model} imgsz={imgsz} epochs={epochs} batch={batch} augment={preset}\n{'='*70}", flush=True)

    if dry_run:
        return result

    t0 = time.time()
    from ultralytics import YOLO
    model_obj = YOLO(f"{model}.pt")
    kwargs = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(RUNS_BASE),
        name=name,
        exist_ok=True,
        workers=12 if device == "cpu" else 8,
        seed=42,
        **AUGMENT_PRESETS[preset],
    )
    model_obj.train(**kwargs)
    result["train_seconds"] = round(time.time() - t0, 1)

    best = Path(model_obj.trainer.best)
    result["best_pt"] = str(best)

    # val 评测（与 sweep/val 口径一致）
    t1 = time.time()
    m = model_obj.val(data=str(data_yaml), split="val", imgsz=imgsz,
                      device=device, conf=0.001, iou=0.7, max_det=300,
                      agnostic_nms=False, verbose=False)
    box = m.box
    names = {int(k): v for k, v in dict(model_obj.names).items()}
    per_class = {}
    for i, cidx in enumerate(list(box.ap_class_index)):
        per_class[names[int(cidx)]] = {
            "precision": round(float(box.p[i]), 4),
            "recall": round(float(box.r[i]), 4),
            "map50": round(float(box.ap50[i]), 4),
        }
    result["val"] = {
        "map50": round(float(box.map50), 4),
        "map50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class": per_class,
        "val_seconds": round(time.time() - t1, 1),
    }
    print(f"  [val] mAP50={result['val']['map50']:.4f} mAP50-95={result['val']['map50_95']:.4f} "
          f"P={result['val']['precision']:.4f} R={result['val']['recall']:.4f}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["group1_large", "group2_small"])
    ap.add_argument("--model", default="yolo11s")
    ap.add_argument("--presets", default="default,strong")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None,
                    help="训练设备（0/cpu/mps）；默认自动选择 GPU")
    ap.add_argument("--data-dir", default=None,
                    help="数据集根目录（含 group1_large/group2_small 分组）")
    ap.add_argument("--runs-dir", default=None,
                    help="训练输出根目录（默认 ~/cleansight-runs）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.data_dir:
        global DATASET_BASE
        DATASET_BASE = Path(args.data_dir)
    if args.runs_dir:
        global RUNS_BASE
        RUNS_BASE = Path(args.runs_dir)

    device = args.device or pick_device()
    print(f"[device] {device}  (GPU 可用: {device != 'cpu'})")

    results = []
    for preset in args.presets.split(","):
        preset = preset.strip()
        if preset not in AUGMENT_PRESETS:
            print(f"未知增强预设: {preset}", file=sys.stderr)
            continue
        r = run_experiment(args.group, args.model, preset, args.imgsz,
                           args.epochs, args.batch, device, dry_run=args.dry_run)
        results.append(r)

    print(f"\n{'='*80}\n汇总\n{'='*80}")
    print(f"{'预设':<14} {'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8} {'训练耗时'}")
    for r in results:
        if "error" in r:
            print(f"{r['preset']:<14} ERROR: {r['error'][:50]}")
        elif "val" in r:
            v = r["val"]
            print(f"{r['preset']:<14} {v['map50']:>8.4f} {v['map50_95']:>10.4f} "
                  f"{v['precision']:>8.4f} {v['recall']:>8.4f} {r.get('train_seconds', 0)/60:.1f}min")

    out = RUNS_BASE / f"aug_compare_{args.group}_{args.model}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    RUNS_BASE.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
