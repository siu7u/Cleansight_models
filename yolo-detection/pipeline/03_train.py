#!/usr/bin/env python3
"""
按组训练 YOLO 目标检测(ultralytics)。各组一套权重,落在 runs/<组>/weights/best.pt。
训练完成后会额外复制一份 best.pt 到 versioned_weights/<模型名>-v<版本>/best.pt。
设备自动选 MPS(Apple) / CUDA / CPU。超参在 config.yaml 的 train: 段。

需 torch + ultralytics —— 用本项目 .venv/bin/python 跑(见 requirements.txt)。

用法(在 yolo_pipeline/ 下执行):
    <py> 03_train.py                         # 全部组,自动导出到下一版 yolo-*-vN
    <py> 03_train.py --version v2            # 全部组,导出到 yolo-*-v2
    <py> 03_train.py group2_small --version 2
"""
import argparse
import re
import shutil
from pathlib import Path

from utils.card import YoloCardWriter, YoloTrainingRecord
from utils.common import ROOT, load_config

DATASETS = ROOT / "datasets"
RUNS = ROOT / "runs"
DEFAULT_VERSIONED_ROOT = ROOT / "versioned_weights"
EXPORT_MODEL_NAMES = {
    "group1_large": "yolo-large",
    "group2_small": "yolo-small",
}


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def print_device_info(device: str) -> None:
    """Print the device selected for YOLO training."""

    print(f"[yolo] device={device}")
    if device == "0":
        import torch

        print(f"[yolo] cuda_device={torch.cuda.get_device_name(0)}")
    elif device == "cpu":
        print("[yolo] cuda_available=False; training will run on CPU")


def version_suffix(version: str) -> str:
    """将用户输入的版本号规范为目录后缀,例如 `0.1` -> `v0.1`。"""

    cleaned = version.strip().replace(" ", "-")
    if cleaned.lower().startswith("ver-"):
        cleaned = cleaned[4:]
    if cleaned.lower().startswith("version-"):
        cleaned = cleaned[8:]
    return cleaned if cleaned.startswith("v") else f"v{cleaned}"


def export_model_name(group: str) -> str:
    """返回面向交付的模型目录名前缀。"""

    return EXPORT_MODEL_NAMES.get(group, f"yolo-{group.replace('_', '-')}")


def next_version(groups: list[str], export_root: Path) -> str:
    """根据已存在的 yolo-*-vN 目录计算本次训练的下一版。"""

    latest = 0
    for group in groups:
        prefix = f"{export_model_name(group)}-v"
        if not export_root.exists():
            continue
        for path in export_root.iterdir():
            if not path.is_dir() or not path.name.startswith(prefix):
                continue
            match = re.fullmatch(rf"{re.escape(prefix)}(\d+)", path.name)
            if match:
                latest = max(latest, int(match.group(1)))
    return f"v{latest + 1}"


def copy_versioned_weight(group: str, best: Path, version: str, export_root: Path) -> Path:
    """将训练得到的 best.pt 复制到版本目录,用于区分多次训练产物。"""

    if not best.exists():
        raise FileNotFoundError(f"训练完成但未找到权重: {best}")
    out_dir = export_root / f"{export_model_name(group)}-{version_suffix(version)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "best.pt"
    shutil.copy2(best, out)
    return out


def parse_args():
    """解析训练分组和版本导出参数。"""

    parser = argparse.ArgumentParser(description="Train grouped YOLO detectors.")
    parser.add_argument("groups", nargs="*", help="只训练指定分组;不填则训练全部已有数据集分组")
    parser.add_argument(
        "--version",
        help="导出版本号,例如 v1 或 1;不填则自动递增为下一版",
    )
    parser.add_argument(
        "--export-root",
        default=str(DEFAULT_VERSIONED_ROOT),
        help="版本化权重导出根目录",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    cfg = load_config()
    tcfg = cfg.get("train", {})
    device = pick_device()
    print_device_info(device)
    card_writer = YoloCardWriter()
    export_root = Path(args.export_root)
    if not export_root.is_absolute():
        export_root = ROOT / export_root

    groups = args.groups or [p.name for p in sorted(DATASETS.iterdir()) if p.is_dir()]
    if not groups:
        raise SystemExit(f"datasets/ 下没有数据集组,请先跑 02_build_dataset.py: {DATASETS}")
    export_version = args.version or next_version(groups, export_root)

    results = []
    for g in groups:
        data = DATASETS / g / "data.yaml"
        if not data.exists():
            print(f"  [skip] {g}: 缺 data.yaml,先跑 02_build_dataset.py")
            continue
        print(f"\n=== 训练 {g}  (device={device}, model={tcfg.get('model')}) ===")
        model = YOLO(tcfg.get("model", "yolo11n.pt"))
        model.train(
            data=str(data),
            epochs=tcfg.get("epochs", 100),
            imgsz=tcfg.get("imgsz", 640),
            batch=tcfg.get("batch", 16),
            patience=tcfg.get("patience", 20),
            device=device,
            project=str(RUNS),
            name=g,
            exist_ok=True,
        )
        best = RUNS / g / "weights" / "best.pt"
        versioned_best = copy_versioned_weight(g, best, export_version, export_root)
        print(f"  运行权重: {best}")
        print(f"  版本导出: {versioned_best}")
        card_writer.append_training_history(
            YoloTrainingRecord.from_training(g, cfg, device, best, versioned_best)
        )
        results.append((g, best, versioned_best))

    print("\n=== 完成 ===")
    for g, best, versioned_best in results:
        print(f"{g}: {best} -> {versioned_best}")
    print("下一步:04_validate.py 出验收报告")


if __name__ == "__main__":
    main()
