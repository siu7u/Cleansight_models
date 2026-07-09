"""Shared temporal model training and evaluation entrypoint.

The temporal-* folders own their model class. This module owns the common
training/evaluation workflow so model selection stays in `model_manager` and
per-model `main.py` files stay thin.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Type

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


MAPPING_PATH = "data/Endo_Project/mapping.txt"
FEATURES_DIR = "data/Endo_Project/features"
TRUTHS_DIR = "data/Endo_Project/groundTruth"

VIDEO_NAMES = [
    "export1", "export2", "export3", "export4",
    "export5", "export6", "export7-480p", "export8",
    "export9", "export10", "export11", "export12",
    "export13", "export14", "export15-480P", "export16-480P",
    "export17", "export18", "export19", "export20",
]

TRAIN_IDX = list(range(0, 16))
TEST_IDX = list(range(16, 20))


def pick_device() -> torch.device:
    """Return the active temporal training/evaluation device."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device_info(args, device: torch.device) -> None:
    """Print the device selected for temporal training/evaluation."""

    print(f"[temporal] model={args.model} mode={args.mode} device={device}")
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        print(f"[temporal] cuda_device={torch.cuda.get_device_name(index)}")
    elif args.mode in ["full", "train"]:
        print("[temporal] cuda_available=False; training will run on CPU")


def build_model(model_name: str, model_class: Type[torch.nn.Module], input_dim: int, num_classes: int):
    """Construct the single temporal model owned by the current subdirectory.

    Cross-model selection belongs to `model_manager/models.yaml`; a temporal
    subdirectory should only expose its own model class.
    """

    return model_class(input_dim, num_classes)


def append_training_history(args, model_path: str | None, device: torch.device) -> None:
    """Append one temporal training run to CARD.md without overwriting history."""

    from util import get_current_timestamp

    card = Path("CARD.md")
    checkpoint = model_path or "未导出"
    entry = [
        "",
        f"### {get_current_timestamp()}",
        "",
        f"- 模型: `{args.model}`",
        f"- 训练模式: `{args.mode}`",
        f"- 数据集映射: `{MAPPING_PATH}`",
        f"- 特征目录: `{FEATURES_DIR}`",
        f"- 标签目录: `{TRUTHS_DIR}`",
        f"- 训练视频索引: `{TRAIN_IDX}`",
        f"- 测试视频索引: `{TEST_IDX}`",
        f"- 输入维度: {args.input_dim}",
        f"- 窗口长度: {args.window}",
        f"- 训练轮数: {args.epochs}",
        f"- batch size: {args.batch_size}",
        f"- 学习率: {args.lr}",
        f"- 设备: `{device}`",
        f"- 输出权重: `{checkpoint}`",
    ]
    text = card.read_text(encoding="utf-8") if card.exists() else f"# 模型卡：{args.model}\n"
    if "## 训练历史" not in text:
        text = text.rstrip() + "\n\n## 训练历史\n"
    text = text.rstrip() + "\n" + "\n".join(entry) + "\n"
    card.write_text(text, encoding="utf-8")


def train_model(
    model,
    dataloader,
    optimizer,
    criterion,
    epochs: int,
    save_prefix: str,
    device: torch.device,
    verbose: bool = False,
    auto_save: bool = False,
) -> None:
    """Train a causal temporal classifier on `[B, window, F]` windows."""

    from util import get_current_timestamp

    model.train()
    for epoch in tqdm(range(1, epochs + 1)):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)                 # [B, window, num_classes]
            last_logits = logits[:, -1, :]    # [B, num_classes]

            loss = criterion(last_logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if verbose and epoch % 5 == 0:
            print(f"Epoch {epoch:3d}  Loss {loss.item():.4f}")

        if auto_save and epoch % 5 == 0:
            os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
            torch.save(model.state_dict(), f"{save_prefix}-{epoch}-{get_current_timestamp()}.pt")


def eval_model(model, dataloader, args, model_path: str | None, device: torch.device, visualize: bool = False) -> dict:
    """Evaluate streaming-style causal predictions and return segmentation metrics."""

    from util import causal_decision, edit_score, f_score, get_current_timestamp, load_mappings, plot_temporal_results

    model.eval()
    video_preds = []
    video_gts = []

    for ds in tqdm(dataloader.dataset.datasets):
        total_frames = ds.x.shape[0]
        window = ds.w
        preds = np.zeros(total_frames, dtype=np.int64)

        idle_id = 0
        preds[: window - 1] = idle_id
        pending = None
        stable = idle_id
        count = 0

        with torch.no_grad():
            for i in range(len(ds)):
                x, _ = ds[i]
                x = x.unsqueeze(0).to(device)
                logits = model(x)            # [1, window, num_classes]
                last = logits[0, -1]
                pending, stable, count = causal_decision(last, pending, stable, count)
                preds[i + window - 1] = stable

        video_preds.append(preds)
        video_gts.append(ds.y.numpy())

    all_preds = np.concatenate(video_preds)
    all_gts = np.concatenate(video_gts)
    mappings = load_mappings(MAPPING_PATH)
    idx_to_action = {v: k for k, v in mappings.items()}
    pred_labels = [idx_to_action[p] for p in all_preds]
    gt_labels = [idx_to_action[g] for g in all_gts]

    correct = sum(p == g for p, g in zip(all_preds, all_gts))
    acc = 100.0 * correct / len(all_preds)
    edit = edit_score(pred_labels, gt_labels)

    f1_scores = {}
    for overlap in [0.1, 0.25, 0.5]:
        tp, fp, fn = f_score(pred_labels, gt_labels, overlap)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_scores[overlap] = 2.0 * precision * recall / (precision + recall + 1e-8) * 100

    result = {
        "timestamp": get_current_timestamp(),
        "model": args.model,
        "path": model_path,
        "num_params": sum(p.numel() for p in model.parameters()),
        "acc": round(acc, 2),
        "edit": round(edit, 2),
        "f1": {k: round(v, 2) for k, v in f1_scores.items()},
    }
    print(json.dumps(result, indent=4, ensure_ascii=False))

    if visualize and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        plot_temporal_results(
            pred_labels,
            gt_labels,
            metadata=result,
            output_path=f"{args.output_dir}/{args.model}-{get_current_timestamp()}.png",
        )
    return result


def parse_args(model_name: str):
    """Parse the shared temporal CLI while limiting model choices to one model."""

    parser = argparse.ArgumentParser(description="Causal Action Recognition")
    parser.add_argument("--mode", type=str, default="full", help="模式选择", choices=["full", "train", "eval"])
    parser.add_argument("--model", type=str, default=model_name, help="模型类型", choices=[model_name])
    parser.add_argument("--input_dim", type=int, default=20, help="特征维度")
    parser.add_argument("--num_classes", type=int, default=3, help="类别数")
    parser.add_argument("--resume", type=str, default=None, help="从指定检查点恢复训练")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--window", type=int, default=64, help="历史帧数")
    parser.add_argument("--verbose", action="store_true", help="是否打印训练损失")
    parser.add_argument("--auto_save", action="store_true", help="是否保存中间结果")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="检查点保存目录")
    parser.add_argument("--export_dir", type=str, default="weights", help="模型权重保存目录")
    parser.add_argument("--visualize", action="store_true", help="是否生成可视化结果")
    parser.add_argument("--output_dir", type=str, default="results", help="图标结果目录")
    return parser.parse_args()


def run_temporal_main(model_name: str, model_class: Type[torch.nn.Module]) -> None:
    """Run the common temporal train/eval workflow for one model directory."""

    from dataloader import build_dataset
    from util import compute_class_weights, get_current_timestamp, load_features, load_mappings, load_truths

    args = parse_args(model_name)
    device = pick_device()
    print_device_info(args, device)
    model = build_model(args.model, model_class, args.input_dim, args.num_classes)

    model_path = None
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        model_path = args.resume
    model = model.to(device)

    mappings = load_mappings(MAPPING_PATH)
    features, truths = [], []
    for name in VIDEO_NAMES:
        features.append(load_features(FEATURES_DIR, name))
        truths.append(load_truths(TRUTHS_DIR, name, mappings=mappings))

    train_dataset = build_dataset(features, truths, idx=TRAIN_IDX, window=args.window)
    test_dataset = build_dataset(features, truths, idx=TEST_IDX, window=args.window)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    if args.mode in ["full", "train"]:
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
        train_weights = compute_class_weights(train_loader)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([train_weights[i] for i in sorted(train_weights.keys())], dtype=torch.float32).to(device)
        )
        train_model(
            model,
            train_loader,
            optimizer,
            criterion,
            args.epochs,
            save_prefix=f"{args.save_dir}/{args.model}-w{args.window}",
            device=device,
            verbose=args.verbose,
            auto_save=args.auto_save,
        )

        if args.export_dir:
            os.makedirs(args.export_dir, exist_ok=True)
            model_path = f"{args.export_dir}/{args.model}-final-{get_current_timestamp()}.pt"
            torch.save(model.state_dict(), model_path)
        append_training_history(args, model_path, device)

    if args.mode in ["full", "eval"]:
        eval_model(model, test_loader, args=args, model_path=model_path, device=device, visualize=args.visualize)
