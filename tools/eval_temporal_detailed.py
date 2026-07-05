#!/usr/bin/env python3
"""Batch temporal model evaluation: per-class recall and confusion matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


VIDEO_NAMES = [
    "export1", "export2", "export3", "export4",
    "export5", "export6", "export7-480p", "export8",
    "export9", "export10", "export11", "export12",
    "export13", "export14", "export15-480P", "export16-480P",
    "export17", "export18", "export19", "export20",
]
TEST_IDX = list(range(16, 20))


def build_model(model_name: str, input_dim: int, num_classes: int):
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model", required=True, choices=["gru", "tcn", "transformer"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--input-dim", type=int, default=20)
    parser.add_argument("--num-classes", type=int, default=3)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))

    from dataloader import build_dataset
    from util import load_features, load_mappings, load_truths

    mapping_path = repo / "data" / "Endo_Project" / "mapping.txt"
    features_dir = repo / "data" / "Endo_Project" / "features"
    truths_dir = repo / "data" / "Endo_Project" / "groundTruth"
    mappings = load_mappings(str(mapping_path))
    idx_to_action = {v: k for k, v in mappings.items()}

    features, truths = [], []
    for name in VIDEO_NAMES:
        features.append(load_features(str(features_dir), name))
        truths.append(load_truths(str(truths_dir), name, mappings=mappings))

    test_dataset = build_dataset(features, truths, idx=TEST_IDX, window=args.window)
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, args.input_dim, args.num_classes).to(device)
    model.load_state_dict(torch.load(repo / args.checkpoint, map_location=device))
    model.eval()

    all_preds = []
    all_gts = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            pred = torch.argmax(logits[:, -1, :], dim=-1).cpu().numpy()
            all_preds.append(pred)
            all_gts.append(y.numpy())

    all_preds = np.concatenate(all_preds)
    all_gts = np.concatenate(all_gts)
    confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    for gt, pred in zip(all_gts, all_preds):
        if 0 <= gt < args.num_classes and 0 <= pred < args.num_classes:
            confusion[int(gt), int(pred)] += 1

    recalls = {}
    for i in range(args.num_classes):
        denom = int(confusion[i].sum())
        label = idx_to_action.get(i, str(i))
        recalls[label] = None if denom == 0 else round(float(confusion[i, i] / denom), 4)

    result = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "num_windows": int(len(all_gts)),
        "note": "Classification metrics use batched last-frame logits, without causal_decision smoothing.",
        "labels": [idx_to_action.get(i, str(i)) for i in range(args.num_classes)],
        "per_class_recall": recalls,
        "confusion_matrix_rows_gt_cols_pred": confusion.tolist(),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
