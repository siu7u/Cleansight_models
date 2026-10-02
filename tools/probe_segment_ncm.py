#!/usr/bin/env python3
"""S-NCM 段级重标注探针：用表示空间的类均值对**每个已预测段**重新定标签。

**动机（本项目实测）**：第 3 轮诊断发现，新配方下模型已经"形成动作性证据"（真值动作帧上
`short_brush_cleaning` 的 `P(真值类)` 从 0.005 升到 0.182），但**类别判定仍是短板**——
`flush` 的段被判成"某个动作"却认错类。逐帧分类器有偏，而表示空间未必有偏。

**机制（Cost-Sensitive Learning for Long-Tailed TAS, BMVC 2024 / arXiv 2503.18358）**：
1. 用分类器的预测 ŷ 定**段边界**；
2. 用**逐帧最近类均值（NCM）**对每个段做**多数投票**定标签。
论文同表里 class-level 重加权（CB/LA/Focal/τ-norm）增益 ≈0，而该法在 MS-TCN 上拿到
+8.1 F1@25 / +3.7 Edit。**训练无关、零参数**，只用到模型的 penultimate 表示。

**本项目实测（5 run，逐 (run,视频) 配对 Wilcoxon）**：`F1@0.1` 在 **4/4** 个 `mstcn2` 臂上显著提升
（旗舰 +3.4 p=0.0020；h64 +7.4 p=0.0000；h32 +5.9 p=0.0132；h16 +8.4 p=0.0000），
`F1@0.25` 在 h16/h64 上亦显著。详见 `docs/experiments/EXPERIMENT_REPORT_SEGMENT_NCM_20260927.md`。

**分层**：本工具只做"参数解析 + 指标 + 渲染"；"读 checkpoint + 前向 + 取表示 + S-NCM 解码"
全部走 [`framework/cleansight_eval/temporal/inference.py`](../framework/cleansight_eval/temporal/inference.py)
——`tools/` 不得直接调用模型库（见 `tests/test_architecture_boundaries.py`）。

用法：

    python tools/probe_segment_ncm.py --run "runs/arch-survey/r2-h32s2l5-cw003-cpu/h32/mstcn2-*"
    python tools/probe_segment_ncm.py --run "<globA>" --label A --run "<globB>" --label B   # 两臂配对
    python tools/probe_segment_ncm.py --run "<glob>" --self-check   # 关闭 NCM，应逐帧等于 argmax
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # 仓库根：import framework


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="S-NCM 段级重标注探针（训练无关、零参数）")
    parser.add_argument("--run", action="append", required=True, help="run 目录 glob（可多次）")
    parser.add_argument("--label", action="append", default=None, help="与 --run 一一对应的显示名")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--self-check", action="store_true",
                        help="自检：跳过 NCM 重标注，三行应逐帧完全相同")
    parser.add_argument("--json", default=None, help="结构化结果输出路径")
    return parser.parse_args(argv)


def per_video(truth: np.ndarray, pred: np.ndarray, num_classes: int) -> dict:
    """逐视频三项：段覆盖 ≥50%、段标签正确率、F1@0.1 / F1@0.25（官方口径）。"""

    from framework.cleansight_eval.core.metrics import temporal_metrics

    starts = np.concatenate([[0], np.where(np.diff(truth) != 0)[0] + 1, [len(truth)]])
    covered = correct = total = 0
    for s, e in zip(starts[:-1], starts[1:]):
        if truth[s] == 0:
            continue
        total += 1
        covered += int((pred[s:e] != 0).mean() >= 0.50)
        correct += int(Counter(pred[s:e].tolist()).most_common(1)[0][0] == truth[s])
    metrics = temporal_metrics({"0": pred.tolist()}, {"0": truth.tolist()}, list(range(num_classes)))
    return {
        "n_action_segments": total,
        "coverage": covered / total if total else None,
        "segment_label_accuracy": correct / total if total else None,
        "f1_01": 100 * metrics["segment"]["f1_at_iou"]["0.10"],
        "f1_025": 100 * metrics["segment"]["f1_at_iou"]["0.25"],
        "edit": 100 * metrics["segment"]["edit"],
        "nonidle_frames": int((pred != 0).sum()),
    }


def evaluate_run(run_dir: Path, device: str, self_check: bool) -> dict:
    """对一个 run 出 argmax / 逐帧 NCM / S-NCM 三种解码的逐视频读数。"""

    from framework.cleansight_eval.core.config import load_config
    from framework.cleansight_eval.temporal.data import load_split
    from framework.cleansight_eval.temporal.inference import (
        class_means, forward_with_hidden, load_temporal_model, ncm_predict, sncm_predict,
    )

    cfg = load_config(run_dir / "config.resolved.json")
    model_type = cfg["model"]["type"]
    num_classes = int(cfg["model"]["num_classes"])
    model, _ = load_temporal_model(cfg, str(run_dir / "checkpoints" / "best.pt"), device)
    features, truths, _ = load_split(cfg["data"], cfg["data"]["split_eval"],
                                     feature_schema=cfg.get("feature_schema"))
    logits, hiddens = forward_with_hidden(model, features, model_type)
    argmax = [lg.argmax(axis=1).astype(np.int64) for lg in logits]

    if self_check:
        variants = {"argmax": argmax, "NCM逐帧": argmax, "S-NCM": argmax}
    else:
        train_features, train_labels, _ = load_split(
            cfg["data"], cfg["data"]["split_train"], feature_schema=cfg.get("feature_schema"))
        _, train_hiddens = forward_with_hidden(model, train_features, model_type)
        centers = class_means(train_hiddens, train_labels, num_classes)
        frame_ncm = ncm_predict(hiddens, centers)
        variants = {"argmax": argmax, "NCM逐帧": frame_ncm, "S-NCM": sncm_predict(argmax, frame_ncm)}

    return {
        name: [per_video(truths[i], preds[i], num_classes) for i in range(len(truths))]
        for name, preds in variants.items()
    }


def report(label: str, rows: list[dict]) -> str:
    """渲染一个臂的 markdown 表 + 与 argmax 的逐 (run,视频) 配对检验。"""

    from scipy.stats import wilcoxon

    keys = [("coverage", "段覆盖≥50%"), ("segment_label_accuracy", "段标签正确"),
            ("f1_01", "F1@0.1"), ("f1_025", "F1@0.25"), ("edit", "edit")]
    lines = [f"### {label}（{len(rows)} run）", "",
             "| 解码 | " + " | ".join(h for _, h in keys) + " |",
             "|---|" + "---:|" * len(keys)]
    for name in ("argmax", "NCM逐帧", "S-NCM"):
        cells = []
        for key, _ in keys:
            values = [v[key] for r in rows for v in r[name] if v[key] is not None]
            cells.append(f"{100 * statistics.median(values):.1f}%" if key in
                         ("coverage", "segment_label_accuracy") else f"{statistics.median(values):.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    for name in ("NCM逐帧", "S-NCM"):
        parts = []
        for key, human in keys:
            base = [v[key] for r in rows for v in r["argmax"] if v[key] is not None]
            cand = [v[key] for r in rows for v in r[name] if v[key] is not None]
            diff = np.asarray(cand) - np.asarray(base)
            diff = diff[diff != 0]
            if len(diff) < 5:
                parts.append(f"{human}: n 不足")
                continue
            scale = 100 if key in ("coverage", "segment_label_accuracy") else 1
            unit = "pp" if scale == 100 else ""
            p_value = wilcoxon(diff).pvalue
            parts.append(f"**{human}** {scale * np.median(diff):+.1f}{unit}"
                         f"（p={p_value:.4f}{'，显著' if p_value < 0.05 else ''}）")
        lines.append(f"- {name} − argmax：" + "；".join(parts))
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    args = parse_args(argv)
    labels = args.label or args.run
    if len(labels) != len(args.run):
        raise SystemExit("--label 数量必须与 --run 一致")
    results, blocks = {}, []
    for label, pattern in zip(labels, args.run):
        run_dirs = sorted({Path(p) for p in glob.glob(pattern)})
        if not run_dirs:
            raise SystemExit(f"{label}: 没有匹配到 run（{pattern}）")
        rows = [evaluate_run(run_dir, args.device, args.self_check) for run_dir in run_dirs]
        results[label] = rows
        blocks.append(report(label, rows))
        print(f"[{label}] 完成 {len(run_dirs)} run", flush=True)
    text = "# S-NCM 段级重标注探针（训练无关、零参数）\n\n" + "\n".join(blocks)
    if args.self_check:
        text += "\n> **自检模式**：NCM 已关闭，三行必须逐帧完全相同——用于验证实现正确性。\n"
    print("\n" + text)
    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"结构化结果已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
