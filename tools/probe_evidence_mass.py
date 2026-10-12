#!/usr/bin/env python3
"""证据质量探针：模型在**真值动作帧**上到底给了目标类多少概率。

**为什么需要它**：段级指标（edit / F1@k / 段覆盖率）掉下来以后，最常见的两种归因是
「解码不行（argmax / 平滑 / 段约束）」与「特征不行」。这两者可以用一个数区分：

对每个类 c，在 **test 上真值标签恰为 c 的帧**上，统计模型输出的
``P(真值类)``、``P(idle)`` 与 ``argmax == c`` 的比例。

- 若 ``P(真值类)`` 明显低于 ``P(idle)`` 但仍在同一量级 → **判决规则问题**，解码侧
  （Viterbi / 时长约束 / 转移先验 / 语法）有空间。
- 若 ``P(真值类)`` 接近 0（例如 0.003）而 ``P(idle)`` 接近 1 → **模型的输出分布里根本没有
  这个假设**，任何解码器都无法凭空造出它——此时修解码是白费功夫，必须回到目标函数/表示。

本探针只读 checkpoint 做前向，不训练、不改任何产物。判读实例见
``docs/experiments/EXPERIMENT_REPORT_ARCH_SURVEY_20260927.md`` §3。

用法：

    python tools/probe_evidence_mass.py --run "runs/arch-survey/baseline-mstcn2-cpu/h128/mstcn2-*"
    python tools/probe_evidence_mass.py --run "<glob>" --json tmp/evidence.json
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # 仓库根：import framework（与本目录其它 run 分析工具一致）


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="真值动作帧上的输出概率质量探针")
    parser.add_argument("--run", action="append", required=True, help="run 目录 glob（可多次）")
    parser.add_argument("--split", default=None, help="评估 split，缺省取配置的 split_eval")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", default=None, help="结构化结果输出路径")
    return parser.parse_args(argv)


def probe_run(run_dir: Path, split: str | None, device: str) -> dict:
    """对单个 run 计算逐类 ``P(真值类) / P(idle) / argmax 命中率``。"""

    import torch

    from framework.cleansight_eval.core.config import load_config
    from framework.cleansight_eval.temporal.data import load_split
    from framework.cleansight_eval.temporal.full_sequence_pipeline import _load_eval_model

    cfg = load_config(run_dir / "config.resolved.json")
    num_classes = int(cfg["model"]["num_classes"])
    model, _ = _load_eval_model(cfg, str(run_dir / "checkpoints" / "best.pt"), torch.device(device))
    features, truths, id2name = load_split(
        cfg["data"], split or cfg["data"]["split_eval"], feature_schema=cfg.get("feature_schema")
    )

    per_class: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for feats, labels in zip(features, truths):
            probs = torch.softmax(
                model(torch.from_numpy(feats).float().unsqueeze(0).to(device))[0], dim=-1
            ).cpu().numpy()
            labels = np.asarray(labels)
            for cls in range(num_classes):
                mask = labels == cls
                if mask.sum() == 0:
                    continue
                entry = per_class.setdefault(
                    id2name[cls], {"frames": 0.0, "p_true_sum": 0.0, "p_idle_sum": 0.0, "hit": 0.0}
                )
                entry["frames"] += float(mask.sum())
                entry["p_true_sum"] += float(probs[mask, cls].sum())
                entry["p_idle_sum"] += float(probs[mask, 0].sum())
                entry["hit"] += float((probs[mask].argmax(axis=1) == cls).sum())
    for entry in per_class.values():
        n = entry["frames"]
        entry["P(true class)"] = entry.pop("p_true_sum") / n
        entry["P(idle)"] = entry.pop("p_idle_sum") / n
        entry["argmax hit"] = entry.pop("hit") / n
        entry["frames"] = int(n)
    return {"run": str(run_dir), "per_class": per_class}


def render(rows: list[dict]) -> str:
    """渲染 markdown：逐类中位（跨 run）。"""

    classes: list[str] = []
    for row in rows:
        for name in row["per_class"]:
            if name not in classes:
                classes.append(name)
    lines = ["# 证据质量探针：真值动作帧上的输出概率", "",
             "判读：`P(真值类)` 接近 0 而 `P(idle)` 接近 1 ⇒ 输出分布里没有该假设，**解码侧无解**；",
             "若两者同量级 ⇒ 判决规则问题，解码侧（Viterbi / 时长约束 / 语法）有空间。", "",
             "| 真值类 | 帧数 | P(真值类) | P(idle) | argmax 命中 |", "|---|---:|---:|---:|---:|"]
    for name in classes:
        entries = [r["per_class"][name] for r in rows if name in r["per_class"]]
        if not entries:
            continue
        lines.append(
            f"| {name} | {int(statistics.median(e['frames'] for e in entries))} "
            f"| {statistics.median(e['P(true class)'] for e in entries):.3f} "
            f"| {statistics.median(e['P(idle)'] for e in entries):.3f} "
            f"| {100 * statistics.median(e['argmax hit'] for e in entries):.1f}% |"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_dirs = sorted({Path(p) for pattern in args.run for p in glob.glob(pattern)})
    if not run_dirs:
        raise SystemExit(f"没有匹配到 run：{args.run}")
    rows = []
    for run_dir in run_dirs:
        if not (run_dir / "checkpoints" / "best.pt").is_file():
            print(f"跳过（无 best.pt）：{run_dir}")
            continue
        rows.append(probe_run(run_dir, args.split, args.device))
        print(f"完成 {run_dir.name}", flush=True)
    print("\n" + render(rows))
    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n结构化结果已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
