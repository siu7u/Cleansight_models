#!/usr/bin/env python3
"""约束感知解码探针：把"逐帧 argmax"换成带结构先验的 Viterbi，训练无关、零参数。

**动机（本项目实测）**：旗舰配方（`mstcn2` s4l10h128 + `roi-grid-144`）在 test 上的主要失败不是
边界抖动，而是**真值动作段几乎没有被覆盖**——以真值动作段为单位，段内被预测为任一非 idle 类的
帧比例**中位数只有 1.4%**（56.1% 的段 ≤10%）；典型 trace 是"在段位置打 2~4 帧的短脉冲后立刻回到
idle"。逐帧 argmax 下，只要 idle 的概率略高就会立刻切回 idle，**没有任何机制要求"一个动作段应该
持续多久"**。复现见 `docs/experiments/EXPERIMENT_REPORT_ARCH_SURVEY_20260927.md` §1.2。

**口径（文献）**：约束感知 Viterbi（CAD, ICPR 2026, arXiv 2605.10149）在逐帧对数概率上叠加
三类**统计结构先验**——转移置信度、合法起止集合、按类时长上下界——再跑 Viterbi；论文消融显示
增益主要来自约束本身（纯 Viterbi edit 39.2 → 约束版 48.9）。本探针实现其中**可判读性最强、
估计最稳**的两项：转移先验与类时长下界（起止集合在 6 类、流程高度有序的数据上等价于转移先验）。

**本工具只读 run 产物 + checkpoint，不训练、不写模型**；指标走与正式评测同一个
`framework.cleansight_eval.core.metrics.temporal_metrics`，因此可直接与模型行比较。

用法：

    python tools/probe_constrained_decoding.py --run "runs/round21-sel-edit-s4l10h128/h128/mstcn2-*"
    python tools/probe_constrained_decoding.py --run "<glob>" --min-duration 10 --transition-strength 0.5
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

METRIC_KEYS = ("acc", "edit", "f1@0.1", "f1@0.25", "f1@0.5")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="约束感知 Viterbi 解码探针（训练无关）")
    parser.add_argument("--run", action="append", required=True, help="run 目录 glob（可多次）")
    parser.add_argument("--split", default=None, help="评估 split，缺省取配置的 split_eval")
    parser.add_argument("--transition-strength", type=float, default=0.0,
                        help="转移先验权重 α。**默认 0**：文献消融显示只加转移先验的朴素 Viterbi "
                             "反而更差（39.2 vs 约束版 48.9 Edit），增益来自约束而非转移先验")
    parser.add_argument("--duration-quantile", type=float, default=0.10,
                        help="从 train 真值段估计类最短时长的分位数（硬约束下界）")
    parser.add_argument("--no-duration", action="store_true",
                        help="关闭时长约束（此时 α=0 即等价逐帧 argmax，用作自检）")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", default=None, help="把逐 run 结构化结果写到此路径")
    return parser.parse_args(argv)


def load_run_config(run_dir: Path) -> dict:
    """读取 run 的 resolved 配置（含 data.root / feature_schema / 模型超参）。"""

    from framework.cleansight_eval.core.config import load_config

    return load_config(run_dir / "config.resolved.json")


def collect_logits(cfg: dict, ckpt: Path, split: str, device: str):
    """按 checkpoint 重建模型，对指定 split 逐视频整段前向，返回 (logits 列表, 真值列表, id2name)。

    ``logits`` 每个为 ``[T, C]`` float32（**未过 softmax**，Viterbi 内部做 log_softmax）。
    """

    import torch

    from framework.cleansight_eval.temporal.data import load_split
    from framework.cleansight_eval.temporal.full_sequence_pipeline import _load_eval_model

    model, _meta = _load_eval_model(cfg, str(ckpt), torch.device(device))
    features, truths, id2name = load_split(
        cfg["data"], split, feature_schema=cfg.get("feature_schema")
    )
    logits = []
    with torch.no_grad():
        for feats in features:
            x = torch.from_numpy(feats).float().unsqueeze(0).to(device)
            logits.append(model(x)[0].cpu().numpy().astype(np.float32))
    return logits, [np.asarray(t) for t in truths], id2name


def estimate_priors(truths: list, num_classes: int, quantile: float):
    """从**训练集**真值估计转移矩阵与逐类时长下界（只用 train，避免偷看 test）。

    返回 ``(log_transition [C,C], min_duration [C])``。未出现的转移给一个很低的平滑概率，
    未出现的类给时长下界 1。段长按极大连续同标签段统计。
    """

    counts = np.zeros((num_classes, num_classes), dtype=np.float64)
    durations: dict[int, list[int]] = {c: [] for c in range(num_classes)}
    for truth in truths:
        for a, b in zip(truth[:-1], truth[1:]):
            counts[int(a), int(b)] += 1
        change = np.concatenate([[0], np.where(np.diff(truth) != 0)[0] + 1, [len(truth)]])
        for s, e in zip(change[:-1], change[1:]):
            durations[int(truth[s])].append(int(e - s))
    # 行归一化 + 拉普拉斯平滑，再取对数（平滑避免未见转移取到 -inf）。
    smoothed = counts + 1e-3
    transition = smoothed / smoothed.sum(axis=1, keepdims=True)
    log_transition = np.log(transition)
    min_duration = np.ones(num_classes, dtype=np.float64)
    for c, lens in durations.items():
        if lens:
            min_duration[c] = max(1.0, float(np.quantile(lens, quantile)))
    return log_transition, min_duration


def viterbi_decode(logits: np.ndarray, log_transition: np.ndarray, min_duration: np.ndarray,
                   transition_strength: float, hard_duration: bool,
                   max_state_duration: int = 64) -> np.ndarray:
    """带**硬最短时长约束**（可选转移先验）的 Viterbi 解码，返回 ``[T]`` 标签序列。

    状态为 ``(类别 c, 当前段已持续帧数 d)``，``d`` 在 ``D_c = min_duration[c]`` 处饱和
    （更长的 ``d`` 在代价上等价，故截断到 ``max_state_duration`` 控制状态数）。递推：

        留在 c   : (c, d) → (c, min(d+1, D_c))          代价 0
        切到 c'  : (c, d) → (c', 1)，**仅当 d ≥ D_c**     代价 α·log A[c, c']

    ``D_c`` 由 **train** 真值段时长的低分位估计。约束是**硬**的：一旦进入类 c，必须至少持续
    ``D_c`` 帧才允许切走——于是解码器只能把 2~4 帧的短脉冲**延长**到最短时长，而不是掐掉它。
    这是逐帧 argmax 完全缺失的结构先验（CAD, ICPR 2026 / arXiv 2605.10149 的"按类时长界约束解码"）。

    两条由文献固定、且被本工具单元校验复现的纪律：

    1. **只加转移先验会变差**：论文消融里"logits + 转移矩阵"的朴素 Viterbi 只有 39.2 Edit，
       而带约束的版本是 48.9——增益主要来自**约束**而非解码本身。故 ``transition_strength``
       默认为 ``0.0``，只作为可选旋钮（否则强转移先验会把稀类脉冲直接抹平，本工具单元测试已复现）。
    2. **硬约束优于软惩罚**（论文 77.9 vs 74.6），故 ``hard_duration=True`` 为默认。

    ``α = 0 且 hard_duration=False`` 时本函数逐帧等价于 argmax（已由单元校验固定）。
    """

    from scipy.special import log_softmax

    t_len, num_classes = logits.shape
    logp = log_softmax(logits.astype(np.float64), axis=1)
    # 约束不得长于视频本身：否则任何路径都不可行。
    D = np.clip(np.round(min_duration).astype(int), 1, max(1, min(max_state_duration, t_len - 1 or 1)))
    states_per_class = int(D.max())

    switch = transition_strength * log_transition.astype(np.float64).copy()
    np.fill_diagonal(switch, -np.inf)

    # 允许"离开类 c"的最小已持续帧数（d 的下标 = d−1）。硬约束：d ≥ D_c 才可切走。
    leave_allowed = np.stack(
        [np.arange(1, states_per_class + 1) >= D[c] for c in range(num_classes)]
    )  # [C, S]

    scores = np.full((num_classes, states_per_class), -np.inf, dtype=np.float64)
    scores[:, 0] = logp[0]
    back_c = np.zeros((t_len, num_classes, states_per_class), dtype=np.int16)
    back_d = np.zeros((t_len, num_classes, states_per_class), dtype=np.int16)

    for t in range(1, t_len):
        # 留在原类：d → d+1，在饱和点保持。
        stay = np.full((num_classes, states_per_class), -np.inf, dtype=np.float64)
        stay[:, 1:] = scores[:, :-1]
        stay[:, states_per_class - 1] = np.maximum(stay[:, states_per_class - 1],
                                                   scores[:, states_per_class - 1])
        stay_src_c = np.tile(np.arange(num_classes)[:, None], (1, states_per_class))
        stay_src_d = np.tile(np.arange(states_per_class)[None, :], (num_classes, 1))
        stay_src_d = np.minimum(stay_src_d + 1, states_per_class - 1)

        # 切进 c'：源状态必须满足"已持续 ≥ D_c"。
        usable = np.where(leave_allowed, scores, -np.inf)
        best_d = np.argmax(usable, axis=1)                     # [C]
        best_val = usable[np.arange(num_classes), best_d]      # [C]
        switch_scores = best_val[:, None] + switch             # [K, C']
        src_of_target = np.argmax(switch_scores, axis=0)       # [C']
        switch_val = switch_scores[src_of_target, np.arange(num_classes)]

        take_switch = switch_val > stay[:, 0]
        new_scores = stay.copy()
        new_scores[:, 0] = np.where(take_switch, switch_val, stay[:, 0])
        back_c[t] = stay_src_c
        back_d[t] = stay_src_d
        for c_prime in range(num_classes):
            if take_switch[c_prime]:
                back_c[t, c_prime, 0] = src_of_target[c_prime]
                back_d[t, c_prime, 0] = best_d[src_of_target[c_prime]]
        scores = new_scores + logp[t][:, None]

    labels = np.zeros(t_len, dtype=np.int64)
    flat = int(np.argmax(scores))
    c, d = divmod(flat, states_per_class)
    labels[-1] = c
    for t in range(t_len - 1, 0, -1):
        prev_c = int(back_c[t, c, d])
        prev_d = int(back_d[t, c, d])
        labels[t - 1] = prev_c
        c, d = prev_c, prev_d
    return labels


def segment_coverage(truth: np.ndarray, pred: np.ndarray) -> list[float]:
    """真值动作段（非 idle 极大连续段）内被预测为任一非 idle 类的帧比例列表。"""

    change = np.concatenate([[0], np.where(np.diff(truth) != 0)[0] + 1, [len(truth)]])
    out = []
    for s, e in zip(change[:-1], change[1:]):
        if truth[s] == 0:
            continue
        out.append(float((pred[s:e] != 0).mean()))
    return out


def render(args, rows: list[dict]) -> str:
    """把逐 run 结果渲染成 markdown 汇总。"""

    lines = ["# 约束感知解码探针（训练无关）", "",
             f"- 转移先验权重 α={args.transition_strength}；类最短时长 = train 真值段时长的 "
             f"q{args.duration_quantile:.2f} 分位，**硬约束**"
             + ("（已关闭，此时等价逐帧 argmax 自检）" if args.no_duration else ""),
             f"- 先验从 **train** 真值估计，评估在 test；指标口径与正式评测一致。", ""]
    lines.append("| 解码 | n run | " + " | ".join(METRIC_KEYS) + " | 段数比 | 段覆盖率中位 |")
    lines.append("|---|---:|" + "---:|" * (len(METRIC_KEYS) + 2))
    for name in ("argmax", "viterbi"):
        subset = [r[name] for r in rows]
        if not subset:
            continue
        med = lambda key: statistics.median([r[key] for r in subset])
        lines.append(
            f"| {name} | {len(subset)} | "
            + " | ".join(f"{med(k):.2f}" for k in METRIC_KEYS)
            + f" | {med('seg_ratio'):.2f} | {100 * med('coverage'):.1f}% |"
        )
    lines.append("")
    lines.append("## 逐类帧级召回（各解码下，跨 run 中位 %）")
    lines.append("")
    all_classes = sorted({c for r in rows for c in r["argmax"]["recall"]})
    lines.append("| 解码 | " + " | ".join(all_classes) + " |")
    lines.append("|---|" + "---:|" * len(all_classes))
    for name in ("argmax", "viterbi"):
        cells = []
        for cls in all_classes:
            vals = [r[name]["recall"][cls] for r in rows if cls in r[name]["recall"]]
            cells.append("n/a" if not vals else f"{100 * statistics.median(vals):.1f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("")
    win = sum(1 for r in rows if r["viterbi"]["edit"] > r["argmax"]["edit"])
    lines.append(f"逐 run edit 提升个数：**{win}/{len(rows)}**")
    lines.append("")
    lines.append("> 判读纪律：编辑距离需与段数比、非 idle 帧数一起看；跨 seed 中位差小于噪声地板"
                 "（本配方 5 seed 下 edit ≈ 9 分）时不可判读，须走 `tools/compare_runs.py` 的配对检验。")
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    from framework.cleansight_eval.core.metrics import temporal_metrics

    run_dirs = sorted({Path(p) for pattern in args.run for p in glob.glob(pattern)})
    if not run_dirs:
        raise SystemExit(f"没有匹配到 run：{args.run}")

    rows: list[dict] = []
    for run_dir in run_dirs:
        ckpt = run_dir / "checkpoints" / "best.pt"
        if not ckpt.is_file():
            print(f"跳过（无 best.pt）：{run_dir}")
            continue
        cfg = load_run_config(run_dir)
        split = args.split or cfg["data"]["split_eval"]
        num_classes = int(cfg["model"]["num_classes"])
        # 先验只用 train 真值；评估用 split。
        from framework.cleansight_eval.temporal.data import load_split

        _f, train_truths, id2name = load_split(cfg["data"], cfg["data"]["split_train"],
                                               feature_schema=cfg.get("feature_schema"))
        log_transition, min_duration = estimate_priors(train_truths, num_classes, args.duration_quantile)
        if args.no_duration:
            min_duration = np.ones(num_classes, dtype=np.float64)
        logits, truths, _ = collect_logits(cfg, ckpt, split, args.device)
        names = [id2name[i] for i in range(num_classes)]

        argmax = [np.argmax(lg, axis=1).astype(np.int64) for lg in logits]
        viterbi = [
            viterbi_decode(lg, log_transition, min_duration, args.transition_strength,
                           not args.no_duration)
            for lg in logits
        ]

        def measure(preds):
            truth_map = {str(i): truths[i].tolist() for i in range(len(truths))}
            pred_map = {str(i): preds[i].tolist() for i in range(len(preds))}
            out = temporal_metrics(pred_map, truth_map, list(range(num_classes)))
            frame, segment = out["frame"], out["segment"]
            recall = {
                names[c]: frame["per_class"][str(c)]["recall"]
                for c in range(num_classes)
                if frame["per_class"][str(c)]["recall"] is not None
            }
            n_gt_seg = sum(len(np.unique(t)) and int(1 + (np.diff(t) != 0).sum()) for t in truths)
            n_pred_seg = sum(int(1 + (np.diff(p) != 0).sum()) for p in preds)
            cov = [c for t, p in zip(truths, preds) for c in segment_coverage(t, p)]
            return {
                "acc": 100 * frame["accuracy"],
                "edit": 100 * segment["edit"],
                "f1@0.1": 100 * segment["f1_at_iou"]["0.10"],
                "f1@0.25": 100 * segment["f1_at_iou"]["0.25"],
                "f1@0.5": 100 * segment["f1_at_iou"]["0.50"],
                "recall": recall,
                "seg_ratio": n_pred_seg / max(n_gt_seg, 1),
                "coverage": float(np.median(cov)) if cov else 0.0,
            }

        rows.append({"run": str(run_dir), "argmax": measure(argmax), "viterbi": measure(viterbi)})
        print(f"[{run_dir.name}] argmax edit={rows[-1]['argmax']['edit']:.2f} → "
              f"viterbi edit={rows[-1]['viterbi']['edit']:.2f}", flush=True)

    text = render(args, rows)
    print("\n" + text)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n结构化结果已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
