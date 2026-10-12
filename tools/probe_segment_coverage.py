#!/usr/bin/env python3
"""段覆盖率探针：按「动作段真的被检出了吗」评估，而不是按 edit。

**为什么需要它（2026-09-27 用户确认的验收口径）**：现行 headline `edit` 与帧 `acc` 都不能回答
"动作段有没有被检出"——

- `edit` 丢弃时长、只比段标签序列，**全预测 idle 反而拿 13.70 的地板分**（见
  `docs/experiments/EXPERIMENT_REPORT_ARCH_SURVEY_20260927.md` §1.1/§1.5）；
- 帧 `acc` 被 idle 占比撑起来，旗舰 53.93 **低于**全 idle 的 55.25。

本探针以**真值动作段**为单位给出三个直接读数：

1. **段覆盖率**：该段内被预测为**任一非 idle 类**的帧比例（只问"这段有没有被当成动作"，
   暂不问是哪一类）——中位数与 ≥10% / ≥50% / ≥80% 的段占比；
2. **逐类段覆盖率**与**逐类帧召回**（后者与官方 micro-pool 口径一致）；
3. **活动量上下文**：非 idle 帧数与段数比（避免"靠多说话换覆盖率"的假增益）。

只读已存盘 `artifacts/*.predictions.json`，不训练、不加载模型。

用法：

    python tools/probe_segment_coverage.py --run "runs/arch-survey/baseline-mstcn2-cpu/h128/mstcn2-*" \
        --label baseline
    python tools/probe_segment_coverage.py --run "<glob>" --label A --run "<glob2>" --label B --json tmp/cov.json
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
    parser = argparse.ArgumentParser(description="按真值动作段的覆盖率评估「段有没有被检出」")
    parser.add_argument("--run", action="append", required=True, help="run 目录 glob（可多次）")
    parser.add_argument("--label", action="append", default=None,
                        help="与 --run 一一对应的显示名（缺省用 glob 本身）")
    parser.add_argument("--json", default=None, help="结构化结果输出路径")
    return parser.parse_args(argv)


def load_predictions(run_dir: Path) -> list[tuple[np.ndarray, np.ndarray, list[str]]]:
    """读一个 run 的 predictions artifact，返回 [(truth, pred, class_names)]。

    ``labels`` 有两种历史形态：``[{"id":0,"name":"idle"}, …]``（现行为）与 ``["idle", …]``
    （早期产物），两者都接受。
    """

    paths = sorted((run_dir / "artifacts").glob("*.predictions.json"))
    if not paths:
        return []
    payload = json.loads(paths[-1].read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, dict) or not items:
        return []
    if "truth_label_ids" not in next(iter(items.values())):
        return []  # 检测（YOLO）等非时序产物
    names = [entry if isinstance(entry, str) else entry["name"] for entry in payload["labels"]]
    out = []
    for item in payload["items"].values():
        out.append((
            np.asarray(item["truth_label_ids"], dtype=np.int64),
            np.asarray(item["predicted_label_ids"], dtype=np.int64),
            names,
        ))
    return out


def measure(sequences: list[tuple[np.ndarray, np.ndarray, list[str]]]) -> dict:
    """汇总逐类段覆盖率、段占比、**动作段 precision** 与帧召回。

    覆盖率单独使用会被"到处都说是动作"刷满（全预测非 idle ⇒ 覆盖率 100%），因此必须与
    **动作段 precision** 配对：预测的非 idle 段中有多大比例落在真值动作段内。两者都按
    ≥50% 帧占比判定，再给一个平衡读数 ``seg_f1``（几何平均的 F1 形式）用于排序。
    """

    coverages: list[float] = []
    per_class: dict[str, list[float]] = {}
    gt_frames = 0
    hit_frames = 0
    nonidle_pred = 0
    total_frames = 0
    n_gt_seg = 0
    n_pred_seg = 0
    action_hit_frames = 0     # 预测为非 idle 且落在真值动作段内的帧（动作帧精确率的分子）
    for truth, pred, names in sequences:
        total_frames += len(truth)
        nonidle_pred += int((pred != 0).sum())
        truth_action = truth != 0
        action_hit_frames += int(((pred != 0) & truth_action).sum())
        change = np.concatenate([[0], np.where(np.diff(truth) != 0)[0] + 1, [len(truth)]])
        for s, e in zip(change[:-1], change[1:]):
            n_gt_seg += 1
            label = int(truth[s])
            if label == 0:
                continue
            cov = float((pred[s:e] != 0).mean())
            coverages.append(cov)
            per_class.setdefault(names[label], []).append(cov)
            gt_frames += e - s
            hit_frames += int((pred[s:e] != 0).sum())
        n_pred_seg += int(1 + (np.diff(pred) != 0).sum())
    cov_arr = np.asarray(coverages) if coverages else np.zeros(1)
    recall = float((cov_arr >= 0.50).mean())
    # 动作帧精确率：预测为非 idle 的帧里有多大比例真的落在真值动作段内。
    # 这是防"到处都说是动作"的关键配对量（覆盖率会被该退化解刷满）。
    precision = action_hit_frames / nonidle_pred if nonidle_pred else 0.0
    seg_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_gt_segments": n_gt_seg,
        "n_action_segments": len(coverages),
        "coverage_median": float(np.median(cov_arr)),
        "coverage_mean": float(cov_arr.mean()),
        "share_ge_10": float((cov_arr >= 0.10).mean()),
        "share_ge_50": recall,
        "share_ge_80": float((cov_arr >= 0.80).mean()),
        "action_frame_precision": precision,
        "seg_f1": seg_f1,
        
        "per_class_coverage": {k: float(np.median(v)) for k, v in per_class.items()},
        "per_class_share_ge_50": {k: float(np.mean(np.asarray(v) >= 0.50)) for k, v in per_class.items()},
        "per_class_n": {k: len(v) for k, v in per_class.items()},
        "action_frame_recall_vs_nonidle": hit_frames / gt_frames if gt_frames else 0.0,
        "nonidle_pred_frames": nonidle_pred,
        "total_frames": total_frames,
        "seg_ratio": n_pred_seg / max(n_gt_seg, 1),
    }



def per_item_readings(sequences: list[tuple[np.ndarray, np.ndarray, list[str]]]) -> list[dict]:
    """逐视频算段召回（≥50% 覆盖的动作段占比）与动作帧精确率，用于配对检验。"""

    out = []
    for truth, pred, _names in sequences:
        action = truth != 0
        change = np.concatenate([[0], np.where(np.diff(truth) != 0)[0] + 1, [len(truth)]])
        covered = 0
        total = 0
        for s, e in zip(change[:-1], change[1:]):
            if truth[s] == 0:
                continue
            total += 1
            if float((pred[s:e] != 0).mean()) >= 0.50:
                covered += 1
        nonidle = int((pred != 0).sum())
        hit = int(((pred != 0) & action).sum())
        recall_v = covered / total if total else None
        precision_v = hit / nonidle if nonidle else None
        # 逐视频 seg_f1：召回与精确率的调和平均（两者都非 0 时才有意义）
        seg_f1_v = (
            2 * precision_v * recall_v / (precision_v + recall_v)
            if recall_v and precision_v else None
        )
        out.append({
            "n_action_segments": total,
            "seg_recall": recall_v,
            "action_frame_precision": precision_v,
            "seg_f1": seg_f1_v,
        })
    return out


def paired_test(left: list[dict], right: list[dict], key: str) -> dict | None:
    """逐视频配对 Wilcoxon（零差值的对剔除），返回 {n, wins, losses, median, p}。"""

    from scipy.stats import wilcoxon

    pairs = [
        (a[key], b[key])
        for a, b in zip(left, right)
        if a.get(key) is not None and b.get(key) is not None and a[key] != b[key]
    ]
    if len(pairs) < 5:
        return None
    diffs = np.asarray([a - b for a, b in pairs], dtype=float)
    wins = int((diffs > 0).sum())
    losses = int((diffs < 0).sum())
    return {
        "n": len(pairs),
        "wins": wins,
        "losses": losses,
        "median": float(np.median(diffs)),
        "p": float(wilcoxon(diffs).pvalue),
    }


def structure_health(run_dirs: list[Path]) -> dict:
    """读同 run 的官方评测 summary，取段结构健康度（F1@0.1 / edit），用于识别退化解。

    覆盖率口径只问「段里有没有非 idle 帧」，不问段边界与类别是否正确。**"到处都报动作"
    可以让覆盖率刷到 100%**（实例见第 2 轮的 asformer 臂：覆盖 61.8% 但 edit 15.71 ≈
    全 idle 地板分）。因此必须把 F1@0.1 / edit 一起报，并对明显退化给出告警。
    """

    f1, edit = [], []
    for run_dir in run_dirs:
        for path in sorted((run_dir / "evals").glob("*.evaluation.json")):
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))["metrics"]["summary"]
            except Exception:
                continue
            for arr, key in ((f1, "f1@0.1"), (edit, "edit")):
                value = summary.get(key)
                value = value.get("value") if isinstance(value, dict) else value
                if value is not None:
                    arr.append(float(value))
    return {
        "f1_01": statistics.median(f1) if f1 else None,
        "edit": statistics.median(edit) if edit else None,
    }


def degenerate_flag(m: dict) -> str:
    """退化解告警：覆盖率不低，但段结构已经崩掉。"""

    recall = m["share_ge_50"]
    f1 = m.get("f1_01")
    if f1 is None:
        return ""
    if recall >= 0.45 and (f1 < 25 or m["seg_ratio"] > 1.8):
        return "⚠ 疑似退化解（覆盖率高但段结构崩）"
    return ""

def render(labels: list[str], per_label: list[dict]) -> str:
    """渲染 markdown 主表与逐类表。"""

    lines = ["# 段覆盖率探针（按「动作段真的被检出」评估）", "",
             "> 覆盖率 = 真值动作段内被预测为**任一非 idle 类**的帧比例。",
             "> **必须与动作帧精确率、段数比、F1@0.1 成对看**：覆盖率单用会被「到处都说是动作」刷满。",
             "> 末列给出**退化解告警**（覆盖率不低但 F1@0.1 崩或段数比 >1.8）——",
             "> 实例：asformer 臂覆盖率 61.8% 却 F1@0.1 13.06、edit 15.71（≈全 idle 地板分）。", ""]
    lines.append("| 臂 | run 数 | 动作段数 | **seg_f1** | 段召回(≥50%) | 动作帧P | ≥80% "
                 "| 段数比 | F1@0.1 | edit | 判读 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for label, m in zip(labels, per_label):
        f1 = "—" if m.get("f1_01") is None else f"{m['f1_01']:.1f}"
        ed = "—" if m.get("edit") is None else f"{m['edit']:.1f}"
        lines.append(
            f"| {label} | {m['n_runs']} | {m['n_action_segments']} "
            f"| **{100 * m['seg_f1']:.1f}%** | {100 * m['share_ge_50']:.1f}% "
            f"| {100 * m['action_frame_precision']:.1f}% "
            f"| {100 * m['share_ge_80']:.1f}% "
            f"| {m['seg_ratio']:.2f} | {f1} | {ed} | {degenerate_flag(m)} |"
        )
    classes = sorted({c for m in per_label for c in m["per_class_coverage"]})
    lines += ["", "## 逐类段覆盖率中位（括号内 = 覆盖率 ≥50% 的段占比）", "",
              "| 臂 | " + " | ".join(classes) + " |", "|---|" + "---:|" * len(classes)]
    for label, m in zip(labels, per_label):
        cells = []
        for cls in classes:
            if cls in m["per_class_coverage"]:
                cells.append(f"{100 * m['per_class_coverage'][cls]:.0f}% ({100 * m['per_class_share_ge_50'][cls]:.0f}%, "
                             f"n={m['per_class_n'][cls]})")
            else:
                cells.append("—")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    args = parse_args(argv)
    labels = args.label or args.run
    if len(labels) != len(args.run):
        raise SystemExit("--label 数量必须与 --run 一致")
    per_label = []
    for label, pattern in zip(labels, args.run):
        run_dirs = sorted({Path(p) for p in glob.glob(pattern)})
        seqs = [s for run_dir in run_dirs for s in load_predictions(run_dir)]
        if not seqs:
            raise SystemExit(f"{label}: 没有可读的 predictions artifact（{pattern}）")
        summary = measure(seqs)
        summary["n_runs"] = len(run_dirs)
        summary.update(structure_health(run_dirs))
        summary["per_item"] = per_item_readings(seqs)
        per_label.append(summary)
        print(f"[{label}] run={len(run_dirs)} 动作段={summary['n_action_segments']} "
              f"覆盖率中位={100 * summary['coverage_median']:.1f}% "
              f"非idle帧={summary['nonidle_pred_frames']}", flush=True)
    text = render(labels, per_label)
    if len(per_label) == 2:
        tests = {}
        for key, human in (("seg_recall", "段召回(≥50%)"), ("action_frame_precision", "动作帧精确率"),
                           ("seg_f1", "seg_f1（平衡分）")):
            outcome = paired_test(per_label[0]["per_item"], per_label[1]["per_item"], key)
            if outcome:
                tests[key] = outcome
                mark = "显著" if outcome["p"] < 0.05 else "不显著"
                text += (f"\n- 配对检验（逐视频 Wilcoxon）**{human}**：{labels[0]} − {labels[1]} "
                         f"中位差 {100 * outcome['median']:+.1f}pp，胜/负 {outcome['wins']}/{outcome['losses']}，"
                         f"n={outcome['n']}，p={outcome['p']:.4f} → **{mark}**")
        per_label[0]["paired_vs_second"] = tests
    print("\n" + text)
    if args.json:
        Path(args.json).write_text(
            json.dumps(dict(zip(labels, per_label)), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"\n结构化结果已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
