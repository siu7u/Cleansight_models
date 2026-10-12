#!/usr/bin/env python3
"""把已存盘的 run 按**验收口径（段覆盖率）**重新排名，零训练成本。

动机：现行 headline 与选点口径都是 `edit`/`f1@k`，而它们**奖励少分段**（全预测 idle 拿 13.70 地板分，
见 `docs/experiments/EXPERIMENT_REPORT_ARCH_SURVEY_20260927.md` §1.1/§1.5）。用户确认的验收口径是
「**动作段真的被检出**」——真值动作段的覆盖率。于是历史 run 的排名**可能与当时不同**：
在 edit 口径下被淘汰的配置（更小容量、无 T-MSE、更"敢说"的权重），在覆盖率口径下可能是最好的。

本工具对每个 run 读 `config.resolved.json` + `artifacts/*.predictions.json`（必要时读
`evals/*.evaluation.json` 的 headline），算覆盖率口径读数，按配置分组排名。

用法：

    python tools/rank_runs_by_coverage.py --root runs --min-runs 3
    python tools/rank_runs_by_coverage.py --root runs --json tmp/coverage_ranking.json
    python tools/rank_runs_by_coverage.py --root runs --eval-dir "_eval_md1"   # 指定口径变体目录
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_segment_coverage import degenerate_flag, load_predictions, measure  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="按段覆盖率口径给历史 run 重新排名")
    parser.add_argument("--root", default="runs", help="run 根目录")
    parser.add_argument("--min-runs", type=int, default=3, help="配置至少要有几个 run 才进榜")
    parser.add_argument("--eval-dir", default=None,
                        help="只在某个口径变体目录下找 predictions（如 _eval_md1）；缺省全找")
    parser.add_argument("--json", default=None, help="结构化结果输出路径")
    parser.add_argument("--top", type=int, default=40, help="打印前几名")
    return parser.parse_args(argv)


def find_runs(root: Path, eval_dir: str | None) -> dict[Path, Path]:
    """返回 {run_dir: predictions_json}；同一 run 多份产物取文件名最大的那份。"""

    found: dict[Path, Path] = {}
    pattern = f"**/{eval_dir}/**/*.predictions.json" if eval_dir else "**/*.predictions.json"
    for path in root.glob(pattern):
        if "artifacts" not in path.parts:
            continue
        run_dir = path.parent.parent
        if not (run_dir / "config.resolved.json").is_file():
            continue
        if run_dir not in found or path.name > found[run_dir].name:
            found[run_dir] = path
    return found


def describe_config(cfg: dict) -> dict:
    """从 resolved 配置里抽出分组键与可读描述。"""

    model = cfg.get("model", {})
    train = cfg.get("train", {})
    schema = cfg.get("feature_schema", {})
    aug = cfg.get("augmentation") or {}
    parts = [model.get("type", "?")]
    for key in ("hidden", "num_stages", "num_layers", "dropout", "tmse_weight"):
        if model.get(key) is not None:
            parts.append(f"{key[:4]}{model[key]}")
    return {
        "pipeline": cfg.get("pipeline"),
        "model": model.get("type"),
        "params_key": tuple(sorted((k, str(v)) for k, v in model.items() if k != "type")),
        "feature": schema.get("version"),
        "dim": schema.get("dim"),
        "epochs": train.get("epochs"),
        "best_metric": train.get("best_metric"),
        "cwclip": str(train.get("class_weight_clip")),
        "train_frac": (cfg.get("data") or {}).get("train_video_fraction"),
        "aug": "on" if (aug.get("target_mask") or {}).get("enabled") else "off",
        "label": " ".join(str(p) for p in parts) + f" | {schema.get('dim')}d | ep{train.get('epochs')}"
                 + (f" | cw{train.get('class_weight_clip')}" if train.get("class_weight_clip") else "")
                 + (" | aug" if (aug.get("target_mask") or {}).get("enabled") else ""),
    }


def headline(evaluation: Path) -> dict:
    """读官方评测 summary 的 edit/acc/f1@0.1（缺省空 dict）。"""

    try:
        payload = json.loads(evaluation.read_text(encoding="utf-8"))
        summary = payload["metrics"]["summary"]
    except Exception:
        return {}
    out = {}
    for key in ("edit", "acc", "f1@0.1", "f1@0.25"):
        value = summary.get(key)
        if isinstance(value, dict):
            value = value.get("value")
        if value is not None:
            out[key] = float(value)
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    root = (ROOT / args.root).resolve() if not Path(args.root).is_absolute() else Path(args.root)
    runs = find_runs(root, args.eval_dir)
    print(f"扫描到 {len(runs)} 个 run（含 predictions + config.resolved.json）", flush=True)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for run_dir, predictions in sorted(runs.items()):
        cfg = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
        meta = describe_config(cfg)
        entries = load_predictions_at(predictions)
        if not entries:
            continue
        m = measure(entries)
        evals = sorted(run_dir.glob("evals/*.evaluation.json"))
        m.update(headline(evals[-1]) if evals else {})
        m["degenerate"] = "⚠" if degenerate_flag({**m, "f1_01": m.get("f1@0.1")}) else ""
        m["run"] = str(run_dir.relative_to(ROOT))
        key = (meta["pipeline"], meta["feature"], meta["label"], meta["train_frac"], meta["best_metric"])
        groups[key].append({"meta": meta, **m, "label": meta["label"]})

    rows = []
    for key, members in groups.items():
        if len(members) < args.min_runs:
            continue
        row = {
            "label": members[0]["label"],
            "pipeline": key[0],
            "feature": key[1],
            "train_frac": key[3],
            "best_metric": key[4],
            "n_runs": len(members),
        }
        for metric in ("share_ge_10", "share_ge_50", "share_ge_80", "seg_f1",
                       "action_frame_precision", "action_frame_recall_vs_nonidle",
                       "nonidle_pred_frames", "coverage_median", "seg_ratio", "edit", "acc",
                       "f1@0.1"):
            values = [m[metric] for m in members if m.get(metric) is not None]
            row[metric] = statistics.median(values) if values else None
        row["action_segments"] = int(statistics.median(m["n_action_segments"] for m in members))
        rows.append(row)
    rows.sort(key=lambda r: (-(r["seg_f1"] or 0), -(r["share_ge_50"] or 0)))

    lines = ["# 历史 run 按「段覆盖率」口径重排（零训练成本）", "",
             f"- 扫描根：`{args.root}`；配置至少 {args.min_runs} 个 run 才进榜；"
             f"各列取**跨 run 中位**。",
             "- **排序键 = `seg_f1`**：`段召回`（真值动作段中 ≥50% 帧被认成动作的比例）与"
             "`动作帧精确率`（预测非 idle 帧中真落在真值动作段内的比例）的调和平均。",
             "- 两者必须成对看：覆盖率单用会被「到处都说是动作」刷满（全预测非 idle ⇒ 召回 100%）。", "",
             "| # | 配置 | run | 动作段 | **seg_f1** | 段召回 | 动作帧P | ≥10% | 非idle帧 | 段数比 "
             "| F1@0.1 | edit | acc | 判读 |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for index, row in enumerate(rows[:args.top], start=1):
        fmt = lambda v, p=1: "—" if v is None else f"{100*v:.{p}f}%" if v <= 1.5 else f"{v:.{p}f}"
        lines.append(
            f"| {index} | {row['label']} | {row['n_runs']} | {row['action_segments']} "
            f"| **{fmt(row['seg_f1'])}** | {fmt(row['share_ge_50'])} "
            f"| {fmt(row['action_frame_precision'])} | {fmt(row['share_ge_10'])} "
            f"| {row['nonidle_pred_frames']:.0f} "
            f"| {row['seg_ratio']:.2f} | {fmt(row['f1@0.1'])} | {fmt(row['edit'])} | {fmt(row['acc'])} "
            f"| {'⚠ 疑似退化解' if row.get('degenerate') else ''} |"
        )
    text = "\n".join(lines)
    print("\n" + text)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n结构化结果已写入 {args.json}")
    return 0


def load_predictions_at(predictions: Path) -> list[tuple[np.ndarray, np.ndarray, list[str]]]:
    """从指定 predictions 文件读 (truth, pred, names) 列表（避免重复扫描目录）。

    ``labels`` 有两种历史形态：``[{"id":0,"name":"idle"}, …]``（现行为）与 ``["idle", …]``
    （早期产物），两者都接受。
    """

    payload = json.loads(predictions.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, dict) or not items:
        return []
    first = next(iter(items.values()))
    if "truth_label_ids" not in first or "predicted_label_ids" not in first:
        return []  # 检测（YOLO）等非时序产物，跳过
    labels = payload["labels"]
    names = [entry if isinstance(entry, str) else entry["name"] for entry in labels]
    return [
        (np.asarray(item["truth_label_ids"], dtype=np.int64),
         np.asarray(item["predicted_label_ids"], dtype=np.int64), names)
        for item in items.values()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
