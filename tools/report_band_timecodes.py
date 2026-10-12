#!/usr/bin/env python3
"""把色带图的内容对应到真实视频时间码（只读已存盘的预测产物，不推理）。

两种输出模式（``--mode``）：

- ``segments``：**逐视频明细**——每个来源（GT 与各臂）的极大连续段表，
  含动作名、真实帧号区间、起止秒、段时长；
- ``matrix``：**GT 动作段 × 各臂**——行 = 一个真实动作段，列 = 该段内每个臂的
  「主力预测类（占该段比例）」与「正确识别比例」，等价于色带图的差异但带时间码；
  末尾附「全部动作帧」的逐臂正确率 / 检出率 / 漏检率汇总。

帧号映射（与数据集口径一致）：第 i 个抽样帧的真实帧号 = ``labels/<split>/<视频>.mp4.txt``
第 i 行的 frame_id（1-based、步长 2）；秒 = (frame_id − 1) / 30（源视频 30fps）。
⚠️ artifact 里的 ``prediction_start_frame`` 是 **0-based 索引基准**，不是真实帧号，勿直接使用。

用法::

    python tools/report_band_timecodes.py --mode matrix          # 默认：GT 段 × 各臂对照
    python tools/report_band_timecodes.py --mode segments        # 逐视频逐段明细
    python tools/report_band_timecodes.py --mode both --csv      # 两个模式 + 附带 CSV

输出（默认 ``outputs/visualizations/bands``）：``band_timecodes_<mode>.md``，``--csv`` 时另出同名 ``.csv``
（UTF-8 BOM，Excel 可直接打开）。
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import statistics
from pathlib import Path

import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/visualizations/bands"
DEFAULT_RUNS = ROOT / "runs/v41"
DEFAULT_DATASET = ROOT / "datasets/cleansight-ActionMixed-auto-lhh"
DEFAULT_ARMS = [
    ("feat-roi-v5", "roi-v5"),
    ("flagship-v41", "roi-v4"),
    ("feat-hand", "hand"),
    ("online-gru-roi-v4", "gru_online"),
]
FPS = 30.0
STRIDE = 2


def parse_args(argv=None) -> argparse.Namespace:
    """解析臂集合、数据集、模式与输出位置。"""

    p = argparse.ArgumentParser(description="色带图 ↔ 视频时间码对照（只读预测产物）")
    p.add_argument("--runs", default=str(DEFAULT_RUNS), help="臂目录父目录")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET), help="数据集根（labels/ + task_ids.yaml）")
    p.add_argument("--split", default="test", help="数据集 split")
    p.add_argument("--arm", action="append", default=[], metavar="NAME[=LABEL]", help="臂（可多次）")
    p.add_argument("--seed-pick", default="median", choices=("median", "best", "worst"))
    p.add_argument("--mode", default="matrix", choices=("segments", "matrix", "both"))
    p.add_argument("--csv", action="store_true", help="同时导出 CSV（UTF-8 BOM）")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="输出目录（默认 outputs/visualizations/bands）")
    return p.parse_args(argv)


def pick_run(arm_dir: Path, pick: str) -> Path:
    """按 acc 选代表 seed 的 run 目录。"""

    rows = []
    for ev in glob.glob(str(arm_dir / "h*" / "*" / "evals" / "*.evaluation.json")):
        acc = json.load(open(ev, encoding="utf-8"))["metrics"]["details"]["temporal"]["frame"]["accuracy"]
        rows.append((float(acc), Path(ev).parent.parent))
    if not rows:
        raise SystemExit(f"{arm_dir.name}: 找不到评测产物")
    ordered = sorted(rows)
    chosen = {"median": statistics.median(a for a, _ in rows),
              "best": ordered[-1][0], "worst": ordered[0][0]}[pick]
    return min(rows, key=lambda item: abs(item[0] - chosen))[1]


def load_pred(run_dir: Path) -> dict[str, np.ndarray]:
    """→ {视频名: 预测 label id 数组}。"""

    path = sorted((run_dir / "artifacts").glob("*.predictions.json"))[-1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {k.split("/")[-1]: np.asarray(v["predicted_label_ids"], dtype=int)
            for k, v in payload["items"].items()}


def frames_and_actions(dataset: Path, split: str, video: str) -> tuple[np.ndarray, np.ndarray]:
    """读 labels 文件 → (真实帧号数组, 动作 id 数组)。"""

    rows = [l.split() for l in (dataset / "labels" / split / f"{video}.txt")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    return (np.array([int(r[0]) for r in rows]), np.array([int(r[1]) for r in rows]))


def segments(label_ids: np.ndarray) -> list[tuple[int, int, int]]:
    """压成极大连续段：[(action_id, 起始索引, 结束索引)]。"""

    out, start = [], 0
    for i in range(1, len(label_ids) + 1):
        if i == len(label_ids) or label_ids[i] != label_ids[start]:
            out.append((int(label_ids[start]), start, i - 1))
            start = i
    return out


def render_segments(tasks: dict, dataset: Path, split: str, preds: dict, names: list[str]) -> list[str]:
    """逐视频逐段明细表。"""

    lines = ["# 色带图 ↔ 真实视频时间码（逐视频明细）", "",
             f"> 帧号映射：第 i 个抽样帧的真实帧号 = `labels/{split}/<视频>.mp4.txt` 第 i 行 frame_id"
             f"（1-based、步长 {STRIDE}）；秒 = (frame_id − 1) / {FPS:.0f}。", ""]
    for video in sorted(tasks):
        start_ms = float(video.split("-clip_")[1].removesuffix(".mp4").split("_")[0])
        t0 = datetime.datetime.fromtimestamp(start_ms / 1000)
        ids, acts = frames_and_actions(dataset, split, video)
        lines += [f"## {video.split('-clip_')[0]}  ·  LS task #{tasks[video]}", "",
                  f"- 录制起点：**{t0:%Y-%m-%d %H:%M:%S}**（本地时区，来自文件名 epoch）",
                  f"- 抽样帧 {len(ids)} 个 · 真实帧号 {ids[0]}..{ids[-1]}（末帧 {(ids[-1]-1)/FPS:.1f}s）", "",
                  "| 来源 | 段 | 动作 | 起始帧 | 结束帧 | 起始秒 | 结束秒 | 时长(s) |",
                  "|---|---:|---|---:|---:|---:|---:|---:|"]
        for tag, arr in [("GT", acts)] + [(lab, preds[lab][video]) for lab in preds]:
            for n, (aid, i0, i1) in enumerate(segments(arr), start=1):
                f0, f1 = int(ids[i0]), int(ids[i1])
                lines.append(f"| {tag} | {n} | {names[aid]} | {f0} | {f1} | "
                             f"{(f0-1)/FPS:.2f} | {(f1-1)/FPS:.2f} | {(f1-f0)/FPS:.2f} |")
        lines.append("")
    return lines


def render_matrix(tasks: dict, dataset: Path, split: str, preds: dict, names: list[str]) -> list[str]:
    """GT 动作段 × 各臂 对照表 + 汇总。"""

    lines = ["# GT 动作段 × 各臂预测（色带图的表格形态）", "",
             f"> 秒 = (真实帧号 − 1) / {FPS:.0f}；抽样步长 {STRIDE}。",
             "> 单元格 = 该段内**主力预测类**（占该段帧数比例）→ 该段被**正确识别**的帧比例。", ""]
    lines.append("| 视频(录制起点) | GT 动作 | 时间段 | 帧号 | 时长 | " + " | ".join(preds) + " |")
    lines.append("|---|---|---|---:|---:|" + "---|" * len(preds))
    stats = {lab: {"total": 0, "hit": 0, "found": 0} for lab in preds}
    for video in sorted(tasks):
        start_ms = float(video.split("-clip_")[1].removesuffix(".mp4").split("_")[0])
        t0 = datetime.datetime.fromtimestamp(start_ms / 1000)
        ids, acts = frames_and_actions(dataset, split, video)
        for aid, i0, i1 in segments(acts):
            if aid == 0:  # 只看真实动作段
                continue
            f0, f1 = int(ids[i0]), int(ids[i1])
            cells = []
            for lab in preds:
                seg = preds[lab][video][i0:i1 + 1]
                cnt = np.bincount(seg, minlength=len(names))
                main = int(cnt.argmax())
                hit = float((seg == aid).mean())
                stats[lab]["total"] += len(seg)
                stats[lab]["hit"] += int((seg == aid).sum())
                stats[lab]["found"] += int((seg != 0).sum())
                mark = "✅" if hit >= 0.5 else ("~" if hit > 0 else "❌")
                cells.append(f"{names[main]} {cnt[main]/len(seg):.0%} → {hit:.0%}{mark}")
            lines.append(f"| {video.split('-clip_')[0]} ({t0:%m-%d %H:%M}) | {names[aid]} | "
                         f"{(f0-1)/FPS:.1f}–{(f1-1)/FPS:.1f}s | {f0}–{f1} | {(f1-f0)/FPS:.1f}s | "
                         + " | ".join(cells) + " |")
    lines += ["", "## 汇总（仅统计 GT 动作帧）", "",
              "| 臂 | 动作帧正确率 | 动作帧被检出（非 idle）率 | 漏检率 |", "|---|---:|---:|---:|"]
    for lab in preds:
        s = stats[lab]
        lines.append(f"| {lab} | {s['hit']/s['total']:.1%} | {s['found']/s['total']:.1%} | "
                     f"{1-s['found']/s['total']:.1%} |")
    return lines


def write_csv(md_path: Path, csv_path: Path) -> int:
    """把 markdown 表格转成 CSV（UTF-8 BOM，Excel 友好）。"""

    rows = []
    for line in md_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and set(line) - set("|-: "):
            rows.append([c.strip() for c in line.strip("|").split("|")])
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        csv.writer(fh).writerows(rows)
    return len(rows)


def main(argv=None) -> int:
    """生成对照表（并可选 CSV），返回退出码。"""

    args = parse_args(argv)
    runs_root, dataset, out_dir = Path(args.runs), Path(args.dataset), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = yaml.safe_load((dataset / "task_ids.yaml").read_text(encoding="utf-8"))[args.split]
    names = yaml.safe_load((dataset / "labels/data.yaml").read_text(encoding="utf-8"))["names"]

    arms = args.arm or [f"{a}={l}" for a, l in DEFAULT_ARMS]
    preds = {}
    for item in arms:
        arm, _, label = item.partition("=")
        run_dir = pick_run(runs_root / arm, args.seed_pick)
        preds[label or arm] = load_pred(run_dir)
        print(f"  {arm:<24} seed-run={run_dir.name}")

    modes = ("segments", "matrix") if args.mode == "both" else (args.mode,)
    for mode in modes:
        body = render_segments(tasks, dataset, args.split, preds, names) if mode == "segments" \
            else render_matrix(tasks, dataset, args.split, preds, names)
        md_path = out_dir / f"band_timecodes_{mode}.md"
        md_path.write_text("\n".join(body) + "\n", encoding="utf-8")
        print(f"  写入 {md_path}（{len(body)} 行）")
        if args.csv:
            csv_path = md_path.with_suffix(".csv")
            print(f"  写入 {csv_path}（{write_csv(md_path, csv_path)} 行）")
    print(f"[timecodes] 完成 → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
