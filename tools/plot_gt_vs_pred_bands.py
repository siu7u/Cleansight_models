#!/usr/bin/env python3
"""色带图渲染：把 GT 与各臂预测画成逐帧色块（action segmentation 时间轴）。

只读已存盘的预测产物（``runs/<run>/artifacts/*.predictions.json``），**不加载 checkpoint、不推理**。
每个视频一个子图，子图内每个来源一条色带，自上而下固定顺序：

    GT · <臂1> · <臂2> …        （每条带后的百分比 = 该臂**该视频**的帧准确率）

**配色约定（与 ``benchmark/visualizers/temporal.py`` 一致，勿改）**：按类别 id 排序后的**位置**
取 ``tab10`` 前 n 色 —— idle 蓝 / flush 橙 / long_brush_insert 绿 / long_brush_withdraw 红 /
short_brush_cleaning 紫。**不要按类别 id 直接索引 tab10**，否则会与官方图串色
（本工具初版就踩过：flush 变绿、withdraw 变紫）。

横轴两种口径（``--axis``）：
- ``index``：0-based 抽样帧索引（与官方评测图一致）；
- ``seconds``：源视频秒数，并在上轴给出**真实视频帧号**（1-based），便于对着播放器定位。
  换算：抽样步长 2、源视频 30fps → 秒 = 索引 / 15；真实帧号 = 1 + 2 × 索引。

用法::

    # 默认：4 个代表臂，时间码横轴，每页 4 个视频
    python tools/plot_gt_vs_pred_bands.py
    # 只对比 GT 与最优臂，每页 1 个视频、放大
    python tools/plot_gt_vs_pred_bands.py --arm feat-roi-v5="roi-v5" --per-page 1 --dpi 180
    # 全部臂（自动扫描 runs/v41 下的臂目录）
    python tools/plot_gt_vs_pred_bands.py --all-arms --out outputs/visualizations/v41_bands/all_arms

输出：``<out>/gt_vs_pred_tc_pNN.png``（``--axis seconds``）或 ``gt_vs_pred_index_pNN.png``。
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/visualizations/bands"
DEFAULT_RUNS = ROOT / "runs/v41"
DEFAULT_ARMS = [
    ("feat-roi-v5", "roi-v5 (best)"),
    ("flagship-v41", "roi-v4 (ex-flagship)"),
    ("feat-hand", "hand (top acc)"),
    ("online-gru-roi-v4", "gru online (causal)"),
]
FPS = 30.0
STRIDE = 2
# 数据集的 6 类动作里 water_injection(1) 在 v4.1 无正样本，图中不出现
CLASS_IDS = (0, 2, 3, 4, 5)


def parse_args(argv=None) -> argparse.Namespace:
    """解析臂集合、页码布局、横轴口径与输出目录。"""

    p = argparse.ArgumentParser(description="GT vs 预测 色带图（只读已存盘预测产物）")
    p.add_argument("--runs", default=str(DEFAULT_RUNS), help="臂目录的父目录（其下每个子目录 = 一个臂）")
    p.add_argument("--arm", action="append", default=[], metavar="NAME[=LABEL]",
                   help="要画的臂（可多次；缺省用 4 个代表臂）")
    p.add_argument("--all-arms", action="store_true", help="自动扫描 runs 下全部臂目录")
    p.add_argument("--seed-pick", default="median", choices=("median", "best", "worst"),
                   help="每个臂取哪个 seed 作代表（缺省 median = acc 最接近中位）")
    p.add_argument("--axis", default="seconds", choices=("seconds", "index"),
                   help="横轴口径：seconds（源视频秒数 + 上轴真实帧号）/ index（抽样索引）")
    p.add_argument("--per-page", type=int, default=4, help="每页视频数")
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--out", default=str(DEFAULT_OUT), help="输出目录（默认 outputs/visualizations/bands）")
    return p.parse_args(argv)


def list_arms(runs_root: Path, all_arms: bool, explicit: list[str]) -> list[tuple[str, str]]:
    """确定要画的 (臂目录名, 图例名) 列表。"""

    if explicit:
        out = []
        for item in explicit:
            name, _, label = item.partition("=")
            out.append((name, label or name))
        return out
    if all_arms:
        return [(d.name, d.name) for d in sorted(runs_root.iterdir())
                if d.is_dir() and glob.glob(str(d / "h*" / "*" / "artifacts" / "*.predictions.json"))]
    return list(DEFAULT_ARMS)


def pick_run(arm_dir: Path, pick: str) -> tuple[Path, float]:
    """按 acc 选代表 seed：返回 (run 目录, acc)。"""

    rows = []
    for ev in glob.glob(str(arm_dir / "h*" / "*" / "evals" / "*.evaluation.json")):
        acc = json.load(open(ev, encoding="utf-8"))["metrics"]["details"]["temporal"]["frame"]["accuracy"]
        rows.append((float(acc), Path(ev).parent.parent))
    if not rows:
        raise SystemExit(f"{arm_dir.name}: 找不到评测产物（先跑 benchmark.cli.eval）")
    ordered = sorted(rows)
    chosen = {"median": statistics.median(a for a, _ in rows),
              "best": ordered[-1][0], "worst": ordered[0][0]}[pick]
    acc, run_dir = min(rows, key=lambda item: abs(item[0] - chosen))
    return run_dir, acc


def load_artifact(run_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    """读 run 的 predictions artifact → (truth_by_video, pred_by_video, class_names)。"""

    path = sorted((run_dir / "artifacts").glob("*.predictions.json"))[-1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = [e if isinstance(e, str) else e["name"] for e in payload["labels"]]
    truth, pred = {}, {}
    for key, item in payload["items"].items():
        video = key.split("/")[-1]
        truth[video] = np.asarray(item["truth_label_ids"], dtype=int)
        pred[video] = np.asarray(item["predicted_label_ids"], dtype=int)
    return truth, pred, names


def main(argv=None) -> int:
    """渲染全部页并落盘，返回进程退出码。"""

    args = parse_args(argv)
    runs_root, out_dir = Path(args.runs), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    series = []
    gts: dict[str, np.ndarray] | None = None
    class_names: list[str] = []
    for arm, label in list_arms(runs_root, args.all_arms, args.arm):
        run_dir, acc = pick_run(runs_root / arm, args.seed_pick)
        truth, pred, names_i = load_artifact(run_dir)
        gts = gts or truth                 # 各臂 test split 相同，取首个臂的真值即可
        class_names = class_names or names_i
        series.append((label, pred, acc, run_dir.name))
        print(f"  {arm:<24} seed-run={run_dir.name}  acc={acc*100:.2f}%")

    videos = sorted(gts)
    shown = [cid for cid in range(len(class_names)) if cid in set(CLASS_IDS)]
    names = [f"{cid}:{class_names[cid]}" for cid in shown]
    class_ids = list(CLASS_IDS)
    index_of = {cid: i for i, cid in enumerate(class_ids)}
    # 配色按位置取 tab10（与官方渲染器一致），勿改成按类别 id 取色
    cmap = matplotlib.colors.ListedColormap(plt.cm.tab10.colors[: len(class_ids)])
    handles = [mpatches.Patch(color=cmap(i), label=n) for i, n in enumerate(names)]

    n_bands = len(series) + 1
    pages = (len(videos) + args.per_page - 1) // args.per_page
    tag = "tc" if args.axis == "seconds" else "index"
    written = []
    for page, start in enumerate(range(0, len(videos), args.per_page), start=1):
        chunk = videos[start:start + args.per_page]
        fig, axes = plt.subplots(len(chunk), 1, figsize=(14, 1.5 * n_bands * len(chunk) + 1.6), squeeze=False)
        for ax, video in zip(axes[:, 0], chunk):
            gt = gts[video]
            rows, labels = [np.array([index_of[int(v)] for v in gt])], ["GT"]
            for label, tab, _acc, _run in series:
                p = tab[video]
                if len(p) != len(gt):
                    raise SystemExit(f"{video}: {label} 预测 {len(p)} 帧 != 真值 {len(gt)} 帧")
                rows.append(np.array([index_of[int(v)] for v in p]))
                labels.append(f"{label}  {(p == gt).mean()*100:.1f}%")
            span = len(gt) if args.axis == "index" else len(gt) * STRIDE / FPS
            ax.imshow(np.vstack(rows), aspect="auto", cmap=cmap, vmin=0, vmax=len(class_ids) - 1,
                      interpolation="nearest", extent=(0, span, len(rows) - 0.5, -0.5))
            for cut in np.flatnonzero(np.diff(gt) != 0) + 1:  # GT 段边界，便于对齐
                ax.axvline(cut if args.axis == "index" else cut * STRIDE / FPS,
                           color="white", lw=0.5, ls=":", alpha=0.7)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_title(f"{video[:8]}  (T={len(gt)} sampled frames / {len(gt)*STRIDE/FPS:.1f}s)",
                         fontsize=9, loc="left")
            ax.set_xlabel("sampled frame index" if args.axis == "index"
                          else "time in source video (seconds, 30fps)", fontsize=8)
            ax.set_xlim(0, span)
            if args.axis == "seconds":
                top = ax.secondary_xaxis(
                    "top",
                    functions=(lambda s: 1 + s * FPS / STRIDE, lambda f: (f - 1) * STRIDE / FPS))
                top.set_xlabel("real video frame id (1-based)", fontsize=8)
                top.tick_params(labelsize=7)
        fig.suptitle(f"GT vs Pred color bands | x = {args.axis} | page {page}/{pages}", fontsize=11)
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
        path = out_dir / f"gt_vs_pred_{tag}_p{page:02d}.png"
        fig.savefig(path, dpi=args.dpi)
        plt.close(fig)
        written.append(path)
        print(f"  写入 {path}")
    print(f"[bands] 完成：{len(written)} 页 → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
