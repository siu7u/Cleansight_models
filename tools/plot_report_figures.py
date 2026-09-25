"""报告图表生成：把已定版报告里的关键实验数据画成可复用的图（`docs/figures/`）。

用途
    周报 / 汇报需要**用图佐证数字**。本脚本从**已定版报告**取数，生成 PNG + 同名 JSON 旁证：
    - ``docs/figures/<name>.png``   直接嵌进 Markdown 报告
    - ``docs/figures/<name>.json``  该图**实际画出的数字** + 来源标注（审计线索）

数据一致性纪律（重要）
    图中每个数字都必须能在其 ``source`` 指向的报告里查到原文。改图 = 先改报告，再改本脚本，
    两边必须一致；**不允许**在这里维护"比报告更新"的数字。
    容量图是唯一例外：它直接读既有的真实 run 级数据
    ``docs/mstcn-capacity/figures/capacity_vs_segmental.json``（不复制、不转抄）。

字体
    本机**没有中文字体**，图内标签一律用英文；中文解释写在报告正文与图注里，避免渲染出豆腐块。

用法
    PYTHONPATH=. <CleanSightBackend venv>/bin/python tools/plot_report_figures.py
    # 可选：--only fig2_architecture_vs_metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("docs/figures")
CAPACITY_SRC = Path("docs/mstcn-capacity/figures/capacity_vs_segmental.json")

# 统一配色（同一含义在不同图里用同一颜色）
C_MODEL = "#1f77b4"
C_BASE = "#7f7f7f"
C_BAD = "#d62728"
C_GOOD = "#2ca02c"
C_ACCENT = "#ff7f0e"
GRID = dict(axis="y", ls=":", lw=0.6, alpha=0.5)


def _style(ax, title: str, ylabel: str = "") -> None:
    """统一子图样式：标题、网格、去掉上/右边框。"""

    ax.set_title(title, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(**GRID)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, name: str, title: str, source: str, data: dict, dpi: int = 115) -> None:
    """保存 PNG 与同名 JSON 旁证（含来源标注），并打印产物清单。"""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png, side = OUT_DIR / f"{name}.png", OUT_DIR / f"{name}.json"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    side.write_text(
        json.dumps({"figure": name, "title": title, "source": source, "data": data},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  {png}  +{side.name}")


# ---------------------------------------------------------------------------
# fig1：容量 vs 参数量（真实 run 级数据，含 seed 极差）
# ---------------------------------------------------------------------------
def fig1_capacity_vs_params():
    """4 条配方曲线：x=参数量(log)、y=段级 edit（均值 + seed 极差误差棒）。"""

    raw = json.loads(CAPACITY_SRC.read_text(encoding="utf-8"))
    recipes = {
        "mstcn · default recipe (lr 2e-3, 30 ep)": dict(color=C_BASE, marker="o", ls="--"),
        "mstcn · lr 5e-4, 60 ep": dict(color=C_MODEL, marker="s"),
        "mstcn · lr 5e-4, 150 ep": dict(color=C_GOOD, marker="^"),
        "mstcn2 · lr 5e-4, 60 ep": dict(color=C_ACCENT, marker="D"),
    }

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    plotted = {}
    for name, style in recipes.items():
        pts = raw[name]
        xs = np.array([int(k) for k in pts], dtype=float)
        order = np.argsort(xs)
        xs = xs[order]
        keys = [list(pts)[i] for i in order]
        ys = np.array([pts[k]["edit"] for k in keys])
        lo = ys - np.array([pts[k]["edit_min"] for k in keys])
        hi = np.array([pts[k]["edit_max"] for k in keys]) - ys
        n = [pts[k]["n"] for k in keys]
        ax.errorbar(xs, ys, yerr=[lo, hi], capsize=3, lw=1.6, ms=6,
                    label=f"{name}  (n={min(n)}-{max(n)} seeds)", **style)
        plotted[name] = {"params": xs.astype(int).tolist(), "edit": np.round(ys, 2).tolist(),
                         "edit_min": [pts[k]["edit_min"] for k in keys],
                         "edit_max": [pts[k]["edit_max"] for k in keys], "n": n}

    ax.set_xscale("log")
    ax.annotate("adding params is useless here\n(flat across 10k -> 2.1M)",
                xy=(5.5e5, 30.5), xytext=(2.0e4, 36.0), fontsize=8, color=C_BASE,
                arrowprops=dict(arrowstyle="->", color=C_BASE, lw=1))
    ax.annotate("h128 (545k) is the ceiling:\nh256 (2.1M) is equivalent",
                xy=(2.14e6, 50.7), xytext=(1.1e5, 55.5), fontsize=8, color=C_GOOD,
                arrowprops=dict(arrowstyle="->", color=C_GOOD, lw=1))
    ax.annotate("mstcn2 @3.31M params: edit 51.47,\nseed spread only 0.64",
                xy=(3.31e6, 51.47), xytext=(6.0e5, 44.0), fontsize=8, color=C_ACCENT,
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1))
    ax.axhline(51.47, color=C_ACCENT, ls=":", lw=0.8, alpha=0.6)

    ax.set_xlabel("trainable parameters (log scale)", fontsize=9)
    _style(ax, "Capacity vs segmental edit (error bar = seed min/max)", "segment edit")
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    _save(fig, "fig1_capacity_vs_params",
          "Capacity vs segmental edit across 4 recipes",
          "docs/mstcn-capacity/figures/capacity_vs_segmental.json（真实 run 级数据）"
          " + docs/mstcn-capacity/MSTCN_CAPACITY_STUDY.md §0/§1",
          plotted)


# ---------------------------------------------------------------------------
# fig2：架构族对照（指标 + 逐 seed 摆幅）
# ---------------------------------------------------------------------------
def fig2_architecture_vs_metrics():
    """左：4 个架构的 edit/F1@0.1/F1@0.25；右：逐 seed 摆幅（稳定性）。"""

    arms = ["mstcn2 s4l10 h128", "mstcn h128", "GRU (md=1)", "Transformer d64x2"]
    metrics = {
        "edit": [51.47, 41.79, 42.40, 27.31],
        "F1@0.1": [44.78, 26.98, 38.66, 23.79],
        "F1@0.25": [35.82, 21.40, 22.73, 18.59],
    }
    swing = {"mstcn2 s4l10 h128": 0.64, "GRU (md=1)": 14.5, "Transformer d64x2": 8.0}
    extra = {"GRU (md=5)": {"edit": 40.89, "F1@0.1": 37.04, "F1@0.25": 22.22}}

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    x = np.arange(len(arms))
    width = 0.26
    for i, (m, vals) in enumerate(metrics.items()):
        bars = ax.bar(x + (i - 1) * width, vals, width, label=m)
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels(arms, fontsize=8, rotation=12, ha="right")
    ax.set_ylim(0, 62)
    _style(ax, "Architecture vs segment metrics (unified recipe, 3-8 seeds)", "score")
    ax.legend(fontsize=8, ncol=3)

    ax2 = axes[1]
    names = list(swing)
    bars = ax2.bar(names, [swing[n] for n in names], color=[C_GOOD, C_BAD, C_BAD], width=0.55)
    ax2.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax2.set_ylim(0, 17)
    ax2.set_xticks(np.arange(len(names)))
    ax2.set_xticklabels(names, fontsize=8, rotation=12, ha="right")
    _style(ax2, "Seed-to-seed swing of edit (lower = more stable)", "")
    ax2.annotate("23x more stable", xy=(0, 0.64), xytext=(0.35, 6.5), fontsize=8, color=C_GOOD,
                 arrowprops=dict(arrowstyle="->", color=C_GOOD, lw=1))

    fig.tight_layout()
    _save(fig, "fig2_architecture_vs_metrics",
          "Architecture comparison and seed stability",
          "docs/EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md §2.3（统一配方，CPU，3~8 seed）",
          {"arms": arms, "metrics": metrics, "seed_swing": swing, "gru_md5_reference": extra})


# ---------------------------------------------------------------------------
# fig3：特征契约的 insert 召回缺口
# ---------------------------------------------------------------------------
def fig3_feature_contract_gap():
    """各替代契约相对 roi-144 的 insert 召回差值（h128）+ 配对 p 值。"""

    contracts = ["bbox-40", "bbox-40-hand", "roi-96-v2\n(visibility re-layout)",
                 "bbox-80-global-hand", "presence-48"]
    delta = [-11.93, -28.15, -32.43, -17.20, -6.66]
    pval = [0.094, 0.0006, 0.0002, 0.068, 0.90]

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    colors = [C_BAD if p < 0.05 else C_BASE for p in pval]
    bars = ax.barh(contracts, delta, color=colors)
    ax.bar_label(bars, labels=[f"{d:+.2f}pp  (p={p})" for d, p in zip(delta, pval)],
                 fontsize=8, padding=3)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlim(-40, 6)
    ax.invert_yaxis()
    _style(ax, "insert recall vs roi-grid-144 (h128)  —  red = significant loss", "")
    ax.annotate("roi-144 is the only contract\nwith stable insert recall",
                xy=(-1.0, -0.45), xytext=(-38, -0.55), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=1))
    _save(fig, "fig3_feature_contract_gap",
          "insert recall deficit of alternative feature contracts vs roi-grid-144",
          "docs/EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md §2.2（mstcn2 s2l5 h128，3 seed，配对 Wilcoxon）",
          {"contracts": [c.replace("\n", " ") for c in contracts], "dim": [40, 40, 96, 80, 48],
           "insert_recall_delta_pp": delta, "paired_p": pval, "baseline": "roi-grid-144 (dim 144)"})


# ---------------------------------------------------------------------------
# fig4：选点口径（免费杠杆）
# ---------------------------------------------------------------------------
def fig4_selection_metric():
    """best_metric 三档口径的 edit / F1@0.1 / acc，并标注 val→test 迁移相关性。"""

    metrics = ["edit", "F1@0.1", "acc"]
    arms = {
        "val_f1_0.5 (current default)": ([49.13, 38.69, 54.64], C_BASE),
        "val_edit (recommended)": ([51.08, 40.01, 53.98], C_GOOD),
        "val_f1_0.25 (3-seed ref)": ([46.97, 39.68, 53.66], C_ACCENT),
    }

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(metrics))
    width = 0.26
    for i, (name, (vals, color)) in enumerate(arms.items()):
        bars = ax.bar(x + (i - 1) * width, vals, width, label=name, color=color)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(30, 60)
    _style(ax, "Checkpoint selection metric (same recipe, 8 seeds)", "score")
    ax.legend(fontsize=8, loc="lower right")
    ax.annotate("val_edit vs default:\nedit +9.63 (p=0.0107), insert recall +5.46 (p=0.0214)",
                xy=(0.0, 51.08), xytext=(0.35, 35.0), fontsize=8, color=C_GOOD,
                arrowprops=dict(arrowstyle="->", color=C_GOOD, lw=1))
    ax.annotate("current default val_f1_0.5 vs test:\nSpearman rho = 0.199 (near random)",
                xy=(-0.26, 49.13), xytext=(1.15, 45.6), fontsize=8, color=C_BAD)
    _save(fig, "fig4_selection_metric",
          "Checkpoint selection metric comparison",
          "docs/EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md §2.4（8 seed 配对；rho 来自 326 run 迁移体检）",
          {"metrics": metrics,
           "arms": {k: v[0] for k, v in arms.items()},
           "val_edit_vs_default": {"edit_delta": 9.63, "p_edit": 0.0107,
                                   "insert_recall_delta": 5.46, "p_insert": 0.0214, "seeds_won": "6/8"},
           "spearman_rho_default_vs_test": 0.199, "n_runs_surveyed": 326})


# ---------------------------------------------------------------------------
# fig5：TimesFM 探针四问
# ---------------------------------------------------------------------------
def fig5_timesfm_probe():
    """2x2：点预测 MAE / 区间校准 / 切换点 F1 / 提前预警命中率。"""

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))

    # (a) 点预测 MAE
    ax = axes[0][0]
    labels = ["TimesFM", "persistence", "constant\n(context mean)"]
    vals = [0.6694, 0.843, 0.6802]
    bars = ax.bar(labels, vals, color=[C_MODEL, C_BASE, C_BAD], width=0.55)
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
    ax.set_ylim(0, 1.0)
    _style(ax, "(a) Point-forecast MAE  (52d2541c / activity_count, H=128)", "MAE (lower better)")
    ax.annotate("ties the constant predictor", xy=(2, 0.6802), xytext=(0.55, 0.90),
                fontsize=8, color=C_BAD, arrowprops=dict(arrowstyle="->", color=C_BAD, lw=1))

    # (b) 区间覆盖率
    ax = axes[0][1]
    series = ["52d2541c\nactivity", "1b2c95ff\nactivity", "1b2c95ff\nc0_hand", "e8ea5bb7\np6_short_brush"]
    cov = [0.826, 0.884, 0.836, 0.756]
    bars = ax.bar(series, cov, color=C_MODEL, width=0.55)
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.axhline(0.80, color=C_GOOD, ls="--", lw=1.2, label="nominal 0.80")
    ax.set_ylim(0.6, 0.95)
    _style(ax, "(b) q10-q90 interval coverage (well calibrated)", "coverage")
    ax.legend(fontsize=8)

    # (c) 切换点检出 F1
    ax = axes[1][0]
    vids = ["52d2541c\n(28 changes)", "1b2c95ff\n(14 changes)"]
    resid = [0.179, 0.000]
    diff = [0.393, 0.214]
    x = np.arange(len(vids))
    b1 = ax.bar(x - 0.18, resid, 0.36, label="TimesFM 1-step residual", color=C_BAD)
    b2 = ax.bar(x + 0.18, diff, 0.36, label="naive |delta| (zero cost)", color=C_GOOD)
    ax.bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(vids, fontsize=8)
    ax.set_ylim(0, 0.5)
    _style(ax, "(c) Change-point detection F1  (top-K, +-6 steps)", "F1")
    ax.legend(fontsize=8)

    # (d) 提前预警命中率
    ax = axes[1][1]
    leads = [1.07, 2.13, 3.20, 4.27]
    curves = {
        "TimesFM": ([0.125, 0.125, 0.625, 0.750], C_MODEL, "o"),
        "context mean": ([0.625, 0.625, 0.750, 0.750], C_GOOD, "s"),
        "tail-16 mean": ([0.125, 0.375, 0.625, 0.750], C_BASE, "^"),
        "persistence": ([0.125, 0.000, 0.375, 0.750], C_BAD, "v"),
    }
    for name, (ys, color, marker) in curves.items():
        ax.plot(leads, ys, marker=marker, color=color, lw=1.8, ms=6, label=name)
    ax.set_ylim(-0.06, 1.0)
    ax.set_xlabel("lead time before the real transition (s)", fontsize=9)
    _style(ax, "(d) Hit rate of anticipating the next regime (1b2c95ff, 8 transitions)", "hit rate")
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    _save(fig, "fig5_timesfm_probe",
          "TimesFM zero-shot probe: four questions with baselines",
          "docs/EXPERIMENT_REPORT_TIMESFM_FEASIBILITY_20260924.md §4（timesfm-2.5-200m 零样本，4 val 视频，CPU）",
          {"a_point_forecast_mae": {"timesfm": 0.6694, "persistence": 0.843, "constant": 0.6802},
           "b_interval_coverage": {"series": [s.replace("\n", " ") for s in series], "coverage": cov,
                                   "nominal": 0.80},
           "c_change_point_f1": {"videos": [v.replace("\n", " ") for v in vids],
                                 "timesfm_residual": resid, "naive_diff": diff, "tolerance_steps": 6},
           "d_lead_hit_rate": {"leads_s": leads,
                               **{k: v[0] for k, v in curves.items()},
                               "n_transitions": 8}})


# ---------------------------------------------------------------------------
# fig6：部署代价
# ---------------------------------------------------------------------------
def fig6_serving_latency():
    """对数刻度：GRU 单 tick / 帧预算 / TimesFM 单序列与批处理折算。

    TimesFM 单序列画**区间条**（287–376 ms = 报告 §4.5 的实测区间原文），不取中位数——
    派生值在报告里查不到，画出来就无法溯源。
    """

    items = [
        ("GRU sliding tick (yours)", 1.49, 1.49, C_GOOD),
        ("frame budget (7.5 fps)", 133.0, 133.0, C_BASE),
        ("TimesFM batch=32 per series", 69.0, 69.0, C_MODEL),
        ("TimesFM single series", 287.0, 376.0, C_BAD),
    ]
    order = items[::-1]
    names = [i[0] for i in order]

    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    y = np.arange(len(order))
    for yi, (_name, lo, hi, color) in zip(y, order):
        if lo == hi:
            bar = ax.barh(yi, lo, color=color, height=0.55)
            ax.bar_label(bar, labels=[f"{lo:.2f} ms"], fontsize=8, padding=3)
        else:
            ax.barh(yi, hi - lo, left=lo, color=color, height=0.55)
            ax.text(hi * 1.15, yi, f"{lo:.0f}-{hi:.0f} ms", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(1, 3000)
    ax.axvline(133.0, color=C_BASE, ls="--", lw=1)
    _style(ax, "Online serving cost: single forecast call vs per-frame budget", "")
    ax.set_xlabel("latency (ms, log scale)", fontsize=9)
    ax.annotate("over the 133 ms frame budget ->\nper-tick streaming is not viable",
                xy=(287, 3), xytext=(1.6, 3.34), fontsize=8, color=C_BAD,
                arrowprops=dict(arrowstyle="->", color=C_BAD, lw=1))
    _save(fig, "fig6_serving_latency",
          "Serving latency: TimesFM vs existing GRU tick and frame budget",
          "docs/EXPERIMENT_REPORT_TIMESFM_FEASIBILITY_20260924.md §4.5（单序列 0.287-0.376 s、批处理折算 69 ms；"
          "本机 CPU 实测）+ docs/INFERENCE_CHAIN_PERF.md（GRU 1.49 ms、帧预算 133 ms）",
          {"unit_note": "报告原文按 s 记录（TimesFM 折算 0.069 s、单序列 0.287-0.376 s；"
                        "GRU 1.49 ms、帧预算 133 ms），本图统一换算为 ms 呈现",
           "items": [{"name": n, "ms_min": lo, "ms_max": hi} for n, lo, hi, _ in items]})


# ---------------------------------------------------------------------------
# fig7：图像 embedding E0 vs E1（段级升、帧级降）
# ---------------------------------------------------------------------------
def fig7_image_embed_e0_e1():
    """E0(bbox-40) vs E1(bbox + 616 维图像 embedding) 的四项指标。"""

    metrics = ["segment\nedit", "frame\nacc", "frame\nmacro-F1", "recall\nair_injection"]
    e0 = [42.49, 47.01, 46.63, 86.96]
    e1 = [45.22, 27.52, 35.66, 0.00]

    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    x = np.arange(len(metrics))
    b1 = ax.bar(x - 0.19, e0, 0.38, label="E0: bbox-40", color=C_MODEL)
    b2 = ax.bar(x + 0.19, e1, 0.38, label="E1: bbox + image embedding (616d)", color=C_ACCENT)
    ax.bar_label(b1, fmt="%.2f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8.5)
    ax.set_ylim(0, 105)
    _style(ax, "Frozen image embedding is NOT a free win (E0 vs E1, 3-seed median)", "score")
    ax.legend(fontsize=8)
    for xi, (a, b) in enumerate(zip(e0, e1)):
        up = b >= a
        ax.annotate(f"{b - a:+.2f}", xy=(xi + 0.19, max(a, b) + 6),
                    ha="center", fontsize=8, color=C_GOOD if up else C_BAD,
                    fontweight="bold")
    _save(fig, "fig7_image_embed_e0_e1",
          "E0 vs E1: effect of adding frozen image embeddings",
          "docs/EXPERIMENT_REPORT_IMAGE_EMBED_E1_20260911.md §4（机制床 9,532 帧，CPU，3 seed 中位数）",
          {"metrics": [m.replace("\n", " ") for m in metrics], "E0_bbox40": e0,
           "E1_bbox_plus_embed616": e1,
           "delta": [round(b - a, 2) for a, b in zip(e0, e1)]})


FIGURES = {
    "fig1_capacity_vs_params": fig1_capacity_vs_params,
    "fig2_architecture_vs_metrics": fig2_architecture_vs_metrics,
    "fig3_feature_contract_gap": fig3_feature_contract_gap,
    "fig4_selection_metric": fig4_selection_metric,
    "fig5_timesfm_probe": fig5_timesfm_probe,
    "fig6_serving_latency": fig6_serving_latency,
    "fig7_image_embed_e0_e1": fig7_image_embed_e0_e1,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只画某一张（figure 名）")
    args = ap.parse_args()

    targets = {args.only: FIGURES[args.only]} if args.only else FIGURES
    print(f"输出目录：{OUT_DIR}")
    for name, fn in targets.items():
        print(f"[{name}]")
        fn()
    print(f"完成：{len(targets)} 张图")


if __name__ == "__main__":
    main()
