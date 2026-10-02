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
          "docs/experiments/EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md §2.3（统一配方，CPU，3~8 seed）",
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
          "docs/experiments/EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md §2.2（mstcn2 s2l5 h128，3 seed，配对 Wilcoxon）",
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
          "docs/experiments/EXPERIMENT_REPORT_FEATURE_ACCURACY_20260923.md §2.4（8 seed 配对；rho 来自 326 run 迁移体检）",
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
          "docs/experiments/EXPERIMENT_REPORT_TIMESFM_FEASIBILITY_20260924.md §4（timesfm-2.5-200m 零样本，4 val 视频，CPU）",
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
          "docs/experiments/EXPERIMENT_REPORT_TIMESFM_FEASIBILITY_20260924.md §4.5（单序列 0.287-0.376 s、批处理折算 69 ms；"
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
          "docs/experiments/EXPERIMENT_REPORT_IMAGE_EMBED_E1_20260911.md §4（机制床 9,532 帧，CPU，3 seed 中位数）",
          {"metrics": [m.replace("\n", " ") for m in metrics], "E0_bbox40": e0,
           "E1_bbox_plus_embed616": e1,
           "delta": [round(b - a, 2) for a, b in zip(e0, e1)]})


def _load_acc_by_seed(patterns):
    """读 run 的 ``evals/*.evaluation.json`` → ``{seed: acc}``（seed 取 ``env.json``）。

    供 fig9/fig10 直接读**真实 run 级数据**（同 fig1 的例外口径：不复制、不转抄报告数字）。
    """

    out = {}
    for pat in patterns:
        for ev in sorted(Path(".").glob(pat)):
            run = ev.parent.parent
            envf = run / "env.json"
            if not envf.is_file():
                continue
            seed = json.loads(envf.read_text())["seed"]
            s = json.loads(ev.read_text())["metrics"]["summary"]["acc"]
            out[seed] = float(s["value"] if isinstance(s, dict) else s)
    return out


def _load_per_class(patterns):
    """读 run 的逐类 frame 指标 → ``{class: {recall:[...], precision:[...], f1:[...]}}``。"""

    acc = {}
    for pat in patterns:
        for ev in sorted(Path(".").glob(pat)):
            pc = json.loads(ev.read_text())["metrics"]["details"]["temporal"]["frame"]["per_class"]
            for cls, d in pc.items():
                slot = acc.setdefault(cls, {"recall": [], "precision": [], "f1": []})
                for k in slot:
                    v = d.get(k)
                    # None = 该 run 没有任何该类预测（precision/F1 不可定义）→ 记 0，
                    # 与 tmp/deepdive/score_audit.py 的逐类口径一致；跳过会偏高中位数。
                    slot[k].append(100.0 * (0.0 if v is None else float(v)))
    return acc


def fig8_acc_vs_action_budget():
    """同动作帧预算下的 6 类准确率（把"决策点偏移"从"判别力"里剥离）。"""

    src = json.loads(Path("tmp/deepdive/score_audit.json").read_text(encoding="utf-8"))
    gated = src["fixed_rate"]["gated_6class"]
    r = np.array([g["r"] for g in gated])
    v1 = np.array([g["v1"] for g in gated])
    v4 = np.array([g["v4"] for g in gated])
    d = np.array([g["d"] for g in gated])       # 配对中位差（与 p 值同源），不是"中位数之差"
    p = np.array([g["p"] for g in gated])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1.55, 1]})
    ax.plot(r, v1, marker="o", ms=5, lw=1.7, color=C_BASE, label="roi-grid-v1 (144d), n=32 seeds")
    ax.plot(r, v4, marker="D", ms=5, lw=1.7, color=C_MODEL, label="roi-grid-v4 (128d), n=32 seeds")
    ax.axvline(44.75, ls="--", lw=1.1, color=C_ACCENT)
    ax.set_ylim(37.5, 64.5)
    ax.annotate("truth action rate 44.75%\n(gain vanishes: +0.98pp n.s.)",
                xy=(44.75, 46.6), xytext=(48.0, 41.2), fontsize=8, color=C_ACCENT,
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1))
    ax.annotate("each arm's own argmax point (20-24%):\nΔ = +3.6pp, p < 1e-6",
                xy=(22.2, 57.0), xytext=(9.5, 62.0), fontsize=8, color=C_MODEL,
                arrowprops=dict(arrowstyle="->", color=C_MODEL, lw=1))
    ax.set_xlabel("action-frame budget  r  (%)  — matched for both arms", fontsize=9)
    _style(ax, "Frame accuracy depends on the decision point\n(6-class acc at a MATCHED action budget)",
           "frame accuracy (%)")
    ax.legend(fontsize=8, loc="lower left")

    sig = ["*" if pi < 0.05 else "" for pi in p]
    bars = ax2.bar([f"{x:.0f}" for x in r], d, color=[C_GOOD if x > 0 else C_BAD for x in d])
    ax2.bar_label(bars, labels=[f"{x:+.2f}{s}" for x, s in zip(d, sig)], fontsize=8, padding=2)
    ax2.axhline(0, lw=0.8, color="k")
    ax2.set_xlabel("action-frame budget r (%)", fontsize=9)
    _style(ax2, "v4 − v1 at each budget\n(* = paired Wilcoxon p < 0.05)", "Δ frame accuracy (pp)")
    ax2.set_ylim(min(-1.0, float(d.min()) - 1.2), float(d.max()) + 1.8)

    _save(fig, "fig8_acc_vs_action_budget",
          "Frame accuracy at a matched action-frame budget (v1 vs v4)",
          "runs/acc-push/acc-roiv{1,4}* 缓存 logits，32 对 seed（tmp/deepdive/score_audit.py → score_audit.json；"
          "口径见 docs/weeks/2026-09-26_2026-10-02/DEEP_DIVE_11_TOPICS_20261001.md §1.2）",
          {"action_budget_pct": r.tolist(), "v1_acc": np.round(v1, 2).tolist(),
           "v4_acc": np.round(v4, 2).tolist(), "delta": np.round(d, 2).tolist(),
           "paired_p": [None if np.isnan(x) else round(float(x), 6) for x in p],
           "win_lose": [f"{g['win']}/{g['lose']}" for g in gated],
           "truth_action_rate_pct": 44.75, "n_seeds": 32,
           "note": "两臂在同一动作帧预算下比较；预算 = 预测非 idle 帧占比。"
                   "两臂 argmax 各自落在 23.97% / 20.06%（不同预算），故 argmax 差 +4.47pp 含工作点效应。"})


def fig9_per_class_change():
    """逐类 F1 / recall：v1 → v4 的重分配（insert 大涨、flush 归零）。"""

    v1 = _load_per_class(["runs/gpu-batch/recheck-flagship/h*/*/evals/*.evaluation.json",
                          "runs/acc-push/acc-roiv1-seedB/h*/*/evals/*.evaluation.json",
                          "runs/acc-push/acc-roiv1-seedC/h*/*/evals/*.evaluation.json"])
    v4 = _load_per_class(["runs/acc-push/acc-roiv4/h*/*/evals/*.evaluation.json",
                          "runs/acc-push/acc-roiv4-seedB/h*/*/evals/*.evaluation.json",
                          "runs/acc-push/acc-roiv4-seedC/h*/*/evals/*.evaluation.json"])
    order = [c for c in ("long_brush_insert", "long_brush_withdraw", "short_brush_cleaning", "flush")
             if c in v1 and c in v4]
    names = [c.replace("long_brush_", "long_brush\n").replace("_", " ") for c in order]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    data = {}
    for axi, key, title in ((ax, "f1", "F1"), (ax2, "recall", "recall")):
        for i, c in enumerate(order):
            a = float(np.median(v1[c][key]))
            b = float(np.median(v4[c][key]))
            col = C_GOOD if b >= a else C_BAD
            axi.plot([a, b], [i, i], lw=2.0, color=col, zorder=2)
            axi.scatter([a], [i], s=52, facecolor="white", edgecolor=C_BASE, zorder=3,
                        label="v1 (144d)" if i == 0 else None)
            axi.scatter([b], [i], s=58, color=C_MODEL, zorder=3,
                        label="v4 (128d)" if i == 0 else None)
            axi.annotate(f"{b - a:+.1f}", xy=(max(a, b), i), xytext=(4, 0), textcoords="offset points",
                         fontsize=8.5, va="center", color=col, fontweight="bold")
            data.setdefault(c, {})[f"v1_{key}"] = round(a, 2)
            data[c][f"v4_{key}"] = round(b, 2)
        axi.set_yticks(range(len(order)))
        axi.set_yticklabels(names if axi is ax else [], fontsize=8.5)
        axi.set_xlim(-3, 62)
        _style(axi, f"per-class {title} (median of 32 paired seeds)", f"{title} (%)")
    ax.legend(fontsize=8, loc="lower right")
    ax2.annotate("flush collapses to 0:\nno flush frame is ever predicted",
                 xy=(0.5, 3), xytext=(10.0, 2.30), fontsize=8, color=C_BAD,
                 arrowprops=dict(arrowstyle="->", color=C_BAD, lw=1))
    ax.annotate("insert / withdraw gain most\n(shape channels w,h)",
                xy=(41.4, 0), xytext=(19.0, 0.42), fontsize=8, color=C_GOOD,
                arrowprops=dict(arrowstyle="->", color=C_GOOD, lw=1))

    _save(fig, "fig9_per_class_change",
          "v1 → v4 is a class reallocation, not a uniform gain",
          "runs/acc-push/acc-roiv{1,4}* 的 evals/*.evaluation.json 逐类 frame 指标（32 对 seed；"
          "口径见 DEEP_DIVE_11_TOPICS_20261001.md §1.3）",
          {"classes": order, "per_class": data, "n_seeds": 32,
           "note": "water_injection 在 test 上 support=0，不可评估；short_brush_cleaning support=51 帧。"})


def fig10_arch_fairness_grid():
    """架构公平性网格：每族 1~7 个配置点，仍无一接近 mstcn2 参考线。"""

    fam_label = {
        "ca": "clean_asformer (never trained before)", "cb": "clean_bigru (never trained before)",
        "t": "transformer", "a": "asformer", "f": "fact", "b": "clean_mstcn_bilstm",
        "m": "mstcn (single-stage)", "g": "gru (full_sequence, causal)",
    }
    fam_order = ["m", "f", "t", "ca", "a", "g", "cb", "b"]
    A_SEEDS = [1, 2, 3, 4, 5, 7, 42, 2026]     # 与参考线、与原配置点同一批 seed
    by_point = {}                              # 配置点 → {seed: acc}（跨目录合并，同点名合并）
    for d in sorted(Path("runs/arch-fair").glob("*/h*")):
        raw_name = d.parent.name
        if raw_name.startswith("smoke") or raw_name.startswith("ref-"):
            continue
        for ev in sorted(d.glob("*/evals/*.evaluation.json")):
            envf = ev.parent.parent / "env.json"
            if not envf.is_file():
                continue
            seed = json.loads(envf.read_text())["seed"]
            if seed not in A_SEEDS:
                continue
            s = json.loads(ev.read_text())["metrics"]["summary"]["acc"]
            by_point.setdefault(raw_name.replace("-s16", ""), {})[seed] = float(
                s["value"] if isinstance(s, dict) else s)
    pts = {}
    for name, seeds in by_point.items():
        fam = name.split("-")[0]
        if fam in fam_label and seeds:
            pts.setdefault(fam, []).append((name, float(np.median(list(seeds.values()))), len(seeds)))

    orig = {
        "t": _load_acc_by_seed(["runs/acc-push/acc-roiv4-transformer/h*/*/evals/*.evaluation.json"]),
        "a": _load_acc_by_seed(["runs/acc-push/acc-roiv4-asformer/h*/*/evals/*.evaluation.json"]),
        "f": _load_acc_by_seed(["runs/acc-push/acc-roiv4-fact/h*/*/evals/*.evaluation.json"]),
        "b": _load_acc_by_seed(["runs/acc-push/acc-roiv4-mstcnbilstm/h*/*/evals/*.evaluation.json"]),
        "m": _load_acc_by_seed(["runs/acc-push/acc-roiv4-mstcn/h*/*/evals/*.evaluation.json"]),
    }
    ref = _load_acc_by_seed(["runs/acc-push/acc-roiv4/h*/*/evals/*.evaluation.json"])
    ref_med = float(np.median([ref[s] for s in A_SEEDS if s in ref]))

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    data = {"reference_acc_median_A8": round(ref_med, 2), "families": {}}
    for i, fam in enumerate(fam_order):
        if fam not in pts:
            continue
        vals = pts[fam]
        xs = [v[1] for v in vals]
        best = max(vals, key=lambda v: v[1])
        ax.scatter(xs, [i] * len(xs), s=34, color="#bbbbbb", zorder=2,
                   label="each config point tried" if fam == fam_order[0] else None)
        ax.scatter([best[1]], [i], s=78, color=C_MODEL, zorder=3, marker="D",
                   label="best point per family" if fam == fam_order[0] else None)
        o = orig.get(fam)
        fam_data = {"best_point": best[0], "best_acc": round(best[1], 2),
                    "n_config_points": len(vals), "all_acc": [round(x, 2) for x in sorted(xs)],
                    "seeds_per_point": sorted({v[2] for v in vals})}
        if o:
            ov = float(np.median([o[s] for s in A_SEEDS if s in o]))
            ax.scatter([ov], [i], s=74, facecolor="white", edgecolor=C_ACCENT, lw=1.8, zorder=4,
                       marker="o", label="the single point in the old report" if fam == fam_order[0] else None)
            fam_data["original_point_acc"] = round(ov, 2)
            best_seeds = by_point[best[0]]
            pair = [best_seeds[s] - o[s] for s in A_SEEDS if s in o and s in best_seeds]
            fam_data["paired_delta_best_minus_orig"] = round(float(np.median(pair)), 2) if pair else None
            fam_data["paired_n"] = len(pair)
        data["families"][fam_label[fam]] = fam_data
    ax.axvline(ref_med, lw=1.6, color=C_BAD)
    ax.set_ylim(-0.6, 7.95)
    ax.text(ref_med + 0.18, 7.62, f"mstcn2 reference {ref_med:.2f}\n(same 8 seeds, roi-grid-v4)",
            fontsize=8.5, color=C_BAD, va="center", ha="left", fontweight="bold")
    ax.set_yticks(range(len(fam_order)))
    ax.set_yticklabels([fam_label[f] for f in fam_order], fontsize=8.5)
    ax.set_xlim(44, 60)
    ax.set_xlabel("test frame accuracy (%), median over seeds", fontsize=9)
    _style(ax, "27 config points / 216 runs: tuning an architecture does not close the gap", "")
    ax.legend(fontsize=8, loc="lower left")
    _save(fig, "fig10_arch_fairness_grid", "Architecture fairness grid (lr x capacity per family)",
          "runs/arch-fair/*（本网格实跑）+ runs/acc-push/acc-roiv4*（参考与原配置点）；"
          "口径见 DEEP_DIVE_11_TOPICS_20261001.md §2⑤⑧(f)",
          dict(data, note="全部读数只用 A 批 seed（42,7,2026,1,2,3,4,5），与参考线同一批，"
                          "故每族内的点、以及与原配置点之间都可逐 seed 配对；补种的 B 批结果见报告 (f4)。"
                          "180 轮同预算对照见同节 (f3)。"))


FIGURES = {
    "fig1_capacity_vs_params": fig1_capacity_vs_params,
    "fig2_architecture_vs_metrics": fig2_architecture_vs_metrics,
    "fig3_feature_contract_gap": fig3_feature_contract_gap,
    "fig4_selection_metric": fig4_selection_metric,
    "fig5_timesfm_probe": fig5_timesfm_probe,
    "fig6_serving_latency": fig6_serving_latency,
    "fig7_image_embed_e0_e1": fig7_image_embed_e0_e1,
    "fig8_acc_vs_action_budget": fig8_acc_vs_action_budget,
    "fig9_per_class_change": fig9_per_class_change,
    "fig10_arch_fairness_grid": fig10_arch_fairness_grid,
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
