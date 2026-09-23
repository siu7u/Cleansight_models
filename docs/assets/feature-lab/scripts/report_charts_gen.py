# -*- coding: utf-8 -*-
"""feature-lab 已完成实验的准确率可视化（数据源：docs/FEATURE_LAB.md §6-§10，2 张 PNG）。

口径：3-seed 中位数（除窗口实验 w32-test n=2 / w64-test n=1、LOVO 单折外）。
指标均为越高越好：edit = 段级编辑相似度；F1@0.25 = 边界容差 25% 帧的段级 F1。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_charts")
os.makedirs(OUT, exist_ok=True)

C_BASE = "#b8bec6"; C_BLUE = "#5b8ff9"; C_GREEN = "#61ddaa"; C_YELLOW = "#f6bd16"
C_RED = "#d64550"; C_PURPLE = "#7262fd"; C_DARK = "#34495e"

# ============ 图 1：特征方案准确率总览 ============
fig = plt.figure(figsize=(16, 9), dpi=150)
fig.suptitle("feature-lab 准确率总览：特征方案演进与 val/test 双口径（GRU · 3-seed 中位数 · CPU exploratory）",
             fontsize=16, fontweight="bold")

# ---- 面板 A：特征方案演进（val edit）----
axA = fig.add_axes([0.05, 0.58, 0.42, 0.32])
schemes = ["hand-40", "global-hand-80", "bbox-40", "roi-144", "clean-v2", "clean-v3", "v2⊕v3", "v4", "roi-v4", "nodep"]
edit_val = [15.81, 17.06, 18.53, 24.83, 23.76, 31.07, 33.85, 29.05, 26.36, 32.32]
colors = [C_BASE]*4 + [C_BLUE]*3 + [C_YELLOW]*2 + [C_GREEN]
bars = axA.bar(range(len(schemes)), edit_val, 0.62, color=colors, edgecolor="white")
for i, v in enumerate(edit_val):
    axA.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9.5,
             fontweight="bold" if schemes[i] in ("v2⊕v3", "nodep") else "normal")
axA.set_xticks(range(len(schemes)))
axA.set_xticklabels(schemes, rotation=38, ha="right", fontsize=9.5)
axA.set_ylabel("val edit（段级编辑相似度）")
axA.set_ylim(0, 40)
axA.spines[["top", "right"]].set_visible(False)
axA.set_title("① 特征方案演进（val 口径）：v2⊕v3 33.85 纪录 → nodep 32.32 定版候选", fontsize=12.5, fontweight="bold", loc="left")
axA.text(0.99, 0.02, "灰=基线 §6  蓝=v2/v3 系  黄=v4 系  绿=nodep 定版候选 §8.4", transform=axA.transAxes, ha="right", fontsize=8.5, color="#666")

# ---- 面板 B：val vs test · edit ----
axB = fig.add_axes([0.55, 0.58, 0.42, 0.32])
five = ["v3", "v2⊕v3", "v4", "roi-v4", "nodep"]
ed_val = [31.07, 33.85, 29.05, 26.36, 32.32]
ed_test = [29.62, 25.72, 29.19, 21.02, 31.54]
x = np.arange(5); w = 0.36
axB.bar(x - w/2, ed_val, w, label="val（固定 4 视频判据口径）", color=C_BLUE, edgecolor="white")
axB.bar(x + w/2, ed_test, w, label="test（project-18 跨批次专项）", color=C_GREEN, edgecolor="white")
for xi, v in zip(x - w/2, ed_val): axB.text(xi, v + 0.4, f"{v:.1f}", ha="center", fontsize=9)
for xi, v in zip(x + w/2, ed_test): axB.text(xi, v + 0.4, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
axB.set_xticks(x); axB.set_xticklabels(five, fontsize=10.5)
axB.set_ylabel("edit"); axB.set_ylim(0, 42)
axB.legend(fontsize=9, loc="upper right"); axB.spines[["top", "right"]].set_visible(False)
axB.set_title("② edit 双口径：v2⊕v3 val 虚高（废弃通道过拟合），nodep test 31.54 全场第一（+5.8）", fontsize=12.5, fontweight="bold", loc="left")

# ---- 面板 C：val vs test · F1@0.25 ----
axC = fig.add_axes([0.08, 0.09, 0.38, 0.36])
f_val = [18.52, 27.03, 16.16, 17.48, 19.42]
f_test = [27.96, 26.09, 30.61, 22.47, 28.00]
axC.bar(x - w/2, f_val, w, label="val", color=C_BLUE, edgecolor="white")
axC.bar(x + w/2, f_test, w, label="test", color=C_GREEN, edgecolor="white")
for xi, v in zip(x - w/2, f_val): axC.text(xi, v + 0.4, f"{v:.1f}", ha="center", fontsize=9)
for xi, v in zip(x + w/2, f_test): axC.text(xi, v + 0.4, f"{v:.1f}", ha="center", fontsize=9, fontweight="bold")
axC.set_xticks(x); axC.set_xticklabels(five, fontsize=10.5)
axC.set_ylabel("F1@0.25（边界容差段级 F1）"); axC.set_ylim(0, 36)
axC.legend(fontsize=9, loc="upper left"); axC.spines[["top", "right"]].set_visible(False)
axC.set_title("③ F1@0.25 双口径：v4 在 test 大幅领先 30.61，nodep 28.00 稳居第二", fontsize=12.5, fontweight="bold", loc="left")

# ---- 面板 D：LOVO 18 折分布 ----
axD = fig.add_axes([0.56, 0.09, 0.40, 0.36])
folds = [28.57, 68.42, 49.21, 50.00, 38.71, 33.33, 15.79, 37.50, 41.18, 71.88, 32.00, 54.17, 12.00, 33.33, 40.00, 38.64, 60.32, 47.06]
order = np.argsort(folds)
sortedv = [folds[i] for i in order]
cols = [C_RED if v < 20 else (C_GREEN if v > 55 else C_BLUE) for v in sortedv]
axD.bar(range(18), sortedv, 0.65, color=cols, edgecolor="white")
axD.axhline(41.78, color=C_DARK, lw=1.4, ls="--"); axD.text(17.4, 42.3, "mean 41.78", ha="right", fontsize=9, color=C_DARK)
axD.axhline(39.36, color=C_PURPLE, lw=1.4, ls="-."); axD.text(0.2, 36.8, "median 39.36", fontsize=9, color=C_PURPLE)
axD.axhline(32.32, color=C_YELLOW, lw=1.6, ls=":")
axD.text(0.2, 29.5, "固定 val 32.32（低估约 7 点）", fontsize=9, color="#a07a00")
for i, v in enumerate(sortedv):
    if v < 16 or v > 71: axD.text(i, v + 0.8, f"{v:.1f}", ha="center", fontsize=8.5, color="#333")
axD.set_xlabel("18 折（按难度排序）"); axD.set_ylabel("LOVO val edit")
axD.set_ylim(0, 78); axD.spines[["top", "right"]].set_visible(False)
axD.set_title("④ LOVO 交叉验证（nodep，§9）：极差 12.0~71.9，视频难度是最大方差源", fontsize=12.5, fontweight="bold", loc="left")

fig.savefig(os.path.join(OUT, "fig1_overview.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)

# ============ 图 2：窗口实验 + 逐类 F1 ============
fig = plt.figure(figsize=(16, 6.5), dpi=150)
fig.suptitle("feature-lab 深入分析：窗口长度实验（§10）与 nodep 增益逐类分解（§8.5）",
             fontsize=15, fontweight="bold")

# ---- 面板 E：窗口实验 ----
axE = fig.add_axes([0.06, 0.14, 0.36, 0.66])
wins = [16, 32, 64]
w_test = [25.72, 23.48, 21.98]
axE.plot(wins, w_test, "o-", color=C_GREEN, lw=2.2, ms=8, label="test edit（v2⊕v3）")
for xi, v, n in zip(wins, w_test, ["n=3", "n=2", "n=1"]):
    axE.annotate(f"{v:.2f}\n({n})", (xi, v), textcoords="offset points", xytext=(8, 8), fontsize=9.5)
axE.plot([32, 64], [33.28, 35.66], "s--", color=C_BLUE, lw=1.8, ms=7, label="val edit（w16 基准点 checkpoint 丢失未补）")
axE.annotate("33.28", (32, 33.28), textcoords="offset points", xytext=(8, 6), fontsize=9.5, color=C_BLUE)
axE.annotate("35.66", (64, 35.66), textcoords="offset points", xytext=(-14, 8), fontsize=9.5, color=C_BLUE)
axE.axhline(25.72, color="#999", lw=1, ls=":")
axE.set_xticks(wins); axE.set_xticklabels(["window 16\n（现行）", "window 32", "window 64"])
axE.set_ylabel("edit"); axE.set_ylim(19, 39)
axE.legend(fontsize=9.5, loc="upper left"); axE.spines[["top", "right"]].set_visible(False)
axE.set_title("⑤ 窗口长度：test 单调变差 → 上下文瓶颈假设证伪，窗口维持 16", fontsize=12, fontweight="bold", loc="left")

# ---- 面板 F：逐类 F1 ----
axF = fig.add_axes([0.55, 0.14, 0.40, 0.66])
cls = ["idle", "flush", "lb_insert", "lb_withdraw", "sb_clean"]
f_nodep = [0.680, 0.100, 0.171, 0.295, 0.000]
f_v2v3 = [0.704, None, 0.238, 0.216, 0.000]
f_v3 = [0.704, None, 0.211, 0.279, 0.000]
xc = np.arange(5); wc = 0.27
def bars_with_na(ax, off, vals, color, label):
    xs = [xc[i] + off for i, v in enumerate(vals) if v is not None]
    vs = [v for v in vals if v is not None]
    ax.bar(xs, vs, wc, color=color, label=label, edgecolor="white")
    for xi, v in zip(xs, vs): ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5)
bars_with_na(axF, -wc, f_nodep, C_GREEN, "nodep")
bars_with_na(axF, 0, f_v2v3, C_BLUE, "v2⊕v3")
bars_with_na(axF, +wc, f_v3, C_YELLOW, "v3")
axF.text(1, 0.02, "n/a", ha="center", fontsize=9, color="#999")  # flush 仅 nodep 非零口径
axF.set_xticks(xc)
axF.set_xticklabels(["idle", "flush", "long_brush\n_insert", "long_brush\n_withdraw", "short_brush\n_cleaning"], fontsize=9.5)
axF.set_ylabel("逐类 F1（中位数，n=3，test）"); axF.set_ylim(0, 0.82)
axF.legend(fontsize=9.5, loc="upper right"); axF.spines[["top", "right"]].set_visible(False)
axF.set_title("⑥ nodep +5.8 test edit 增益来源：withdraw +36%（P0 对判别）", fontsize=12, fontweight="bold", loc="left")
axF.text(0.99, 0.02, "flush 仅 nodep 报告 0.10；sb_cleaning 全方案为 0（数据不足）", transform=axF.transAxes, ha="right", fontsize=8.5, color="#666")

fig.savefig(os.path.join(OUT, "fig2_analysis.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print("saved:", os.listdir(OUT))
