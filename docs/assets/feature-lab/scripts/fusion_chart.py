# -*- coding: utf-8 -*-
"""融合探针可视化：左=策略柱状图，右=逐类 F1 对比。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=140)

# 左：策略柱状图
strategies = ["GRU\nonly", "TCN\nonly", "Prob\navg", "0.7G\n+0.3T", "Max\nconf", "Class\nroute"]
edits = [46.34, 39.16, 43.37, 42.58, 43.28, 49.14]
f25s = [44.65, 20.26, 38.06, 38.98, 35.34, 45.66]
x = np.arange(len(strategies))
w = 0.35
bars1 = ax1.bar(x - w/2, edits, w, label="edit", color="#5b8ff9", edgecolor="white")
bars2 = ax1.bar(x + w/2, f25s, w, label="F1@0.25", color="#61ddaa", edgecolor="white")
# 高亮 Class route
bars1[-1].set_color("#d64550"); bars2[-1].set_color("#d64550")
for b in bars1: ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{b.get_height():.1f}", ha="center", fontsize=8.5)
for b in bars2: ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{b.get_height():.1f}", ha="center", fontsize=8.5)
ax1.set_xticks(x); ax1.set_xticklabels(strategies, fontsize=9)
ax1.set_ylabel("test 分数"); ax1.set_ylim(0, 58)
ax1.legend(fontsize=10); ax1.spines[["top","right"]].set_visible(False)
ax1.set_title("融合策略对比（Class route 唯一胜出）", fontsize=12, fontweight="bold", loc="left")

# 右：逐类 F1
classes = ["idle", "flush", "lb_insert", "lb_withdraw", "sb_clean"]
gru_f1 = [0.734, 0.118, 0.774, 0.474, 0.286]
tcn_f1 = [0.758, 0.602, 0.791, 0.537, 0.370]
route_f1 = [0.739, 0.419, 0.774, 0.483, 0.323]
x2 = np.arange(len(classes))
w2 = 0.25
ax2.bar(x2 - w2, gru_f1, w2, label="GRU", color="#5b8ff9", edgecolor="white")
ax2.bar(x2, tcn_f1, w2, label="TCN", color="#f6bd16", edgecolor="white")
ax2.bar(x2 + w2, route_f1, w2, label="Class route", color="#d64550", edgecolor="white")
for xi, g, t, r in zip(x2, gru_f1, tcn_f1, route_f1):
    for v, off in [(g, -w2), (t, 0), (r, w2)]:
        ax2.text(xi + off, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5)
ax2.set_xticks(x2); ax2.set_xticklabels(classes, fontsize=9.5)
ax2.set_ylabel("帧级 F1"); ax2.set_ylim(0, 0.90)
ax2.legend(fontsize=9.5, loc="upper right")
ax2.spines[["top","right"]].set_visible(False)
ax2.set_title("逐类 F1：TCN 在 flush 上 5.1 倍领先 GRU", fontsize=12, fontweight="bold", loc="left")

fig.tight_layout()
fig.savefig("tmp/report_charts/fig8_fusion_probe.png", bbox_inches="tight", facecolor="white")
print("fig8 saved")
