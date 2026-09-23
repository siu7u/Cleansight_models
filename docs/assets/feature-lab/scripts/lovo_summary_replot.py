# -*- coding: utf-8 -*-
"""重画 fig3 汇总图：修红框坐标系错位、左标签截断、行间无分隔、图例拥挤。"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(BASE, "report_charts", "lovo_probs")

EDIT = {"fold00":28.57,"fold01":68.42,"fold02":49.21,"fold03":50.00,"fold04":38.71,"fold05":33.33,
        "fold06":15.79,"fold07":37.50,"fold08":41.18,"fold09":71.88,"fold10":32.00,"fold11":54.17,
        "fold12":12.00,"fold13":33.33,"fold14":40.00,"fold15":38.64,"fold16":60.32,"fold17":47.06}
CLS_COLOR = {"idle":"#9aa5b1","water_injection":"#5b8ff9","air_injection":"#5b8ff9","flush":"#61ddaa",
             "long_brush_insert":"#f6bd16","long_brush_withdraw":"#f08bb4","short_brush_cleaning":"#7262fd"}
def c4(n): return CLS_COLOR.get(n, "#b8bec6")

def segments(arr):
    out, s = [], 0
    for i in range(1, len(arr) + 1):
        if i == len(arr) or arr[i] != arr[s]:
            out.append((s, i, int(arr[s]))); s = i
    return out

records = []
for fold in sorted(EDIT):
    d = np.load(os.path.join(NPZ, fold + ".npz"), allow_pickle=True)
    gt, pred = d["gt"], d["probs"].argmax(axis=1)
    records.append((fold, str(d["video"]).split("-")[0], EDIT[fold], gt, pred,
                    [str(n) for n in d["names"]], float((pred == d["gt"]).mean())))
records.sort(key=lambda r: r[2])
maxT = max(len(r[3]) for r in records)
N = len(records)

fig = plt.figure(figsize=(15, 9), dpi=140)
# 左侧标签区独立坐标（axes 分数），主图区从 0.14 起，杜绝截断
ax = fig.add_axes([0.15, 0.10, 0.80, 0.80])

for i, (fold, vid8, ed, gt, pred, names, acc) in enumerate(records):
    y = N - 1 - i
    hl = fold in ("fold12", "fold09")
    # 行交替底色 + 难折淡红底
    if hl:
        ax.axhspan(y - 0.05, y + 0.95, xmin=0, xmax=1, color="#fdecea", zorder=0)
    elif i % 2 == 1:
        ax.axhspan(y - 0.05, y + 0.95, xmin=0, xmax=1, color="#f6f7f9", zorder=0)
    # GT 上条 / 预测下条
    for s, e, ci in segments(gt):
        ax.add_patch(plt.Rectangle((s, y + 0.52), e - s, 0.38, facecolor=c4(names[ci]),
                                   edgecolor="white", lw=0.4, zorder=2))
    for s, e, ci in segments(pred):
        ax.add_patch(plt.Rectangle((s, y + 0.06), e - s, 0.38, facecolor=c4(names[ci]),
                                   edgecolor="white", lw=0.4, zorder=2))
    # 行间细白线
    ax.axhline(y - 0.05, color="white", lw=1.2, zorder=3)
    # 红框：纯 data 坐标，覆盖该行全部
    if hl:
        ax.add_patch(plt.Rectangle((0, y - 0.05), len(gt), 0.98, facecolor="none",
                                   edgecolor="#d64550", lw=2.0, zorder=4, clip_on=False))

# 左标签：画在 axes 坐标 (fig.text)，绝不截断
for i, (fold, vid8, ed, gt, pred, names, acc) in enumerate(records):
    y = N - 1 - i
    hl = fold in ("fold12", "fold09")
    # data y -> axes 分数
    fy = 0.10 + 0.80 * ((y + 0.45) / N)
    fig.text(0.145, fy, f"{fold}  {vid8}", ha="right", va="center", fontsize=9.5,
             color="#c0392b" if hl else "#333", fontweight="bold" if hl else "normal")
    fig.text(0.113, fy, f"{ed:.1f}", ha="right", va="center", fontsize=9.5,
             color="#c0392b" if hl else "#555", fontweight="bold" if hl else "normal")
    fig.text(0.955, fy, f"{acc*100:.0f}%", ha="left", va="center", fontsize=8.5,
             color="#c0392b" if hl else "#888", fontweight="bold" if hl else "normal")

fig.text(0.145, 0.915, "折 / 视频", ha="right", fontsize=10, fontweight="bold", color="#333")
fig.text(0.113, 0.915, "edit", ha="right", fontsize=10, fontweight="bold", color="#333")
fig.text(0.955, 0.915, "帧acc", ha="left", fontsize=10, fontweight="bold", color="#333")

ax.set_xlim(0, maxT); ax.set_ylim(-0.15, N - 0.1)
ax.set_yticks([]); ax.set_xticks([])
ax.set_xlabel("帧号（各折长度不同，条带按各自帧长绘制；上=GT，下=预测）", fontsize=10)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("LOVO 18 折汇总：GT vs 预测段（按 edit 升序）｜红框 = 最难 fold12 (12.0) / 最易 fold09 (71.9)",
             fontsize=13.5, fontweight="bold", loc="left", pad=14)

handles = [plt.Rectangle((0,0),1,1, facecolor=c4(n)) for n in
           ["idle","water_injection","flush","long_brush_insert","long_brush_withdraw","short_brush_cleaning"]]
fig.legend(handles, ["idle","water_injection","flush","long_brush_insert","long_brush_withdraw","short_brush_cleaning"],
           fontsize=9.5, ncol=6, loc="lower center", bbox_to_anchor=(0.53, 0.015), frameon=False)

out = os.path.join(BASE, "report_charts", "fig3_lovo_summary.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved:", out)
