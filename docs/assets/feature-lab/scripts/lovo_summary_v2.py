# -*- coding: utf-8 -*-
"""v2 汇总图：18 折 × 三行段条（GT/原始/时长过滤后），按原始 edit 升序。
红框 = 过滤受益最大的难折 fold12/fold06；行尾标注过滤后帧acc。"""
import os
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
# §11 posthoc 报告的过滤后 edit（对比标注用）
EDIT_F = {"fold00":33.33,"fold01":41.18,"fold02":60.61,"fold03":40.00,"fold04":35.71,"fold05":50.00,
          "fold06":60.00,"fold07":63.64,"fold08":59.09,"fold09":47.83,"fold10":41.67,"fold11":44.90,
          "fold12":60.00,"fold13":33.33,"fold14":66.67,"fold15":58.33,"fold16":50.00,"fold17":23.53}
NAMES = ["idle","water_injection","flush","long_brush_insert","long_brush_withdraw","short_brush_cleaning"]
MIN_LEN = [5, 20, 7, 28, 14, 4]
CLS_COLOR = {"idle":"#9aa5b1","water_injection":"#5b8ff9","air_injection":"#5b8ff9","flush":"#61ddaa",
             "long_brush_insert":"#f6bd16","long_brush_withdraw":"#f08bb4","short_brush_cleaning":"#7262fd"}
def c4(n): return CLS_COLOR.get(n, "#b8bec6")

def segments(arr):
    out, s = [], 0
    for i in range(1, len(arr) + 1):
        if i == len(arr) or arr[i] != arr[s]:
            out.append((s, i, int(arr[s]))); s = i
    return out

def merge_short(pred, min_len):
    a = pred.tolist()
    segs, s = [], 0
    for i in range(1, len(a) + 1):
        if i == len(a) or a[i] != a[s]:
            segs.append([s, i, a[s]]); s = i
    changed = True
    while changed:
        changed = False
        j = 0
        while j < len(segs) - 1:
            if segs[j][2] == segs[j + 1][2]:
                segs[j][1] = segs[j + 1][1]; del segs[j + 1]; changed = True
            else:
                j += 1
    changed, guard = True, 0
    while changed and guard < 200:
        changed = False; guard += 1
        for j, (st, en, c) in enumerate(segs):
            if en - st >= min_len[c]:
                continue
            left = segs[j - 1] if j > 0 else None
            right = segs[j + 1] if j < len(segs) - 1 else None
            if left and (not right or (left[1] - left[0]) >= (right[1] - right[0])):
                segs[j - 1] = [left[0], en, left[2]]
            elif right:
                segs[j + 1] = [st, right[1], right[2]]
            else:
                continue
            del segs[j]; changed = True; break
    out = []
    for st, en, c in segs:
        out.extend([c] * (en - st))
    return np.array(out, dtype=pred.dtype)

records = []
for fold in sorted(EDIT):
    d = np.load(os.path.join(NPZ, fold + ".npz"), allow_pickle=True)
    names = [str(n) for n in d["names"]]
    gt = d["gt"]; pred = d["probs"].argmax(axis=1)
    filt = merge_short(pred, [MIN_LEN[names.index(n)] for n in names])
    records.append((fold, str(d["video"]).split("-")[0], EDIT[fold], EDIT_F[fold], gt, pred, filt, names))
records.sort(key=lambda r: r[2])
maxT = max(len(r[4]) for r in records)
N = len(records)
ROW_H = 3.0  # 每折三行

fig = plt.figure(figsize=(16, 12), dpi=135)
ax = fig.add_axes([0.17, 0.09, 0.74, 0.82])

for i, (fold, vid8, ed, edf, gt, pred, filt, names) in enumerate(records):
    y = (N - 1 - i) * ROW_H
    hl = fold in ("fold12", "fold06")
    if hl:
        ax.axhspan(y - 0.1, y + ROW_H - 0.25, color="#fdecea", zorder=0)
    elif i % 2 == 1:
        ax.axhspan(y - 0.1, y + ROW_H - 0.25, color="#f6f7f9", zorder=0)
    # 三行：GT / 原始 / 过滤后
    for arr, y0 in ((gt, y + 2.0), (pred, y + 1.1), (filt, y + 0.2)):
        for s, e, ci in segments(arr):
            ax.add_patch(plt.Rectangle((s, y0), e - s, 0.66, facecolor=c4(names[ci]),
                                       edgecolor="white", lw=0.35, zorder=2))
    ax.axhline(y - 0.1, color="white", lw=1.4, zorder=3)
    if hl:
        ax.add_patch(plt.Rectangle((0, y - 0.1), len(gt), ROW_H - 0.15, facecolor="none",
                                   edgecolor="#d64550", lw=2.0, zorder=4, clip_on=False))
    # 左右标签（fig.text，防截断/重叠）
    fy = 0.09 + 0.82 * ((y + ROW_H / 2 - 0.2) / (N * ROW_H))
    gain = edf - ed
    gcol = "#1e8e4e" if gain > 0 else ("#c0392b" if gain < 0 else "#888")
    fig.text(0.165, fy, f"{fold} · {vid8} · edit {ed:.1f}", ha="right", va="center", fontsize=9,
             color="#c0392b" if hl else "#333", fontweight="bold" if hl else "normal")
    fig.text(0.925, fy, f"→ {edf:.1f} ({gain:+.1f})", ha="left", va="center", fontsize=9,
             color=gcol, fontweight="bold" if abs(gain) >= 20 else "normal")

fig.text(0.165, 0.925, "折 / 视频 / 原始 edit", ha="right", fontsize=10, fontweight="bold")
fig.text(0.925, 0.925, "过滤后 edit（Δ）", ha="left", fontsize=10, fontweight="bold")
ax.set_xlim(0, maxT)
ax.set_ylim(-0.3, N * ROW_H - 0.2)
ax.set_yticks([]); ax.set_xticks([])
ax.set_xlabel("帧号（各折长度不同；每折三行 = GT / 原始预测 / 时长过滤后）", fontsize=10.5)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("LOVO 18 折汇总 v2：时长先验过滤（P10）前后对比，按原始 edit 升序｜红框 = 过滤受益最大的碎段难折 fold12/fold06",
             fontsize=14, fontweight="bold", loc="left", pad=14)
handles = [plt.Rectangle((0,0),1,1, facecolor=c4(n)) for n in NAMES]
fig.legend(handles, NAMES, fontsize=9.5, ncol=6, loc="lower center", bbox_to_anchor=(0.55, 0.012), frameon=False)

out = os.path.join(BASE, "report_charts", "fig4_lovo_summary_v2.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved:", out)
