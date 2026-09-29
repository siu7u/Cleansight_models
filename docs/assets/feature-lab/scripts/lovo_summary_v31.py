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
NPZ = os.path.join(BASE, "report_charts", "lovo_probs_v31")

EDIT = {"fold00":44.44,"fold01":37.93,"fold02":34.67,"fold03":26.32,"fold04":36.67,"fold05":53.33,
        "fold06":33.33,"fold07":47.62,"fold08":39.29,"fold09":86.96,"fold10":23.91,"fold11":37.61,
        "fold12":42.42,"fold13":57.14,"fold14":34.55,"fold15":60.38,"fold16":57.14}
# §11 posthoc 报告的过滤后 edit（对比标注用）
EDIT_F = {}  # 由 posthoc_v31_folds.txt 注入
NAMES = ["idle","water_injection","flush","long_brush_insert","long_brush_withdraw","short_brush_cleaning"]
MIN_LEN = [5, 3, 7, 28, 14, 3]
MAX_LEN = [212, 3, 36, 93, 41, 44]
CLS_COLOR = {"idle":"#9aa5b1","water_injection":"#5b8ff9","air_injection":"#5b8ff9","flush":"#61ddaa",
             "long_brush_insert":"#f6bd16","long_brush_withdraw":"#f08bb4","short_brush_cleaning":"#7262fd"}
def c4(n): return CLS_COLOR.get(n, "#b8bec6")

def segments(arr):
    out, s = [], 0
    for i in range(1, len(arr) + 1):
        if i == len(arr) or arr[i] != arr[s]:
            out.append((s, i, int(arr[s]))); s = i
    return out

def split_long(pred, mx):
    segs, s = [], 0
    a = pred.tolist()
    for i in range(1, len(a) + 1):
        if i == len(a) or a[i] != a[s]:
            segs.append([s, i, a[s]]); s = i
    for j in range(len(segs)):
        st, en, c = segs[j]
        if en - st <= mx[c]:
            continue
        nc = segs[j + 1][2] if j < len(segs) - 1 else segs[j - 1][2]
        segs[j] = [st, st + mx[c], c]
        segs.insert(j + 1, [st + mx[c], en, nc])
    out = []
    for st, en, c in segs:
        out.extend([c] * (en - st))
    return np.array(out, dtype=pred.dtype)

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

import json, re
pf = os.path.join(BASE, 'report_charts', 'lovo_probs_v31', 'posthoc_v31_folds.txt')
if os.path.exists(pf) and not EDIT_F:
    for ln in open(pf, encoding="utf-8"):
        m = re.match(r'(fold\d\d): edit ([\d.]+)', ln)
        if m:
            EDIT_F[m.group(1)] = float(m.group(2))
records = []
for fold in sorted(EDIT):
    d = np.load(os.path.join(NPZ, fold + ".npz"), allow_pickle=True)
    names = [str(n) for n in d["names"]]
    gt = d["gt"]; pred = d["probs"].argmax(axis=1)
    filt = merge_short(pred, [MIN_LEN[names.index(n)] for n in names])
    records.append((fold, str(d["video"]).split("-")[0], EDIT[fold], EDIT_F.get(fold, EDIT[fold]), gt, pred, filt, names))
records.sort(key=lambda r: r[2])
maxT = max(len(r[4]) for r in records)
N = len(records)
ROW_H = 3.0  # 每折三行

fig = plt.figure(figsize=(16, 12), dpi=135)
ax = fig.add_axes([0.17, 0.09, 0.74, 0.82])

for i, (fold, vid8, ed, _edf, gt, pred, filt, names) in enumerate(records):
    y = (N - 1 - i) * ROW_H
    hl = fold in ("fold12", "fold09")
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
    gain = _edf - ed
    gcol = "#1e8e4e" if gain > 0 else ("#c0392b" if gain < 0 else "#888")
    fig.text(0.165, fy, f"{fold} · {vid8} · edit {ed:.1f}", ha="right", va="center", fontsize=9,
             color="#c0392b" if hl else "#333", fontweight="bold" if hl else "normal")
    fig.text(0.925, fy, f"→ {_edf:.1f} ({gain:+.1f})", ha="left", va="center", fontsize=9,
             color=gcol, fontweight="bold" if abs(gain) >= 20 else "normal")

fig.text(0.165, 0.925, "折 / 视频 / 原始 edit", ha="right", fontsize=10, fontweight="bold")
fig.text(0.925, 0.925, "过滤后 edit（Δ）", ha="left", fontsize=10, fontweight="bold")
ax.set_xlim(0, maxT)
ax.set_ylim(-0.3, N * ROW_H - 0.2)
ax.set_yticks([]); ax.set_xticks([])
ax.set_xlabel("帧号（各折长度不同；每折三行 = GT / 原始预测 / 时长过滤后）", fontsize=10.5)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title("LOVO 18 折汇总 v2：时长先验过滤（P10）前后对比，按原始 edit 升序｜红框 = fold12(c1367d51, 重标修复视频)/fold09(789d58df, 最易折)",
             fontsize=14, fontweight="bold", loc="left", pad=14)
handles = [plt.Rectangle((0,0),1,1, facecolor=c4(n)) for n in NAMES]
fig.legend(handles, NAMES, fontsize=9.5, ncol=6, loc="lower center", bbox_to_anchor=(0.55, 0.012), frameon=False)

out = os.path.join(BASE, "report_charts", "fig5_lovo_summary_v31.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved:", out)
