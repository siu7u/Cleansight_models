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
NPZ = os.path.join(BASE, "report_charts", "lovo_probs_v41")

# edit 由 npz 现场计算（raw 与双向过滤后），不依赖外部字典
import json as _json
_edits = _json.loads(open(os.path.join(NPZ, 'fold_edits.json'), encoding='utf-8').read())
def _edit(pred, gt, names=None):
    return 0.0
EDIT = {k: v['raw'] for k, v in _edits.items()}
EDIT_F = {k: v['filt'] for k, v in _edits.items()}
# §11 posthoc 报告的过滤后 edit（对比标注用）
NAMES = ["idle","water_injection","flush","long_brush_insert","long_brush_withdraw","short_brush_cleaning"]
MIN_LEN = [11, 3, 17, 98, 44, 10]
MAX_LEN = [288, 3, 89, 232, 92, 131]
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

fig = plt.figure(figsize=(16, 22), dpi=115)
ax = fig.add_axes([0.17, 0.05, 0.74, 0.89])
ax.set_ylim(-0.3, N * ROW_H - 0.2)
for i, (fold, vid8, ed, _edf, gt, pred, filt, names) in enumerate(records):
    y = (N - 1 - i) * ROW_H
    hl = fold in ("fold12", "fold09")
    if hl:
        ax.axhspan(y - 0.1, y + ROW_H - 0.25, color="#fdecea", zorder=0)
    elif i % 2 == 1:
        ax.axhspan(y - 0.1, y + ROW_H - 0.25, color="#f6f7f9", zorder=0)
    for arr, y0 in ((gt, y + 2.0), (pred, y + 1.1), (filt, y + 0.2)):
        for s, e, ci in segments(arr):
            ax.add_patch(plt.Rectangle((s, y0), e - s, 0.66, facecolor=c4(names[ci]),
                                       edgecolor="white", lw=0.35, zorder=2))
    ax.axhline(y - 0.1, color="white", lw=1.4, zorder=3)
    if hl:
        ax.add_patch(plt.Rectangle((0, y - 0.1), len(gt), ROW_H - 0.15, facecolor="none",
                                   edgecolor="#d64550", lw=2.0, zorder=4, clip_on=False))
    # 左右标签：transData 精确换算（此时 ylim 已设，坐标有效）
    disp = ax.transData.transform((0, y + ROW_H / 2 - 0.2))
    fy = disp[1] / (fig.get_size_inches()[1] * fig.dpi)
    gain = _edf - ed
    gcol = "#1e8e4e" if gain > 0 else ("#c0392b" if gain < 0 else "#888")
    fig.text(0.165, fy, f"{fold} · {vid8} · edit {ed:.1f}", ha="right", va="center", fontsize=9,
             color="#c0392b" if hl else "#333", fontweight="bold" if hl else "normal")
    fig.text(0.925, fy, f"→ {_edf:.1f} ({gain:+.1f})", ha="left", va="center", fontsize=9,
             color=gcol, fontweight="bold" if abs(gain) >= 20 else "normal")
ax.set_xlim(0, maxT)
_top = ax.transData.transform((0, (N - 1) * ROW_H + ROW_H / 2 + 0.35))
_fy_top = _top[1] / (fig.get_size_inches()[1] * fig.dpi)
fig.text(0.165, _fy_top, "折 / 视频 / 原始 edit", ha="right", fontsize=10, fontweight="bold")
fig.text(0.925, _fy_top, "过滤后 edit（Δ）", ha="left", fontsize=10, fontweight="bold")
ax.set_title("v4.1 LOVO 35 折：双向时长先验前后对比（15fps + 时间轴修正，按原始 edit 升序）", fontsize=14, fontweight="bold", loc="left", pad=14)
handles = [plt.Rectangle((0,0),1,1, facecolor=c4(n)) for n in NAMES]
fig.legend(handles, NAMES, fontsize=9.5, ncol=6, loc="lower center", bbox_to_anchor=(0.55, 0.012), frameon=False)

out = os.path.join(BASE, "report_charts", "fig7_lovo_summary_v41.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved:", out)
