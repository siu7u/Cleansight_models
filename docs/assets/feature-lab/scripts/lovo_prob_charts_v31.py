# -*- coding: utf-8 -*-
"""LOVO v3.1 概率-时间图：叠加 §13.3 双向时长先验（min+max）效果。

上图：6 类 softmax 概率曲线 + GT 背景色带（同 v1）。
下图：三行段条——GT / 原始 argmax 预测 / 时长过滤后预测（P10 最短段长合并），
直接可视化 §11 修复了什么（碎段→整段）。
最短段长取 GT 标签 P10（FEATURE_LAB §11 记录值），与 posthoc_duration.py 口径一致。
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(BASE, "report_charts", "lovo_probs_v31")
OUT = os.path.join(NPZ, "png_v31")
os.makedirs(OUT, exist_ok=True)

EDIT = {"fold00":44.44,"fold01":37.93,"fold02":34.67,"fold03":26.32,"fold04":36.67,"fold05":53.33,
        "fold06":33.33,"fold07":47.62,"fold08":39.29,"fold09":86.96,"fold10":23.91,"fold11":37.61,
        "fold12":42.42,"fold13":57.14,"fold14":34.55,"fold15":60.38,"fold16":57.14}
NAMES = ["idle","water_injection","flush","long_brush_insert","long_brush_withdraw","short_brush_cleaning"]
# P10 最短段长（GT 统计，§11）：与 tmp/posthoc_duration.py 输出一致
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
    """超长段拆分：保留前 mx[c] 帧，余下并给后续段类（末段给前类）。与 posthoc_duration_v2.py 一致。"""
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
    """与 posthoc_duration.py 相同的两阶段合并（同类相并 + 短段并入长邻居）。"""
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

def draw_bands(ax, arr, names, y0, h):
    for s, e, ci in segments(arr):
        ax.add_patch(plt.Rectangle((s, y0), e - s, h, facecolor=c4(names[ci]),
                                   edgecolor="white", lw=0.4))

records = []
for fold in sorted(EDIT):
    d = np.load(os.path.join(NPZ, fold + ".npz"), allow_pickle=True)
    probs, gt = d["probs"], d["gt"]
    names = [str(n) for n in d["names"]]
    ml = [MIN_LEN[names.index(n)] if n in names else 3 for n in NAMES]  # 按 id 对齐
    ml_by_id = [MIN_LEN[names.index(n)] for n in names]
    pred = probs.argmax(axis=1)
    filt = split_long(merge_short(pred, ml_by_id), [MAX_LEN[names.index(n)] if False else MAX_LEN[names.index(n)] for n in names])
    T = len(gt)
    vid8 = str(d["video"]).split("-")[0]
    acc0 = float((pred == gt).mean()); acc1 = float((filt == gt).mean())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), dpi=130,
                                   gridspec_kw={"height_ratios": [2.4, 1]})
    for s, e, ci in segments(gt):
        ax1.axvspan(s, e, color=c4(names[ci]), alpha=0.14, lw=0)
    x = np.arange(T)
    for ci, name in enumerate(names):
        ax1.plot(x, probs[:, ci], color=c4(name), lw=1.3 if name != "idle" else 0.9, alpha=0.95)
    ax1.set_xlim(0, T); ax1.set_ylim(0, 1.02)
    ax1.set_ylabel("P(class | window)")
    ax1.legend([plt.Line2D([],[],color=c4(n),lw=1.5) for n in names], names,
               fontsize=8.5, ncol=6, loc="upper center", framealpha=0.85)
    hl = fold in ("fold12", "fold06")
    ax1.set_title(f"LOVO {fold} · {vid8} · 原始 edit={EDIT[fold]:.2f}（{'碎段型难折' if hl else 'held-out'}）"
                  f"  nodep-226d GRU w=16 ｜ 下图：双向时长过滤(min+max)效果", fontsize=12.5,
                  fontweight="bold", loc="left")
    ax1.spines[["top","right"]].set_visible(False)

    draw_bands(ax2, gt, names, 2.15, 0.72)
    draw_bands(ax2, pred, names, 1.18, 0.72)
    draw_bands(ax2, filt, names, 0.21, 0.72)
    ax2.text(0.995, 2.98, f"帧acc {acc0*100:.0f}% → {acc1*100:.0f}%",
             transform=ax2.get_yaxis_transform(), ha="right", fontsize=9, color="#333")
    ax2.set_xlim(0, T); ax2.set_ylim(0, 3.05)
    ax2.set_yticks([2.5, 1.5, 0.5])
    ax2.set_yticklabels(["GT", "原始预测", "双向过滤后"], fontsize=10)
    ax2.set_xlabel("帧号（7.5fps 采样）")
    ax2.spines[["top","right","left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{fold}_{vid8}_edit{EDIT[fold]:.2f}.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # 过滤后 edit 由 posthoc 报告给出（图标题用原始值）；记录过滤后段数变化
    records.append((fold, vid8, EDIT[fold], sum(1 for _ in segments(pred)), sum(1 for _ in segments(filt))))
    print(fold, "段数", records[-1][3], "->", records[-1][4])

tot0 = sum(r[3] for r in records); tot1 = sum(r[4] for r in records)
print(f"18 折总段数: 原始 {tot0} -> 过滤后 {tot1}（-{100*(1-tot1/tot0):.0f}%）")
print("saved to", OUT)
