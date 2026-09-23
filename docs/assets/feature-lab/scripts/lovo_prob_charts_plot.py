# -*- coding: utf-8 -*-
"""LOVO 各折 held-out 视频的"各标签概率-时间"图（读 lovo_probs/*.npz）。

单图（每折一张）：上=6 类 softmax 概率曲线 + GT 段背景色带；下=GT vs 预测(argmax) 段条。
汇总图：18 折按 edit 排序的段条缩略，红框标最难(fold12, 12.0)与最易(fold09, 71.9)。
概率为逐窗末帧原始 softmax（无 causal_decision 平滑），冷启动前 15 帧置 idle=1。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(BASE, "report_charts", "lovo_probs")
PNG = os.path.join(NPZ, "png")
os.makedirs(PNG, exist_ok=True)

EDIT = {"fold00":28.57,"fold01":68.42,"fold02":49.21,"fold03":50.00,"fold04":38.71,"fold05":33.33,
        "fold06":15.79,"fold07":37.50,"fold08":41.18,"fold09":71.88,"fold10":32.00,"fold11":54.17,
        "fold12":12.00,"fold13":33.33,"fold14":40.00,"fold15":38.64,"fold16":60.32,"fold17":47.06}

CLS_COLOR = {"idle":"#9aa5b1","water_injection":"#5b8ff9","air_injection":"#5b8ff9","flush":"#61ddaa",
             "long_brush_insert":"#f6bd16","long_brush_withdraw":"#f08bb4","short_brush_cleaning":"#7262fd"}
def c4(name): return CLS_COLOR.get(name, "#b8bec6")

def segments(arr):
    """把类别 id 序列切成 [(start, end, cls_id)] 段。"""
    out, s = [], 0
    for i in range(1, len(arr) + 1):
        if i == len(arr) or arr[i] != arr[s]:
            out.append((s, i, int(arr[s]))); s = i
    return out

def draw_bands(ax, arr, names, y0, h, alpha=0.9, labels=False):
    for s, e, ci in segments(arr):
        ax.add_patch(plt.Rectangle((s, y0), e - s, h, facecolor=c4(names[ci]),
                                   edgecolor="white", lw=0.4, alpha=alpha))

records = []
for fold in sorted(EDIT):
    d = np.load(os.path.join(NPZ, fold + ".npz"), allow_pickle=True)
    probs, gt, names = d["probs"], d["gt"], [str(n) for n in d["names"]]
    video, win = str(d["video"]), int(d["window"])
    pred = probs.argmax(axis=1)
    T = len(gt)
    vid8 = video.split("-")[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6.4), dpi=130,
                                   gridspec_kw={"height_ratios": [2.6, 1]})
    # GT 段背景色带
    for s, e, ci in segments(gt):
        ax1.axvspan(s, e, color=c4(names[ci]), alpha=0.14, lw=0)
    x = np.arange(T)
    for ci, name in enumerate(names):
        ax1.plot(x, probs[:, ci], color=c4(name), lw=1.3 if name != "idle" else 0.9,
                 label=name, alpha=0.95)
    ax1.set_xlim(0, T); ax1.set_ylim(0, 1.02)
    ax1.set_ylabel("P(class | window)")
    ax1.legend(fontsize=8.5, ncol=6, loc="upper center", framealpha=0.85)
    ax1.set_title(f"LOVO {fold} · {vid8} · edit={EDIT[fold]:.2f}（{'最难' if fold=='fold12' else '最易' if fold=='fold09' else 'held-out'}）"
                  f"  nodep-226d GRU w={win}", fontsize=12.5, fontweight="bold", loc="left")
    ax1.spines[["top","right"]].set_visible(False)

    # GT vs 预测段条
    draw_bands(ax2, gt, names, 1.15, 0.75)
    draw_bands(ax2, pred, names, 0.15, 0.75)
    acc = float((pred == gt).mean())
    ax2.text(0.995, 1.98, f"帧准确率 {acc*100:.1f}%", transform=ax2.get_yaxis_transform(),
             ha="right", fontsize=9, color="#333")
    ax2.set_xlim(0, T); ax2.set_ylim(0, 2.05)
    ax2.set_yticks([1.5, 0.5]); ax2.set_yticklabels(["GT", "预测"], fontsize=10)
    ax2.set_xlabel("帧号（7.5fps 采样）")
    ax2.spines[["top","right","left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(PNG, f"{fold}_{vid8}_edit{EDIT[fold]:.2f}.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    records.append((fold, vid8, EDIT[fold], gt, pred, names, acc))
    print(fold, "saved, acc=", round(acc, 3))

# ---------- 汇总图：18 折按 edit 排序 ----------
records.sort(key=lambda r: r[2])
fig, ax = plt.subplots(figsize=(14, 8.6), dpi=140)
for i, (fold, vid8, ed, gt, pred, names, acc) in enumerate(records):
    y = len(records) - 1 - i
    draw_bands(ax, gt, names, y + 0.52, 0.40)
    draw_bands(ax, pred, names, y + 0.04, 0.40)
    hl = (fold in ("fold12", "fold09"))
    ax.text(-0.012, y + 0.5, f"{fold} {vid8}  edit={ed:.1f}", ha="right", va="center",
            fontsize=9, color="#c0392b" if hl else "#333",
            fontweight="bold" if hl else "normal", transform=ax.get_yaxis_transform())
    ax.text(1.008, y + 0.5, f"帧acc {acc*100:.0f}%", ha="left", va="center", fontsize=8,
            color="#666", transform=ax.get_yaxis_transform())
    if hl:
        ax.add_patch(plt.Rectangle((0, y - 0.02), 1, 1.0, transform=ax.get_xaxis_transform(),
                                   facecolor="none", edgecolor="#d64550", lw=1.8, clip_on=False))
ax.set_xlim(0, max(len(r[3]) for r in records))
ax.set_ylim(-0.3, len(records) - 0.1)
ax.set_yticks([])
ax.set_xlabel("帧号（上=GT，下=预测；按折内帧号，各折长度不同）")
ax.set_title("LOVO 18 折汇总：GT vs 预测段（按 edit 升序）——红框 = 最难 fold12 (12.0) / 最易 fold09 (71.9)",
             fontsize=13.5, fontweight="bold", loc="left")
handles = [plt.Rectangle((0,0),1,1, facecolor=c4(n)) for n in ["idle","water_injection","flush",
          "long_brush_insert","long_brush_withdraw","short_brush_cleaning"]]
ax.legend(handles, ["idle","water_injection","flush","long_brush_insert","long_brush_withdraw","short_brush_cleaning"],
          fontsize=9, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.075))
for s in ax.spines.values(): s.set_visible(False)
fig.savefig(os.path.join(BASE, "report_charts", "fig3_lovo_summary.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)
print("summary saved")
