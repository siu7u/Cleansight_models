"""时序纵的分段可视化（呈现层，与 ``metrics`` 平级、纵自持）。

把逐帧 GT/预测画成经典 action-segmentation 时间轴：每个视频两条色带（上 GT、下 Pred），
颜色即动作类别，标题带该视频逐帧准确率。这是**只对帧序列输出成立**的呈现，故归时序纵，
不进脊柱；matplotlib 依赖**隔离在本模块**，评估/训练主链路不 import 本模块即不会引入它
（``full_sequence_pipeline`` 以 lazy import 仅在需要出图时触达）。

纯渲染：只吃已算好的 preds/gts + 名称，不做推理、不读 checkpoint——那是流水线的事。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面后端，直接落 PNG
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def render_segmentation(
    preds: list,
    gts: list,
    id2name: dict,
    names: list,
    title_prefix: str,
    out_dir: str | Path,
    base_name: str,
    per_page: int = 6,
) -> list[Path]:
    """渲染逐视频 GT vs 预测 的分段条带图，**每页至多 ``per_page`` 个视频**，返回各页 PNG 路径。

    ``preds[i]`` / ``gts[i]`` 为第 i 个视频的逐帧类别 id（``[T_i]``），与 ``names[i]`` 对齐。
    视频多时单图会又长又难读，故按页切分：写 ``<base_name>-p01.png`` / ``-p02.png`` …
    图内文字用 ASCII（类名本身英文），避免默认字体缺 CJK 字形。
    """
    class_ids = sorted(id2name)
    cmap = ListedColormap(plt.cm.tab10.colors[: len(class_ids)])
    handles = [mpatches.Patch(color=cmap(i), label=f"{cid}:{id2name[cid]}") for i, cid in enumerate(class_ids)]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_page = max(1, per_page)
    n = len(preds)
    pages = (n + per_page - 1) // per_page
    paths: list[Path] = []
    for page, start in enumerate(range(0, n, per_page), start=1):
        cp, cg, cn = preds[start : start + per_page], gts[start : start + per_page], names[start : start + per_page]
        fig, axes = plt.subplots(len(cp), 1, figsize=(12, 1.9 * len(cp) + 1.2), squeeze=False)
        for ax, pred, gt, name in zip(axes[:, 0], cp, cg, cn):
            acc = float((np.asarray(pred) == np.asarray(gt)).mean()) * 100
            strip = np.stack([gt, pred])  # [2, T]：上 GT / 下 Pred
            ax.imshow(strip, aspect="auto", cmap=cmap, vmin=0, vmax=len(class_ids) - 1, interpolation="nearest")
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["GT", "Pred"])
            ax.set_xlabel("sampled frame index")
            ax.set_title(f"{name}  (T={len(gt)}, frame-acc={acc:.1f}%)", fontsize=9, loc="left")

        fig.legend(handles=handles, loc="lower center", ncol=len(class_ids), fontsize=8, frameon=False)
        page_tag = f" (page {page}/{pages})" if pages > 1 else ""
        fig.suptitle(f"{title_prefix} | per-frame segmentation (top=GT / bottom=Pred){page_tag}", fontsize=11)
        fig.tight_layout(rect=(0, 0.06, 1, 0.97))

        out_path = out_dir / f"{base_name}-p{page:02d}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        paths.append(out_path)
    return paths
