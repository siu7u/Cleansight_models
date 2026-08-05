"""benchmark 时序分段可视化。

把逐帧 GT/预测画成经典 action-segmentation 时间轴：每个视频两条色带（上 GT、下 Pred），
颜色即动作类别，标题带该视频逐帧准确率。该呈现只消费 framework 已产生的
``PredictionOutput``，不进入模型训练或推理实现；matplotlib 依赖隔离在本模块并按需导入。

纯渲染：只吃已经产生的 ``PredictionOutput`` 或 preds/gts，不做推理、不读 checkpoint。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面后端，直接落 PNG
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def _value(output, name: str, default=None):
    """兼容 ``PredictionOutput`` 和等字段 mapping，保持呈现层不绑定具体实现类型。"""

    if isinstance(output, dict):
        return output.get(name, default)
    return getattr(output, name, default)


def render_prediction_timeline(
    output,
    *,
    out_dir: str | Path,
    per_page: int = 6,
) -> list[Path]:
    """直接从一次时序预测事实渲染测试 timeline，不重新加载模型或执行推理。

    ``predictions`` / ``targets`` 必须是 ``item -> [label_name]``，每个 item 保留独立
    视频边界；``labels`` 的顺序同时决定图例和颜色。滑窗与全序列 PredictionOutput
    使用同一契约，因此共享该呈现入口而不共享各自的推理实现。
    """

    predictions = dict(_value(output, "predictions", {}) or {})
    targets = dict(_value(output, "targets", {}) or {})
    labels = list(_value(output, "labels", []) or [])
    if not predictions:
        return []
    if not labels:
        raise ValueError("时序 timeline 缺少 labels")

    prediction_items = list(predictions)
    missing_targets = [name for name in prediction_items if name not in targets]
    if missing_targets:
        raise ValueError(f"时序 timeline 缺少真值 item: {missing_targets}")

    name_to_id = {str(label): index for index, label in enumerate(labels)}

    def encode(item: str, values) -> list[int]:
        encoded: list[int] = []
        for value in values:
            key = str(value)
            if key not in name_to_id:
                raise ValueError(f"时序 timeline item={item!r} 包含未知标签: {key!r}")
            encoded.append(name_to_id[key])
        return encoded

    pred_ids = [encode(name, predictions[name]) for name in prediction_items]
    truth_ids = [encode(name, targets[name]) for name in prediction_items]
    for name, prediction, truth in zip(prediction_items, pred_ids, truth_ids):
        if len(prediction) != len(truth):
            raise ValueError(
                f"时序 timeline item={name!r} 预测/真值长度不同: "
                f"{len(prediction)} != {len(truth)}"
            )

    metadata = dict(_value(output, "metadata", {}) or {})
    split = str(metadata.get("split") or "test")
    model_type = str(_value(output, "model_type", "temporal"))
    return render_segmentation(
        pred_ids,
        truth_ids,
        {index: str(label) for index, label in enumerate(labels)},
        prediction_items,
        title_prefix=f"{model_type} | {split}",
        out_dir=out_dir,
        base_name=f"segmentation-{split}",
        per_page=per_page,
    )


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
