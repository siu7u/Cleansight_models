"""时序模型的推理侧公共能力：加载 checkpoint、批量前向、取 penultimate 表示、S-NCM 解码。

**为什么这一层放在 framework**：`benchmark/` 与 `tools/` **不得直接调用模型库或重新加载
checkpoint**（见 ``tests/test_architecture_boundaries.py::test_active_non_framework_code_does_not_execute_models``）。
任何需要"读 checkpoint + 前向"的诊断工具或推理后处理都必须经由 framework，否则会把模型执行
能力扩散到非 framework 层。本模块把该能力集中一次，供工具与（未来的）推理链路共用。

**S-NCM（段级最近类均值重标注）** 是第一个消费者：它用模型 penultimate 表示算出的类均值，
对分类器已定好的**每个段**重新投票定标签。训练无关、零参数。机制与出处见
``docs/experiments/EXPERIMENT_REPORT_SEGMENT_NCM_20260927.md``。

术语与形状约定：

- ``[B,T,F]`` 输入 → ``[B,T,C]`` logits（框架统一约定）；
- penultimate 表示统一取成 ``[T,H]``（内部从 ``[B,H,T]`` 转置而来）。
"""

from __future__ import annotations

from collections import Counter

import numpy as np

__all__ = [
    "HIDDEN_HOOKS",
    "class_means",
    "forward_with_hidden",
    "load_temporal_model",
    "ncm_predict",
    "sncm_predict",
]

# 模型类型 → penultimate 表示的挂载路径。
# 每项为 ``(attr | None, index | None, leaf, is_batch_time_hidden)``：
#   ``target = getattr(model, attr)[index] if attr else model``，再取 ``getattr(target, leaf)``；
#   ``is_batch_time_hidden=True`` 表示 hook 收到的是 ``[B, T, H]``，否则 ``[B, H, T]``。
#
# **布局必须显式声明，不能按形状猜**：卷积头（Conv1d）收到 ``[B, H, T]``，
# 而线性头（Linear，如 GRU/Transformer）收到 ``[B, T, H]``——两者在同一份模型里也可能混用。
HIDDEN_HOOKS = {
    # 卷积分类头：forward_pre_hook 收到 [B, H, T]
    "mstcn": (None, None, "classifier", False),        # MSTCN.forward: self.classifier(z)
    "mstcn2": ("refines", -1, "out", False),           # Refinement.forward: self.out(z)
    # asformer 的 classifier 收到的是**已经转置过**的 [B, H, T]（见 _forward_stages）
    "asformer": (None, None, "classifier", False),
    # 双分支头：两个头共享同一份 z，挂其中任一个即可
    "actionness_tcn": ("heads", -1, "actionness", False),
    # 线性逐帧头（frame_head 是 nn.Linear，收到 [B, T, D]）
    "fact": (None, None, "frame_head", True),
    # 线性分类头：forward_pre_hook 收到 [B, T, H]
    "transformer": (None, None, "classifier", True),   # TransformerClassifier.forward: self.classifier(self.norm(z))
    "gru": (None, None, "head", True),                 # GRUClassifier.forward: self.head(...)
    # 框集合编码器 + MS-TCN++：penultimate 表示取**内部核心**的精化头，
    # 故 attr 用点路径（`core.refines`）下钻一层——编码器输出本身不是分类头输入。
    "boxset_mstcn2": ("core.refines", -1, "out", False),
}


def load_temporal_model(cfg: dict, ckpt: str, device):
    """按配置与 checkpoint 重建全序列时序模型（复用流水线的加载口径）。

    返回 ``(model, meta)``；模型处于 ``eval()``。``cfg`` 为 resolved 配置（含 ``model`` /
    ``evaluation`` 段），``ckpt`` 为 ``best.pt`` 之类路径。
    """

    import torch

    from .full_sequence_pipeline import _load_eval_model

    return _load_eval_model(cfg, ckpt, torch.device(device) if isinstance(device, str) else device)


def _hidden_module(model, model_type: str):
    """按 ``HIDDEN_HOOKS`` 解析出 penultimate 表示的挂载模块。"""

    if model_type not in HIDDEN_HOOKS:
        raise ValueError(
            f"未登记 penultimate 表示的挂载点: model_type={model_type!r}；"
            f"已登记: {sorted(HIDDEN_HOOKS)}（新增模型请在 HIDDEN_HOOKS 补一行）"
        )
    attr, index, leaf, _is_btH = HIDDEN_HOOKS[model_type]
    target = model
    if attr is not None:
        # attr 支持点路径（如 "core.refines"）：包装型的模型需要下钻到内部核心，
        # 才能挂到真正的分类头前一层。
        for part in attr.split("."):
            target = getattr(target, part)
        if index is not None:
            target = target[index]
    return getattr(target, leaf)


def forward_with_hidden(model, features: list, model_type: str):
    """逐序列整段前向，返回 ``(logits 列表 [T,C], hidden 列表 [T,H])``。

    用 ``forward_pre_hook`` 在 penultimate 位置截取表示；hook 在返回前一定会被摘除。
    """

    import torch

    is_batch_time_hidden = HIDDEN_HOOKS[model_type][3]
    buffer: dict = {}
    module = _hidden_module(model, model_type)
    handle = module.register_forward_pre_hook(lambda m, a: buffer.__setitem__("z", a[0].detach()))
    logits, hiddens = [], []
    try:
        with torch.no_grad():
            for feats in features:
                out = model(torch.from_numpy(feats).float().unsqueeze(0))
                z = buffer["z"][0]
                # 统一成 [T, H]：卷积头给 [H,T]（转置），线性头给 [T,H]（原样）。
                hiddens.append((z if is_batch_time_hidden else z.transpose(0, 1)).cpu().numpy())
                logits.append(out[0].cpu().numpy())
    finally:
        handle.remove()
    return logits, hiddens


def class_means(hiddens: list, labels: list, num_classes: int) -> np.ndarray:
    """按类求表示空间均值，返回 ``[C, H]``；某类无样本时报错（避免静默产生 NaN 中心）。"""

    centers = []
    for cls in range(num_classes):
        parts = [h[labels[i] == cls] for i, h in enumerate(hiddens) if (labels[i] == cls).any()]
        if not parts:
            raise ValueError(f"类 {cls} 在给定的标签里没有样本，无法估计类均值")
        centers.append(np.concatenate(parts, axis=0).mean(axis=0))
    return np.stack(centers)


def ncm_predict(hiddens: list, centers: np.ndarray) -> list:
    """逐帧最近类均值（欧氏距离），返回每个序列的 ``[T]`` 标签。"""

    out = []
    for z in hiddens:
        dist = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        out.append(dist.argmin(axis=1).astype(np.int64))
    return out


def sncm_predict(boundaries: list, votes: list) -> list:
    """S-NCM：沿用 ``boundaries`` 的段边界，每段取 ``votes`` 的众数作为该段标签。

    注意：若 ``boundaries`` 与 ``votes`` 来自同一套逐帧预测，段内 vote 必然恒定，
    该调用会**逐帧恒等**（决策上是无操作）——自检用例即用这一点。
    """

    out = []
    for base, vote in zip(boundaries, votes):
        if len(base) != len(vote):
            raise ValueError(f"边界序列与投票序列长度不一致: {len(base)} vs {len(vote)}")
        seg = base.copy()
        starts = np.concatenate([[0], np.where(np.diff(base) != 0)[0] + 1])
        ends = np.concatenate([starts[1:], [len(base)]])
        for s, e in zip(starts, ends):
            seg[s:e] = Counter(vote[s:e].tolist()).most_common(1)[0][0]
        out.append(seg)
    return out
