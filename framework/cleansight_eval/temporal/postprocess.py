"""v1.2 批式后处理：概率平滑 + 逐类时长先验（min 合并 / max 拆分）。

定版依据 FEATURE_LAB §22.4（2026-10-17，LOVO 35 折）：
  edit 中位 71.43 / F1@0.25 中位 55.88（较 v1.1 +2.2 edit / +3.1 F1）
test split 泛化验证：2 改善 / 6 持平 / 0 恶化（均值 +3.05）。

三参数的消融依据（§22.5 + 后处理文档）：
- 逐类 min_len 是类时长先验（各类 GT 段长中位差 5~10 倍），统一值 -6.7 edit；
- idle 优先合并消除动作段互相吞噬（fold13 级联案例）；
- max_len 用 P95 已确认最优（扫描 0.5x~10x 无改进）；
- 概率平滑 k=5 为均值滑动窗（中心窗，用未来 2 帧）——流式部署需因果版（待办）。

配置入口（evaluation 段）：
  postprocess: "v12"            # 启用；缺省 "none" 保持 causal_decision 行为
  postprocess_params:           # 可选覆盖（下面的值即默认）
    min_len: [11, 1, 3, 28, 10, 6]
    max_len: [288, 3, 89, 232, 92, 131]
    smooth_k: 5
    merge_mode: idle_preferred  # 或 longer_neighbor（v1.1 行为）
"""

from __future__ import annotations

import numpy as np

# v4.1 / 15fps 定版参数（类序：idle/water/flush/lbi/lbw/sbc）
DEFAULT_MIN_LEN = (11, 1, 3, 28, 10, 6)
DEFAULT_MAX_LEN = (288, 3, 89, 232, 92, 131)
DEFAULT_SMOOTH_K = 5
DEFAULT_MERGE_MODE = "idle_preferred"

_VALID_MODES = ("longer_neighbor", "idle_preferred")


def smooth_probs(probs, k=DEFAULT_SMOOTH_K):
    """逐类通道均值滑动窗（边缘复制填充）。k<=1 原样返回。"""
    if k <= 1:
        return probs
    half = k // 2
    pad = np.pad(probs, ((half, half), (0, 0)), mode="edge")
    kernel = np.ones(k) / k
    out = np.empty_like(probs)
    for c in range(probs.shape[1]):
        out[:, c] = np.convolve(pad[:, c], kernel, mode="valid")
    return out


def _to_segments(labels):
    segs, start = [], 0
    values = labels.tolist()
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            segs.append([start, i, values[start]])
            start = i
    return segs


def _segments_to_labels(segs, total):
    out = np.empty(total, dtype=np.int64)
    for start, end, cls in segs:
        out[start:end] = cls
    return out


def merge_short(labels, min_len=DEFAULT_MIN_LEN, mode=DEFAULT_MERGE_MODE):
    """短于 min_len[c] 的段并入邻居。

    mode:
      - longer_neighbor（v1.1）：并入更长的邻居——对 idle 有系统性偏向，
        极端情况下动作段被级联吞掉（fold13 案例）；
      - idle_preferred（v1.2 定版）：短段优先并入 idle 邻居，
        消除动作段互相吞噬；无 idle 邻居时回退 longer_neighbor 规则。
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"merge_mode 必须是 {_VALID_MODES}，实际 {mode!r}")
    ml = list(min_len)
    segs = _to_segments(labels)
    changed = True
    while changed:
        changed = False
        j = 0
        while j < len(segs) - 1:
            if segs[j][2] == segs[j + 1][2]:
                segs[j][1] = segs[j + 1][1]
                del segs[j + 1]
                changed = True
            else:
                j += 1
    guard, changed = 0, True
    while changed and guard < 300:
        changed = False
        guard += 1
        for j, (start, end, cls) in enumerate(segs):
            if end - start >= ml[cls]:
                continue
            left = segs[j - 1] if j > 0 else None
            right = segs[j + 1] if j < len(segs) - 1 else None
            if mode == "idle_preferred":
                if left is not None and left[2] == 0:
                    segs[j - 1] = [left[0], end, 0]
                elif right is not None and right[2] == 0:
                    segs[j + 1] = [start, right[1], 0]
                elif left is not None and (
                    right is None or (left[1] - left[0]) >= (right[1] - right[0])
                ):
                    segs[j - 1] = [left[0], end, left[2]]
                elif right is not None:
                    segs[j + 1] = [start, right[1], right[2]]
                else:
                    continue
            else:
                if left is not None and (
                    right is None or (left[1] - left[0]) >= (right[1] - right[0])
                ):
                    segs[j - 1] = [left[0], end, left[2]]
                elif right is not None:
                    segs[j + 1] = [start, right[1], right[2]]
                else:
                    continue
            del segs[j]
            changed = True
            break
    return _segments_to_labels(segs, len(labels))


def split_long(labels, max_len=DEFAULT_MAX_LEN):
    """超过 max_len[c] 的段在阈值处截断，余下并入邻居类。"""
    mx = list(max_len)
    segs = _to_segments(labels)
    for j in range(len(segs)):
        start, end, cls = segs[j]
        if end - start <= mx[cls]:
            continue
        neighbor_cls = segs[j + 1][2] if j < len(segs) - 1 else segs[j - 1][2]
        segs[j] = [start, start + mx[cls], cls]
        segs.insert(j + 1, [start + mx[cls], end, neighbor_cls])
    return _segments_to_labels(segs, len(labels))


def apply_v12(probs, min_len=DEFAULT_MIN_LEN, max_len=DEFAULT_MAX_LEN,
              smooth_k=DEFAULT_SMOOTH_K, merge_mode=DEFAULT_MERGE_MODE):
    """整条 v1.2 流水线：概率 → 平滑 → argmax → min 合并 → max 拆分。

    probs: [T, C] 逐帧类别概率（冷启动帧请预先填充 idle one-hot）。
    返回 [T] 类 id 序列。
    """
    if probs.ndim != 2:
        raise ValueError(f"probs 必须是 [T, C]，实际 {probs.shape}")
    if len(min_len) != probs.shape[1] or len(max_len) != probs.shape[1]:
        raise ValueError(
            f"min_len/max_len 长度 ({len(min_len)}/{len(max_len)}) 必须等于类别数 {probs.shape[1]}"
        )
    smoothed = smooth_probs(probs, smooth_k)
    labels = smoothed.argmax(axis=1)
    labels = merge_short(labels, min_len, merge_mode)
    return split_long(labels, max_len)


def resolve_params(evaluation_cfg):
    """从 evaluation 配置解析 v1.2 参数（含合法性检查）。"""
    params = dict(evaluation_cfg.get("postprocess_params") or {})
    unknown = set(params) - {"min_len", "max_len", "smooth_k", "merge_mode"}
    if unknown:
        raise ValueError(f"postprocess_params 含未知字段: {sorted(unknown)}")
    out = {
        "min_len": tuple(params.get("min_len", DEFAULT_MIN_LEN)),
        "max_len": tuple(params.get("max_len", DEFAULT_MAX_LEN)),
        "smooth_k": int(params.get("smooth_k", DEFAULT_SMOOTH_K)),
        "merge_mode": str(params.get("merge_mode", DEFAULT_MERGE_MODE)),
    }
    if out["merge_mode"] not in _VALID_MODES:
        raise ValueError(f"merge_mode 必须是 {_VALID_MODES}，实际 {out['merge_mode']!r}")
    return out
