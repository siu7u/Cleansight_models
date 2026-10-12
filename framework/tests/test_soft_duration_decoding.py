"""离线软时长惩罚解码的框架侧用例（`framework/cleansight_eval/temporal/util.py`）。

背景：`tools/probe_constrained_decoding.py` 已验证"软时长惩罚 Viterbi"在三个契约上能显著改善段级
F1（见 `docs/weeks/2026-10-10_2026-10-16/TEMPORAL_SOFTDUR_20261012.md`），但实现原先只在探针里。
2026-10-12 把实现迁入 framework（归属推理链路的层），探针改为**再导出**以避免两份实现漂移。

本文件的**第一条用例是判别性的**：`β=0` 且不加任何时长代价时必须逐帧等于 argmax。
若这条不过，说明解码器在未被察觉的情况下改了**全部历史读数**——迁移后必须复跑它。
"""

from __future__ import annotations

import numpy as np
import pytest

from cleansight_eval.temporal.util import estimate_min_duration, viterbi_decode


def _logits(n_classes: int = 3, length: int = 16) -> np.ndarray:
    """构造 ``[T, C]`` 的 logits：类 1 一段、类 2 一个单帧脉冲，其余偏 idle（类 0）。"""

    out = np.zeros((length, n_classes), dtype=np.float64)
    out[:, 0] = 1.0
    out[3:11, 1] = 2.0
    out[12, 2] = 5.0
    return out


def _segments(labels: np.ndarray) -> list[tuple[int, int]]:
    """把标签序列压成 ``[(类别, 长度), ...]``。"""

    out: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((int(labels[start]), i - start))
            start = i
    return out


@pytest.mark.parametrize("min_duration", [np.ones(3), np.array([1.0, 4.0, 4.0]), np.array([1.0, 8.0, 2.0])])
def test_zero_penalty_equals_argmax(min_duration):
    """**判别性用例**：不施加任何时长代价时，解码必须逐帧等于 argmax（D>1 也要成立）。

    只在 ``min_duration=ones``（状态数 S=1）上验是不够的——历史上正是这个盲区让一处回溯指针
    方向错误潜伏了下来（详见 TEMPORAL_DECODE_FIX_20261012.md）。
    """

    logits = _logits()
    got = viterbi_decode(logits, np.zeros((3, 3)), min_duration, 0.0,
                         hard_duration=False, soft_duration_strength=0.0)
    np.testing.assert_array_equal(got, np.argmax(logits, axis=1))


def test_hard_duration_forbids_short_interior_segment():
    """硬约束：``D_c>1`` 的类不得出现短于 ``D_c`` 的**内部**段（末段可被序列长度截断）。"""

    labels = viterbi_decode(_logits(), np.zeros((3, 3)), np.array([1.0, 4.0, 4.0]), 0.0,
                            hard_duration=True)
    short = [(c, n) for c, n in _segments(labels)[:-1] if c in (1, 2) and n < 4]
    assert short == [], f"硬约束下出现过短内部段: {short}（全部段 {_segments(labels)}）"


def test_soft_penalty_keeps_short_segment():
    """软惩罚：同样的 ``D``，但**允许**提前切走——单帧脉冲在较小 β 下应被保留。"""

    labels = viterbi_decode(_logits(), np.zeros((3, 3)), np.array([1.0, 4.0, 4.0]), 0.0,
                            hard_duration=False, soft_duration_strength=0.1)
    assert labels[12] == 2, f"软惩罚下脉冲被抹掉: {_segments(labels)}"


def test_estimate_min_duration_semantics():
    """``estimate_min_duration``：逐类取段长分位、下界为 1、分位越小结果不增。"""

    truths = [np.array([0, 0, 1, 1, 1, 2]), np.array([0, 1, 1, 1, 1, 1])]
    d50 = estimate_min_duration(truths, 3, quantile=0.50)
    d10 = estimate_min_duration(truths, 3, quantile=0.10)
    assert d50.shape == (3,) and (d50 >= 1).all()
    assert (d10 <= d50 + 1e-9).all(), f"分位越小 D 不应更大: q0.10={d10} q0.50={d50}"
    # 类 1 的段长是 [3, 5] ⇒ 中位 4
    assert d50[1] == pytest.approx(4.0)
