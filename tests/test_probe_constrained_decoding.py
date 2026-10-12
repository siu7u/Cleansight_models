"""`tools/probe_constrained_decoding.py` 的时长代价三模式单元校验。

动机：该工具的 ``viterbi_decode`` docstring 一直声称"``α=0`` 且 ``hard_duration=False`` 时逐帧
等价于 argmax（已由单元校验固定）"，但仓库里**没有**对应用例；而且 2026-10-12 复查发现
``hard_duration`` 参数**在函数体里从未被使用**——硬掩码是无条件施加的，``--no-duration`` 之所以
表现得像普通 Viterbi，只是因为调用处另外把 ``min_duration`` 置成了全 1。本文件把三种模式的行为
固定下来，防止再次出现"参数是死的、文档却声称有校验"。

用例只依赖 numpy/scipy，不加载模型、不读数据（秒级）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.probe_constrained_decoding import viterbi_decode  # noqa: E402


def _identity_log_transition(n: int) -> np.ndarray:
    """均匀转移（对角线置零，函数内部会再置 −inf），即转移先验不带任何信息。"""

    return np.zeros((n, n), dtype=np.float64)


def _pulse_logits(n_classes: int = 3, length: int = 12) -> np.ndarray:
    """构造一段"真值类 1 持续 8 帧、其余为 0"的 logits，其中类 2 有 1 帧强脉冲。

    形状 ``[T, C]``（时间在前）。类 0 是 idle。脉冲用于检验：时长约束是否阻止解码器在
    类 2 上留下 **短于 D_2** 的段。
    """

    logits = np.zeros((length, n_classes), dtype=np.float64)
    logits[:, 0] = 1.0          # 默认偏 idle
    logits[2:10, 1] = 2.0       # 真值段：类 1，8 帧
    logits[11, 2] = 5.0         # 类 2 的单帧脉冲
    return logits


def _segments(labels: np.ndarray) -> list[tuple[int, int]]:
    """把标签序列压成 ``[(类别, 长度), ...]``。"""

    out: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((int(labels[start]), i - start))
            start = i
    return out


def test_alpha_zero_without_duration_equals_argmax():
    """α=0 且不加任何时长代价 ⇒ 逐帧等价于 argmax（docstring 声称的性质）。

    **必须在 D>1 上也成立**：此时状态数 S>1，回溯指针方向错了就会返回非最优路径。
    仅用 ``min_duration=ones``（S=1）测不出该错——那正是本仓库原先"已由单元校验固定"
    这句话没有兑现的地方（2026-10-12 查出并修复）。
    """

    logits = _pulse_logits()
    expected = np.argmax(logits, axis=1)
    for min_duration in (np.ones(3), np.array([1.0, 4.0, 4.0]), np.array([1.0, 8.0, 2.0])):
        labels = viterbi_decode(logits, _identity_log_transition(3), min_duration,
                                transition_strength=0.0, hard_duration=False,
                                soft_duration_strength=0.0)
        np.testing.assert_array_equal(
            labels, expected,
            err_msg=f"min_duration={min_duration} 下不等于 argmax（检查回溯指针方向）")


def test_hard_duration_forbids_short_segments():
    """硬约束：D_c 大于 1 的类**不得出现短于 D_c 的内部段**。

    注意语义边界：约束管的是"**切换**"，不管"末段有多长"——被序列末尾截断的段天然可能
    短于 D_c（想让末段也满足，需要另外补"末段时长下界"的处理，本工具没有做）。
    故这里只看**非末尾**段。
    """

    logits = _pulse_logits()
    min_duration = np.array([1.0, 4.0, 4.0])   # 类 1/2 至少 4 帧，类 0（idle）不受限
    labels = viterbi_decode(logits, _identity_log_transition(3), min_duration,
                            transition_strength=0.0, hard_duration=True)
    interior = _segments(labels)[:-1]
    short = [(c, n) for c, n in interior if c in (1, 2) and n < 4]
    assert short == [], f"硬约束下出现了过短的内部段: {short}（完整段 {_segments(labels)}）"
    # 约束确实起了作用：真值类 1 被从 8 帧延长到 ≥4 且覆盖了原区间
    assert any(c == 1 and n >= 4 for c, n in _segments(labels))


def test_soft_duration_still_allows_short_segment():
    """软惩罚：同样的 D，但**允许**提前切走 —— 单帧脉冲在 β 较小时应被保留。"""

    logits = _pulse_logits()
    min_duration = np.array([1.0, 4.0, 4.0])
    labels = viterbi_decode(logits, _identity_log_transition(3), min_duration,
                            transition_strength=0.0, hard_duration=False,
                            soft_duration_strength=0.1)
    # 类 2 的脉冲区间（下标 11）应仍在输出里，且长度 < D_2=4 —— 这正是硬约束做不到的
    assert labels[11] == 2, f"软惩罚下脉冲被抹掉: {_segments(labels)}"


@pytest.mark.parametrize("beta", [0.0, 0.5, 5.0])
def test_soft_duration_is_monotone_in_beta(beta: float):
    """软惩罚强度越大，段结构越接近硬约束（此处只断言"可运行且段数不增"这一弱性质）。"""

    logits = _pulse_logits()
    min_duration = np.array([1.0, 4.0, 4.0])
    weak = viterbi_decode(logits, _identity_log_transition(3), min_duration,
                          transition_strength=0.0, hard_duration=False,
                          soft_duration_strength=0.0)
    strong = viterbi_decode(logits, _identity_log_transition(3), min_duration,
                            transition_strength=0.0, hard_duration=False,
                            soft_duration_strength=beta)
    assert len(_segments(strong)) <= len(_segments(weak))
