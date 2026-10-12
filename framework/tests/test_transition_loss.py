"""转移级代价敏感（`TransitionWeightedCE` / `estimate_transition_rarity`）的单元测试。

覆盖四类不变量：

1. **稀有度表语义**：最高频转移（通常是段内 `a==a`）→ 0；未出现转移 → 1。
2. **零强度等价性**：`strength=0` 必须与普通类别加权 ``CrossEntropyLoss`` **数值完全相等**
   ——这是"该旋钮关闭时不改变任何既有结论"的前提。
3. **权重归一化**：逐帧权重按均值归一，保证损失量级与基线可比；当所有帧转移相同时，
   加权结果必须与不加权**完全相同**（因为归一化后每帧权重都是 1）。
4. **分批保护**：`single_sequence=False` 时必须退化为普通加权 CE，不得在序列边界处
   引入伪转移。
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from framework.cleansight_eval.temporal.util import (
    TransitionWeightedCE,
    estimate_transition_rarity,
)

CLASS_WEIGHTS = torch.tensor([0.1, 1.0, 0.5], dtype=torch.float32)


def _rarity():
    # 0→0 出现 2 次（最高频）；0→1、1→0、0→2、2→0 各 1 次；1→2、2→1、1→1、2→2 未出现
    return estimate_transition_rarity([np.array([0, 0, 0, 1, 1, 0, 2, 2, 0, 1])], num_classes=3)


def test_rarity_is_zero_for_most_frequent_and_one_for_absent():
    rarity = _rarity()
    assert rarity.shape == (3, 3)
    assert rarity[0, 0] == pytest.approx(0.0)      # 最高频转移
    assert rarity[1, 2] == pytest.approx(1.0)      # 从未出现
    assert 0.0 < rarity[0, 2] < 1.0


def test_rarity_of_all_idle_sequence_marks_absent_transitions():
    """全 idle 序列：唯一的真实转移 0→0 稀有度为 0；未出现的转移为 1（不得除零）。"""

    rarity = estimate_transition_rarity([np.zeros(20, dtype=np.int64)], num_classes=3)
    assert rarity[0, 0] == pytest.approx(0.0)   # 唯一的高频转移
    assert rarity[0, 1] == pytest.approx(1.0)   # 从未出现
    assert np.isfinite(rarity).all()


def test_zero_strength_matches_plain_weighted_cross_entropy():
    """`strength=0` 必须与普通加权 CE 数值完全相等——否则会污染所有既有对照。"""

    logits = torch.randn(40, 3)
    target = torch.randint(0, 3, (40,))
    plain = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)(logits, target)
    ours = TransitionWeightedCE(CLASS_WEIGHTS, _rarity(), strength=0.0)(logits, target)
    assert torch.allclose(plain, ours, atol=1e-7)


def test_uniform_transitions_give_identical_loss_after_normalisation():
    """所有帧转移相同时，归一化使每帧权重恒为 1 → 加权结果与不加权完全相同。"""

    logits = torch.randn(30, 3)
    target = torch.zeros(30, dtype=torch.long)          # 全部 idle：转移只有 0→0
    plain = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)(logits, target)
    ours = TransitionWeightedCE(CLASS_WEIGHTS, _rarity(), strength=3.0)(logits, target)
    assert torch.allclose(plain, ours, atol=1e-6)


def test_positive_strength_changes_loss_when_transitions_vary():
    """有稀有转移时必须真的改变损失（否则这个旋钮等于没生效）。"""

    logits = torch.randn(30, 3)
    target = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)   # 含 0→1、1→2 两个转移
    plain = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)(logits, target)
    ours = TransitionWeightedCE(CLASS_WEIGHTS, _rarity(), strength=3.0)(logits, target)
    assert not torch.allclose(plain, ours)


def test_batched_sequences_disable_the_reweighting():
    """`single_sequence=False` 必须退化为普通加权 CE（分批时没有序列边界信息）。"""

    logits = torch.randn(60, 3)
    target = torch.randint(0, 3, (60,))
    plain = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)(logits, target)
    ours = TransitionWeightedCE(
        CLASS_WEIGHTS, _rarity(), strength=3.0, single_sequence=False
    )(logits, target)
    assert torch.allclose(plain, ours, atol=1e-7)


def test_negative_strength_raises():
    with pytest.raises(ValueError, match="≥0"):
        TransitionWeightedCE(CLASS_WEIGHTS, _rarity(), strength=-1.0)


def test_gradients_flow():
    """加权损失必须可反传（不能因索引操作断开梯度）。"""

    logits = torch.randn(30, 3, requires_grad=True)
    target = torch.tensor([0] * 15 + [2] * 15)
    loss = TransitionWeightedCE(CLASS_WEIGHTS, _rarity(), strength=2.0)(logits, target)
    loss.backward()
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0
