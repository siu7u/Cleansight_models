"""FACT（frame–action 交叉注意力 + 段匹配损失）的单元测试。

覆盖四类不变量：

1. **接口契约**：`forward` 必须是 `[B,T,F] → [B,T,C]`；内部三路输出形状正确。
2. **时序分布归一化**：`token_time` 是沿时间轴的**对数**分布，`exp().sum(-1)` 必须为 1——
   它是匹配代价与段覆盖损失的基础，不归一化会让代价失去意义且难以察觉。
3. **段提取**：`extract_segments` 对全 idle / 单段 / 多段 / 首尾贴边的行为。
4. **边界与错误**：token 数少于真值段数时**不得崩溃**（匈牙利算法对非方阵须正常工作，
   只是部分段拿不到监督）；非法超参必须报错而不是静默接受。
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from framework.cleansight_eval.temporal.models import build_model
from framework.cleansight_eval.temporal.models.fact import FACTNet, extract_segments


def _model(**overrides):
    cfg = {"type": "fact", "input_dim": 12, "num_classes": 4, "d_model": 16, "nhead": 2,
           "num_blocks": 2, "num_tokens": 8, "frame_layers": 2, **overrides}
    return build_model(cfg)


def test_forward_and_internal_shapes():
    """`forward` 契约 + 三路内部输出形状。"""

    model = _model()
    batch, length = 2, 40
    x = torch.randn(batch, length, 12)
    assert model(x).shape == (batch, length, 4)

    frame_logits, token_logits, token_time = model._forward_all(x)
    assert frame_logits.shape == (batch, length, 4)
    assert token_logits.shape == (batch, 8, 5)      # C + 1（no-object）
    assert token_time.shape == (batch, 8, length)   # 每个 token 一个时间分布


def test_token_time_is_normalised_log_distribution():
    """token 的时间分布必须沿 T 归一化（否则匹配代价不可比）。"""

    model = _model()
    _, _, token_time = model._forward_all(torch.randn(1, 30, 12))
    totals = token_time.exp().sum(dim=-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)


def test_token_mixture_output_is_normalised():
    """token 混合输出必须是合法概率分布（Σ_c exp = 1）。

    这个不变量极易被漏掉：`token_time` 只沿 T 归一化、类别维截断 no-object 后也不再归一化，
    两处都要补 `log_softmax`。第一版实现漏了，输出概率和只有 0.107——数值上"能跑"、
    语义上完全是错的，因此必须由测试钉住。
    """

    model = _model(output_mode="token")
    out = model(torch.randn(2, 50, 12))
    assert out.shape == (2, 50, 4)
    totals = out.exp().sum(dim=-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-5)


def test_token_mixture_matches_shapes_of_frame_mode():
    """两种输出模式的接口必须一致（同一流水线要能互换）。"""

    x = torch.randn(1, 30, 12)
    assert _model(output_mode="frame")(x).shape == _model(output_mode="token")(x).shape


def test_extract_segments_handles_all_idle_and_boundaries():
    """段提取：全 idle 为空；贴着首尾的段必须被正确切出。"""

    assert extract_segments(torch.zeros(10, dtype=torch.long)) == []
    # 首帧就是动作、末帧也是动作
    labels = torch.tensor([2, 2, 0, 0, 3, 3, 3, 0, 1])
    assert extract_segments(labels) == [(0, 2, 2), (4, 7, 3), (8, 9, 1)]


def test_matching_loss_runs_when_sample_has_no_action_segment():
    """全 idle 样本必须走 no-object 分支，不得崩溃或产生 NaN。"""

    model = _model()
    x = torch.randn(1, 25, 12)
    y = torch.zeros(1, 25, dtype=torch.long)
    loss = model.compute_loss(x, y, nn.CrossEntropyLoss())
    assert torch.isfinite(loss)


def test_matching_loss_survives_fewer_tokens_than_segments():
    """token 数少于真值段数时不得崩溃。

    匈牙利算法对非方阵代价矩阵本就能工作（返回 min(R,K) 个匹配），此时部分段没有 token 监督——
    这是**精度损失**而非错误，但必须显式验证它不退化成异常。
    """

    model = _model(num_tokens=2)
    x = torch.randn(1, 60, 12)
    y = torch.zeros(1, 60, dtype=torch.long)
    # 造 5 个真值段，远多于 token 数 2
    for index, start in enumerate(range(0, 50, 10)):
        y[0, start:start + 6] = index % 3 + 1  # 类别须落在 1..num_classes-1
    loss = model.compute_loss(x, y, nn.CrossEntropyLoss())
    assert torch.isfinite(loss)


def test_compute_loss_backpropagates():
    """匹配损失必须真的参与反传（否则 token 分支等于没训）。"""

    model = _model()
    x = torch.randn(1, 40, 12)
    y = torch.zeros(1, 40, dtype=torch.long)
    y[0, 5:15] = 1
    y[0, 25:35] = 2
    loss = model.compute_loss(x, y, nn.CrossEntropyLoss())
    loss.backward()
    token_grad = model.tokens.grad
    assert token_grad is not None and float(token_grad.abs().sum()) > 0, "action token 未收到梯度"
    head_grad = model.token_class_head.weight.grad
    assert head_grad is not None and float(head_grad.abs().sum()) > 0


def test_invalid_hyperparameters_raise():
    """非法超参必须报错，不能静默接受。"""

    with pytest.raises(ValueError, match="整除"):
        FACTNet(input_dim=12, num_classes=4, d_model=15, nhead=4)
    with pytest.raises(ValueError, match="num_tokens"):
        FACTNet(input_dim=12, num_classes=4, num_tokens=0)
    with pytest.raises(ValueError, match="output_mode"):
        FACTNet(input_dim=12, num_classes=4, output_mode="bogus")
