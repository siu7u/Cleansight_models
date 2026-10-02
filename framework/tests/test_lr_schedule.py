"""学习率调度（`train.lr_schedule` / `warmup_epochs` / `min_lr_ratio`）的单元测试。

**为什么需要**：`constant` 必须与历史行为**逐位一致**（默认口径不能被这次改动悄悄改掉），
`cosine` 的 warmup 峰值与衰减下界要可验证，非法配置要立刻报错而不是静默退回默认。
"""

import math

import pytest

from cleansight_eval.temporal.full_sequence_pipeline import (
    VALID_LR_SCHEDULES,
    lr_factor,
    resolve_lr_schedule,
)


def test_default_is_constant_and_returns_unit_factor():
    """缺省即 `constant`，且倍率恒为 1.0（= 历史行为不变）。"""

    assert resolve_lr_schedule({}) == ("constant", 0, 0.0)
    assert resolve_lr_schedule(None) == ("constant", 0, 0.0)
    assert {lr_factor("constant", e, 60, 0, 0.0) for e in range(1, 61)} == {1.0}


def test_cosine_warmup_peak_and_floor():
    """warmup 期内线性升到 1.0；之后单调不增，并在末轮落到 `min_lr_ratio`。"""

    schedule, warmup, floor = resolve_lr_schedule(
        {"lr_schedule": "cosine", "warmup_epochs": 3, "min_lr_ratio": 0.05})
    assert (schedule, warmup, floor) == ("cosine", 3, 0.05)

    factors = [lr_factor(schedule, e, 60, warmup, floor) for e in range(1, 61)]
    assert factors[0] == pytest.approx(1 / 3)      # 线性 warmup 第 1 轮
    assert factors[2] == pytest.approx(1.0)        # 第 3 轮到峰值
    tail = factors[2:]
    assert all(a >= b - 1e-12 for a, b in zip(tail, tail[1:]))   # 单调不增
    assert factors[-1] == pytest.approx(0.05)      # 末轮落到下界
    assert min(factors) >= 0.05 - 1e-12


def test_cosine_without_warmup_decays_over_the_whole_run():
    """无 warmup 时余弦跨度为 (0, epochs]：首轮为 cos(π/epochs)，末轮恰为下界。"""

    factors = [lr_factor("cosine", e, 10, 0, 0.0) for e in range(1, 11)]
    assert factors[0] == pytest.approx(0.5 * (1 + math.cos(math.pi / 10)))   # ≈0.976
    assert factors[-1] == pytest.approx(0.0)
    assert all(a >= b - 1e-12 for a, b in zip(factors, factors[1:]))


def test_invalid_values_raise():
    """非法值立即报错，不静默退回默认。"""

    with pytest.raises(ValueError, match="lr_schedule"):
        resolve_lr_schedule({"lr_schedule": "step"})
    with pytest.raises(ValueError, match="warmup_epochs"):
        resolve_lr_schedule({"lr_schedule": "cosine", "warmup_epochs": -1})
    with pytest.raises(ValueError, match="min_lr_ratio"):
        resolve_lr_schedule({"lr_schedule": "cosine", "min_lr_ratio": 1.5})
    assert set(VALID_LR_SCHEDULES) == {"constant", "cosine"}
