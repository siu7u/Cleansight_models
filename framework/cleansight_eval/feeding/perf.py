"""性能测量（需求 §8.4）。

是否测量由喂入模式决定：有界因果窗测单 tick 延迟，全序列标记为 N/A 而非造假。
性能只作为事实输出，不与业务门槛比较。迁移自 ``tools/measure_temporal_latency.py``。
"""

from __future__ import annotations

import time

import torch

from ..core.envelope import MetricValue

SPEC_LATENCY = "latency/single_tick_ms/v1"


def measure_single_tick(family, model, window: int, input_dim: int, device, warmup: int = 20, runs: int = 200) -> dict[str, MetricValue]:
    """测量单窗口 ``[1, window, input_dim]`` 前向延迟。"""

    model.eval()
    x = torch.randn(1, window, input_dim, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            family.predict_frame_logits(model, x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        samples = []
        for _ in range(runs):
            t0 = time.perf_counter()
            family.predict_frame_logits(model, x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()
    mean_ms = sum(samples) / len(samples)
    median_ms = samples[len(samples) // 2]
    p95_ms = samples[min(len(samples) - 1, int(0.95 * len(samples)))]
    spec = f"{SPEC_LATENCY}; device={device}; window={window}; warmup={warmup}; runs={runs}"
    return {
        "latency_mean_ms": MetricValue.computed(round(mean_ms, 4), spec=spec),
        "latency_median_ms": MetricValue.computed(round(median_ms, 4), spec=spec),
        "latency_p95_ms": MetricValue.computed(round(p95_ms, 4), spec=spec),
    }


def not_applicable_perf(reason: str = "该喂入模式不测量实时延迟") -> dict[str, MetricValue]:
    return {
        "latency_mean_ms": MetricValue.not_applicable(reason, spec=SPEC_LATENCY),
        "latency_median_ms": MetricValue.not_applicable(reason, spec=SPEC_LATENCY),
        "latency_p95_ms": MetricValue.not_applicable(reason, spec=SPEC_LATENCY),
    }
