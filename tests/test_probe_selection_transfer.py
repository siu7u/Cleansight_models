"""tools/probe_selection_transfer.py：选点口径 val→test 迁移体检的单元测试。

验收标准（verification-first）：
- `best_of` 对 val_loss 取**最小**、对其它指标取**最大**，并返回 1-based 轮次；
- `collect` 能把 history.csv 的 val 峰值与该 run 的 evaluation.json test 指标配对，
  缺 test 指标（not_applicable/missing）时跳过而不是记 0；
- 相关系数方向可判读：合成"val 与 test 强正相关"与"完全无关"两组数据，ρ 必须一个高一个接近 0；
- 没有 history+evaluation 的 run 不进入统计。

不依赖真实数据集与 checkpoint（用 tmp_path 造 run 目录）。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.probe_selection_transfer import best_of, collect  # noqa: E402


def write_run(root: Path, name: str, val_edit: list[float], test_edit: float,
              *, include_test: bool = True) -> Path:
    run = root / name
    (run / "evals").mkdir(parents=True)
    with (run / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "val_loss", "val_acc", "val_edit", "val_f1_0.1", "val_f1_0.25", "val_f1_0.5"])
        for index, value in enumerate(val_edit):
            writer.writerow([index + 1, 9.0 - index * 0.1, 68.0, value, value / 2, value / 3, value / 5])
    summary = {"edit": {"state": "computed", "value": test_edit}} if include_test else \
        {"edit": {"state": "not_applicable", "value": None}}
    (run / "evals" / "full_sequence_temporal-x.evaluation.json").write_text(
        json.dumps({"metrics": {"summary": summary}}))
    return run


def test_best_of_picks_max_for_scores_and_min_for_loss():
    values = [10.0, 30.0, 20.0]
    assert best_of(values, "val_edit") == (2, 30.0)          # 最大 → 第 2 轮
    assert best_of([9.0, 3.0, 5.0], "val_loss") == (2, 3.0)  # 最小 → 第 2 轮
    assert best_of([1.0], "val_f1_0.5") == (1, 1.0)


def test_collect_pairs_val_peak_with_test_and_skips_missing(tmp_path):
    write_run(tmp_path, "run-a", [10.0, 40.0, 20.0], test_edit=30.0)
    write_run(tmp_path, "run-b", [5.0, 15.0, 25.0], test_edit=99.0, include_test=False)

    buckets, details = collect(tmp_path)

    assert len(details) == 1                       # run-b 的 test 不可评估 → 不进明细
    assert details[0]["edit"] == {"epoch": 2, "val": 40.0, "test": 30.0}
    assert buckets["edit"] == [(40.0, 30.0)]


def test_collect_ignores_history_without_evaluation(tmp_path):
    run = tmp_path / "run-c"
    run.mkdir()
    (run / "history.csv").write_text("epoch,val_edit\n1,5.0\n", encoding="utf-8")

    buckets, details = collect(tmp_path)

    assert details == [] and buckets["edit"] == []


def test_transfer_correlation_separates_predictive_from_useless_metric(tmp_path):
    """合成两组：val 与 test 强相关（应高 ρ）vs 完全独立（应接近 0）。"""
    rng = np.random.default_rng(7)
    predictive = rng.normal(size=12) * 5 + 30
    useless = rng.normal(size=12) * 5 + 30
    for index in range(12):
        write_run(tmp_path, f"pred-{index}", [predictive[index], predictive[index]], test_edit=float(predictive[index]))
        write_run(tmp_path, f"rand-{index}", [useless[index], useless[index]], test_edit=float(rng.normal() * 5 + 30))

    buckets, _details = collect(tmp_path)

    from scipy.stats import spearmanr
    # 只看 pred-* 的配对（collect 会把两组混在一起，这里手工分离）
    pred = [(v, t) for v, t in buckets["edit"] if any(abs(v - p) < 1e-9 for p in predictive)]
    assert len(pred) >= 10
    rho = spearmanr([p[0] for p in pred], [p[1] for p in pred]).statistic
    assert rho > 0.9   # val==test 的构造 → 秩完全一致
