"""tools/probe_seed_ensemble.py：多种子预测集成的单元测试。

验收标准（verification-first）：
- `majority_vote` 的语义：逐帧取多数；**平局取标签 id 较小者**（与文档声明一致，可复现）；
- 单成员时投票恒等于该成员本身（集成退化为原预测）；
- 成员意见一致时投票等于该意见；
- `load_members` 对同一 seed 的重复 run 去重（否则同一模型会被计两次票）；
- 配对统计方向正确：集成的逐视频指标优于成员均值时，差值应显著为正。

不加载真实数据集：用 tmp_path 造最小 run 产物。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.probe_seed_ensemble import load_members, majority_vote, paired  # noqa: E402

LABELS = ["idle", "flush", "insert"]
N = len(LABELS)


def test_majority_vote_picks_per_frame_majority():
    stack = np.array([[0, 1, 2],
                      [0, 1, 0],
                      [0, 2, 0]])
    # 逐帧票数：frame0 → idle 3 票；frame1 → flush 2 票；frame2 → idle 2 票
    assert list(majority_vote(stack, N)) == [0, 1, 0]


def test_single_member_vote_is_identity():
    sequence = np.array([1, 1, 2, 2, 0])
    assert list(majority_vote(np.stack([sequence]), N)) == list(sequence)


def test_unanimous_vote_equals_that_label():
    stack = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
    assert list(majority_vote(stack, N)) == [2, 2, 2]


def test_tie_rule_is_smallest_label_id():
    """2 vs 2 平局：取 id 较小者（可复现，不依赖顺序）。"""
    stack = np.array([[2, 0], [2, 0], [0, 2], [0, 2]])
    assert list(majority_vote(stack, N)) == [0, 0]


def write_run(root: Path, name: str, seed: int, predictions: dict[str, list[str]]) -> Path:
    run = root / name
    (run / "artifacts").mkdir(parents=True)
    (run / "env.json").write_text(json.dumps({"seed": seed}))
    (run / "artifacts" / "full_sequence_temporal-x.predictions.json").write_text(json.dumps({
        "labels": [{"name": name} for name in LABELS],
        "items": {video: {"predicted_labels": labels, "truth_labels": labels}
                  for video, labels in predictions.items()},
    }))
    return run


def test_load_members_deduplicates_same_seed(tmp_path, monkeypatch, capsys):
    write_run(tmp_path, "run-a", 7, {"v0": ["idle", "idle"]})
    write_run(tmp_path, "run-b", 7, {"v0": ["flush", "flush"]})  # 同 seed 的第二个 run

    monkeypatch.setattr("tools.probe_seed_ensemble.REPO", tmp_path)
    members, labels, seed_of_run = load_members(["run-*"])

    assert labels == LABELS
    assert list(seed_of_run) == [7]                      # 只保留第一个
    assert list(members["v0"][7]) == [0, 0]              # 且用的是保留那个 run 的预测
    assert "跳过 seed=7" in capsys.readouterr().out


def test_paired_reports_direction_and_needs_enough_videos():
    base = {f"v{i}": {"edit": 40.0} for i in range(8)}
    better = {f"v{i}": {"edit": 40.0 + i + 1} for i in range(8)}
    outcome = paired(base, better, "edit")
    assert outcome["win"] == 8 and outcome["median"] > 0 and outcome["p"] < 0.05

    few = {f"v{i}": {"edit": 40.0} for i in range(3)}
    assert paired(few, few, "edit") is None              # n < 6 → 报样本不足


def test_paired_ignores_missing_and_tied_entries():
    base = {"v0": {"flush": 0.0}, "v1": {"flush": 0.0}, "v2": {"flush": None}, "v3": {"flush": 10.0}}
    candidate = {"v0": {"flush": 0.0}, "v1": {"flush": 5.0}, "v2": {"flush": 5.0}, "v3": {"flush": 10.0}}
    assert paired(base, candidate, "flush") is None      # 只有 1 个有效差值 → 样本不足
