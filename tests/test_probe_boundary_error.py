"""tools/probe_boundary_error.py：段级误差分解（标签错 vs 边界错）的单元测试。

验收标准（verification-first）：
- `segments` 正确切出极大连续段（含单帧段与整段同标签）；
- `analyse_video` 的三类判定：标签错 / 标签对且完全重合 / 标签对但边界偏移——
  尤其**标签错时 IoU 恒为 0**（无论边界多准都不可能匹配，这是分解的正确性前提）；
- 边界偏移符号正确（预测早于真值 → 负偏移）；
- `summarise` 的算术：fn = 总段数 − 匹配数 = 标签错 + 标签对但边界不达标（两路相加必须闭合）；
- 时长统计与按类明细正确。

纯函数测试，不依赖真实数据集与 checkpoint。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.probe_boundary_error import (  # noqa: E402
    analyse_video,
    boundary_distance_profile,
    bucket_of,
    profile_summary,
    segments,
    summarise,
)

LABELS = ["idle", "flush", "insert"]


def ids(*names: str) -> np.ndarray:
    return np.array([LABELS.index(name) for name in names], dtype=int)


def test_segments_splits_runs_including_single_frame():
    assert segments(ids("idle", "idle", "flush", "idle")) == [(0, 0, 2), (1, 2, 3), (0, 3, 4)]
    assert segments(ids("flush")) == [(1, 0, 1)]
    assert segments(np.array([], dtype=int)) == []


def test_label_correct_and_perfect_boundary():
    truth = ids("idle", "flush", "flush", "idle")
    pred = ids("idle", "flush", "flush", "idle")

    records = analyse_video(pred, truth, LABELS)

    flush = next(r for r in records if r["class"] == "flush")
    assert flush["label_correct"] is True
    assert flush["iou"] == pytest.approx(1.0)
    assert flush["start_offset"] == 0 and flush["end_offset"] == 0


def test_label_wrong_yields_zero_iou_even_with_perfect_boundaries():
    """真值段内预测成别的类 → 标签错，IoU 记 0（不许因为边界对齐就算匹配）。"""

    truth = ids("idle", "flush", "flush", "idle")
    pred = ids("idle", "insert", "insert", "idle")

    records = analyse_video(pred, truth, LABELS)

    flush = next(r for r in records if r["class"] == "flush")
    assert flush["label_correct"] is False
    assert flush["iou"] == 0.0
    assert flush["start_offset"] is None and flush["end_offset"] is None


def test_boundary_offset_sign_and_iou_when_shifted():
    """预测段整体右移 1 帧：首偏移 +1（预测偏晚）、尾偏移 +1，IoU < 1。"""

    truth = ids("idle", "flush", "flush", "flush", "idle")
    pred = ids("idle", "idle", "flush", "flush", "flush")

    flush = next(r for r in analyse_video(pred, truth, LABELS) if r["class"] == "flush")

    assert flush["label_correct"] is True  # 段内多数仍是 flush（2/3）
    assert flush["start_offset"] == 1 and flush["end_offset"] == 1
    assert 0.0 < flush["iou"] < 1.0


def test_late_boundary_gives_negative_start_offset():
    truth = ids("idle", "flush", "flush", "flush", "idle")
    pred = ids("flush", "flush", "flush", "idle", "idle")

    flush = next(r for r in analyse_video(pred, truth, LABELS) if r["class"] == "flush")

    assert flush["label_correct"] is True
    assert flush["start_offset"] == -1  # 预测比真值早 1 帧


def test_summarise_arithmetic_closes():
    """fn 必须闭合：总段数 − 匹配数 == 标签错 + 标签对但边界不达标。

    真值 = idle×3 + flush×13（两个段）：
    - perfect：两段全中；
    - shifted（idle×9 + flush×7）：flush 真值段内仍以 flush 为多数（7/13）但 IoU≈0.54 刚过 0.5，
      idle 真值段被拉长到 IoU≈0.33 → 0.25 匹配、0.5 不匹配；
    - wrong（全 insert）：两段都标签错。
    """

    truth = ids(*(["idle"] * 3 + ["flush"] * 13))
    perfect = truth.copy()
    shifted = ids(*(["idle"] * 9 + ["flush"] * 7))
    wrong = ids(*(["insert"] * 16))

    records = analyse_video(perfect, truth, LABELS) + analyse_video(shifted, truth, LABELS) \
        + analyse_video(wrong, truth, LABELS)
    summary = summarise(records, LABELS)

    assert summary["truth_segments"] == 6  # 每条视频 2 个真值段 × 3
    for threshold in (0.25, 0.50):
        key = f"{threshold:.2f}"
        assert summary[f"fn@{key}"] == summary["truth_segments"] - summary[f"matched@{key}"]
        assert summary[f"fn@{key}"] == summary["label_wrong"] + summary[f"boundary_miss@{key}"]
    assert summary["label_wrong"] == 2           # wrong 那条视频的 2 个段
    assert summary["matched@0.50"] == 3          # perfect 2 + shifted 的 flush 1
    assert summary["matched@0.25"] == 4          # perfect 2 + shifted 2
    assert summary["boundary_miss@0.50"] == 1    # shifted 的 idle 段（IoU≈0.33）
    assert summary["boundary_miss@0.25"] == 0


def test_summarise_duration_and_per_class_stats():
    truth = ids(*(["idle"] * 2 + ["flush"] * 3 + ["idle"] * 2))  # 段长 [2, 3, 2]
    records = analyse_video(truth, truth, LABELS)

    summary = summarise(records, LABELS)

    assert summary["duration"]["median"] == pytest.approx(2)
    assert summary["duration"]["p25"] == pytest.approx(2)
    assert summary["duration"]["p75"] == pytest.approx(2.5)
    assert summary["per_class"]["flush"]["truth_segments"] == 1
    assert summary["per_class"]["flush"]["label_wrong"] == 0
    assert summary["per_class"]["flush"]["matched@0.50"] == 1
    assert summary["per_class"]["flush"]["median_duration"] == 3
    assert summary["boundary_offset_frames"]["median_abs_start"] == 0


def test_boundary_distance_profile_counts_distance_to_label_changes_only():
    """距离只由**标签切换点**决定：整段同标签时全部落最远桶（视频首尾不算边界）。"""

    truth = ids(*(["idle"] * 5 + ["flush"] * 5))  # 唯一的切换点在 index 5

    profile = boundary_distance_profile(truth, truth)

    assert profile[5][0] == 0  # 切换点本身
    assert profile[4][0] == 1 and profile[6][0] == 1
    assert profile[0][0] == 5 and profile[9][0] == 4
    assert all(correct for _distance, correct in profile)

    flat = boundary_distance_profile(ids(*(["idle"] * 4)), ids(*(["idle"] * 4)))
    assert [distance for distance, _c in flat] == [4, 4, 4, 4]  # 无切换 → 距离取序列长度


def test_profile_summary_buckets_and_shares():
    truth = ids(*(["idle"] * 3 + ["flush"] * 3))
    pred = truth.copy()
    pred[2] = LABELS.index("insert")   # 距切换点 1 帧处的一个错误
    pred[4] = LABELS.index("insert")   # 距切换点 1 帧处的另一个错误

    summary = profile_summary(boundary_distance_profile(pred, truth))

    assert summary["total_frames"] == 6 and summary["total_errors"] == 2
    # 切换点在 index 3 → 距离 [3,2,1,0,1,2] → 0-1 桶 = index 2,3,4（3 帧）；2-3 桶 = index 0,1,5（3 帧）
    assert summary["buckets"]["0-1"]["frames"] == 3
    assert summary["buckets"]["0-1"]["errors"] == 2
    assert summary["buckets"]["0-1"]["error_rate"] == pytest.approx(200.0 / 3)
    assert summary["buckets"]["0-1"]["share_of_errors"] == pytest.approx(100.0)
    assert summary["buckets"]["2-3"]["frames"] == 3 and summary["buckets"]["2-3"]["errors"] == 0


def test_bucket_of_boundaries():
    assert bucket_of(0) == "0-1" and bucket_of(1) == "0-1"
    assert bucket_of(2) == "2-3" and bucket_of(3) == "2-3"
    assert bucket_of(4) == "4-7" and bucket_of(7) == "4-7"
    assert bucket_of(8) == "8-15" and bucket_of(15) == "8-15"
    assert bucket_of(16) == "16+" and bucket_of(999) == "16+"
