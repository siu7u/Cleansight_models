"""tools/probe_channel_subsets.py：ROI 通道语义 / 区域子集消融探针的单元测试。

验收标准（verification-first）：
- 通道索引严格按契约布局 `k*block + r*channels + c`（block = rows*cols*channels，区域行优先）；
- 子集大小符合算术：8 类 × 6 区域 × 3 通道 = 144；单通道子集 48；presence+count 96；单区域 24；
- 子集互不重叠、并集等于全集（防"漏掉/重复取通道"这类静默错误）；
- 非网格契约（如 bbox-40）给出明确报错，而不是算出一堆错索引；
- 子集间的逐视频配对检验方向与显著性判定正确（合成数据）。

不加载真实数据集：`load_sequences` 之外的部分全部纯函数。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.probe_channel_subsets import (  # noqa: E402
    CHANNEL_NAMES,
    channel_indices,
    main,
    median_of,
    paired,
)

N_BLOCKS = 8   # roi-grid-v1 的检测类数
ROWS, COLS, CHANNELS = 2, 3, 3
BLOCK = ROWS * COLS * CHANNELS  # 18


def test_block_layout_matches_contract_dimension():
    assert BLOCK == 18
    assert N_BLOCKS * BLOCK == 144  # 与 catalog 的 actionmixed-roi-grid-v1 input_dim 一致


def test_all_subset_is_identity():
    indices = channel_indices(N_BLOCKS, ROWS, COLS, CHANNELS)
    assert list(indices) == list(range(144))


def test_single_channel_subsets_have_expected_size_and_layout():
    for index, name in enumerate(CHANNEL_NAMES):
        indices = channel_indices(N_BLOCKS, ROWS, COLS, CHANNELS, keep_channels=[index])
        assert len(indices) == N_BLOCKS * ROWS * COLS == 48, name
        assert ((indices % CHANNELS) == index).all(), name
        # 每类块内、每区域都恰好取到该通道
        for k in range(N_BLOCKS):
            block = indices[(indices >= k * BLOCK) & (indices < (k + 1) * BLOCK)]
            assert len(block) == ROWS * COLS
            assert sorted((b - k * BLOCK) % CHANNELS for b in block) == [index] * (ROWS * COLS)


def test_presence_plus_count_and_region_subsets_are_disjoint_and_cover():
    presence = set(channel_indices(N_BLOCKS, ROWS, COLS, CHANNELS, keep_channels=[0]).tolist())
    count = set(channel_indices(N_BLOCKS, ROWS, COLS, CHANNELS, keep_channels=[1]).tolist())
    area = set(channel_indices(N_BLOCKS, ROWS, COLS, CHANNELS, keep_channels=[2]).tolist())
    both = set(channel_indices(N_BLOCKS, ROWS, COLS, CHANNELS, keep_channels=[0, 1]).tolist())
    assert both == presence | count and not (presence & count)
    assert len(both) == 96
    assert presence | count | area == set(range(144))  # 三通道并集 = 全集

    regions = [set(channel_indices(N_BLOCKS, ROWS, COLS, CHANNELS, keep_regions=[r]).tolist())
               for r in range(ROWS * COLS)]
    assert all(len(region) == N_BLOCKS * CHANNELS == 24 for region in regions)
    union = set().union(*regions)
    assert union == set(range(144))
    assert sum(len(region) for region in regions) == 144  # 两两不交


def test_non_grid_contract_is_rejected(tmp_path, monkeypatch, capsys):
    """bbox-40（无 layout）必须明确报错，不能静默算错索引。"""
    fake_run = tmp_path / "fake-run"
    fake_run.mkdir()
    (fake_run / "config.resolved.json").write_text(
        '{"schema_version": 1, "pipeline": "full_sequence_temporal",'
        ' "model": {"type": "mstcn", "num_classes": 6, "input_dim": 40},'
        ' "data": {"split_train": "train", "split_eval": "test"},'
        ' "feature_schema": {"dim": 40, "version": "actionmixed-bbox-8cls-v1"}}'
    )
    monkeypatch.setattr("tools.probe_channel_subsets.REPO", tmp_path)
    # load_split 不会被走到（维度校验先失败），因此无需真实数据
    with pytest.raises(SystemExit, match="网格布局契约"):
        main(["--config-run", "fake-run"])


def test_paired_reports_direction_and_flags_small_samples():
    base = {f"v{i}": {"edit": 10.0} for i in range(8)}
    better = {f"v{i}": {"edit": 10.0 + i + 1} for i in range(8)}
    outcome = paired(base, better, "edit")
    assert outcome["n"] == 8 and outcome["win"] == 8
    assert outcome["median"] > 0 and outcome["p"] < 0.05

    trimmed = {f"v{i}": {"edit": 10.0} for i in range(3)}
    assert paired(trimmed, trimmed, "edit") is None  # n < 6 → 报样本不足


def test_median_of_skips_missing_values():
    per_video = {"v0": {"recall:x": 10.0}, "v1": {"recall:x": None}, "v2": {"recall:x": 30.0}}
    assert median_of(per_video, "recall:x") == pytest.approx(20.0)
    assert median_of({"v0": {"recall:y": None}}, "recall:y") is None


def test_presence_only_layout_yields_48_dims():
    """presence-only 契约（rows=2, cols=3, channels=1）：全量 48 维、单区域 8 维。"""
    assert len(channel_indices(8, 2, 3, 1)) == 48
    region0 = channel_indices(8, 2, 3, 1, keep_regions=[0])
    assert len(region0) == 8
    assert list(region0) == [0, 6, 12, 18, 24, 30, 36, 42]  # 每类块宽 6，区域行优先
    only_channel = channel_indices(8, 2, 3, 1, keep_channels=[0])
    assert list(only_channel) == list(range(48))  # 单通道契约里 channel 子集 == 全集


def test_channel_redundancy_detects_correlated_and_independent_channels():
    """两通道完全相关 → |r|=1；恒零通道被跳过而不是算成 nan/0。"""
    from tools.probe_channel_subsets import channel_redundancy

    rng = np.random.default_rng(0)
    n_frames = 500
    # 单块（1 类 × 1 区域 × 3 通道）：channel0 与 channel1 完全相关，channel2 与它们独立
    base = rng.normal(size=n_frames)
    independent = rng.normal(size=n_frames)
    frames = np.stack([base, base * 3.0 + 1.0, independent], axis=1)
    stats = channel_redundancy([frames], n_blocks=1, rows=1, cols=1, channels=3)
    assert stats["presence~count"]["mean_abs_r"] == pytest.approx(1.0, abs=1e-6)
    assert stats["presence~max_area"]["mean_abs_r"] < 0.2

    zero_channel = np.stack([base, base, np.zeros(n_frames)], axis=1)
    stats = channel_redundancy([zero_channel], n_blocks=1, rows=1, cols=1, channels=3)
    assert stats["count~max_area"]["n_pairs"] == 0  # 恒零通道 → 无有效对照，不当成 0 相关
