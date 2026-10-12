"""手部区域 recipe v2（并集锚点）的 feature 层单元测试。

覆盖：**单 hand 框时与 v1 逐位等价**（并集 = 那个框）、**多 hand 框时区域确实变大**（并集语义）、
扩张倍数只缩放区域不改变"看哪些框"的相对结构、空帧/无 hand 全零、类级遮罩、
以及**真数据端到端宽度 == 契约维度**（40 / 80 两条分派链）。
"""

from pathlib import Path

import numpy as np
import pytest

from cleansight_eval.temporal.data import load_split
from cleansight_eval.temporal.features import (
    GLOBAL_HAND_V2_DIM,
    GLOBAL_HAND_V2_EXPAND,
    GLOBAL_HAND_V2_VERSION,
    HAND_BBOX_V2_VERSION,
    HAND_V2_DIM,
    HAND_V2_EXPAND,
    build_hand_frame_features,
    build_hand_v2_frame_features,
)


def _write_frame(path: Path, rows) -> None:
    path.write_text("\n".join(f"{c} {cx} {cy} {w} {h}" for c, cx, cy, w, h in rows) + "\n",
                    encoding="utf-8")


def test_dims_and_expands():
    """契约维度与扩张倍数固定：40 维/2.0（仅手部）、80 维/1.5（全局+手部）。"""

    assert HAND_V2_DIM == 40 and HAND_V2_EXPAND == 2.0
    assert GLOBAL_HAND_V2_DIM == 80 and GLOBAL_HAND_V2_EXPAND == 1.5


def test_single_hand_matches_v1(tmp_path):
    """只有一个 hand 框时，并集区域 ≡ 该框，故 v2(expand=1.5) 与 v1 逐位相同。"""

    path = tmp_path / "f.txt"
    _write_frame(path, [(0, 0.5, 0.5, 0.2, 0.1), (1, 0.55, 0.5, 0.05, 0.05)])
    v1 = build_hand_frame_features(path)
    v2 = build_hand_v2_frame_features(path, expand=1.5)
    assert v1.shape == v2.shape == (HAND_V2_DIM,)
    assert np.allclose(v1, v2)


def test_union_covers_both_hands(tmp_path):
    """两个 hand 框（左右各一）时，区域覆盖两者的并集 → 远处的手也被编码进去。

    构造：左手在 x=0.2、右手在 x=0.8，一个器械框在 x=0.8 附近。
    v1（最大框锚点，两者同尺寸 → 取先出现的那个）区域只在左侧，器械框落在区域外，
    对应类通道全零；v2（并集）区域横跨两侧，器械框被算进来。
    """

    path = tmp_path / "f.txt"
    _write_frame(path, [(0, 0.2, 0.5, 0.1, 0.1), (0, 0.8, 0.5, 0.1, 0.1), (1, 0.85, 0.5, 0.05, 0.05)])
    v1 = build_hand_frame_features(path).reshape(8, 5)
    v2 = build_hand_v2_frame_features(path, expand=1.5).reshape(8, 5)
    assert v1[1, 0] == 0.0            # v1：器械在最大 hand 框区域外 → 该类无 presence
    assert v2[1, 0] == 1.0            # v2：并集区域覆盖右手 → 器械被编码
    # 两个 hand 框都在并集区域内，最大面积者被编码为 hand 类
    assert v2[0, 0] == 1.0


def test_missing_file_is_all_zero(tmp_path):
    """无 hand 框 / 文件不存在 → 整帧全零（与 v1 同语义）。"""

    missing = tmp_path / "nope.txt"
    assert not build_hand_v2_frame_features(missing).any()
    path = tmp_path / "no_hand.txt"
    _write_frame(path, [(3, 0.5, 0.5, 0.2, 0.2)])
    assert not build_hand_v2_frame_features(path).any()


def test_mask_zeroes_target_block(tmp_path):
    """类级遮罩：命中的类整块为零，其它类不受影响。"""

    path = tmp_path / "f.txt"
    _write_frame(path, [(0, 0.5, 0.5, 0.2, 0.2), (1, 0.5, 0.5, 0.1, 0.1)])
    masked = build_hand_v2_frame_features(path, mask_target_ids=frozenset({1})).reshape(8, 5)
    assert masked[1].sum() == 0.0
    assert masked[0, 0] == 1.0


@pytest.mark.parametrize("version,dim,dim_const", [
    (HAND_BBOX_V2_VERSION, 40, HAND_V2_DIM),
    (GLOBAL_HAND_V2_VERSION, 80, GLOBAL_HAND_V2_DIM),
])
def test_load_split_real_data_width_matches_contract(version, dim, dim_const):
    """真数据端到端：加载器分派出的宽度 == 契约维度、帧数 == v4.1 test 的 5,272。

    这是**加载器分派**层的断言：只测特征构造器测不到"分派链被顶掉、静默回落到 40 维 bbox"。
    """

    root = Path(__file__).resolve().parents[2] / "datasets/cleansight-ActionMixed-auto-lhh"
    if not root.is_dir():
        pytest.skip("本地缺少 datasets/cleansight-ActionMixed-auto-lhh")

    data_cfg = {"root": str(root), "dataset_ref": None, "split_train": "train",
                "split_val": "val", "split_eval": "test"}
    feats, truths, id2name = load_split(
        data_cfg, "test", feature_schema={"dim": dim, "version": version})
    assert dim == dim_const
    assert len(feats) == 8
    assert all(f.shape[1] == dim_const for f in feats)
    assert sum(len(t) for t in truths) == 5272  # 数据 v4.1（15fps）test split
    assert len(id2name) == 6
