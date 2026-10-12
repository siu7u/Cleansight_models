"""框集合契约（`actionmixed-boxset-v1`，96 维）与集合编码器模型（`boxset_mstcn2`）的单元测试。

覆盖：特征布局（面积降序 / one-hot / 坐标 / 零填充）、空帧、越界类忽略，
模型前向形状、**置换不变性**（槽顺序不影响输出）、空帧不产生 NaN，
以及 `load_split` 真数据端到端。
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from cleansight_eval.temporal.data import load_split
from cleansight_eval.temporal.features import (
    BOXSET_DIM,
    BOXSET_N_SLOTS,
    BOXSET_SLOT_DIM,
    BOXSET_VERSION,
    build_boxset_frame_features,
)
from cleansight_eval.temporal.models import build_model


def _write_frame(path: Path, rows) -> None:
    path.write_text("\n".join(f"{c} {cx} {cy} {w} {h}" for c, cx, cy, w, h in rows) + "\n",
                    encoding="utf-8")


def test_dim_and_missing_file(tmp_path):
    assert BOXSET_DIM == BOXSET_N_SLOTS * BOXSET_SLOT_DIM == 96
    feat = build_boxset_frame_features(tmp_path / "nope.txt")
    assert feat.shape == (BOXSET_DIM,)
    assert not feat.any()


def test_sorted_by_area_and_onehot(tmp_path):
    """框按**面积降序**填槽；槽内前 8 维是 one-hot，后 4 维是 cx,cy,w,h。"""

    path = tmp_path / "f.txt"
    _write_frame(path, [
        (0, 0.10, 0.10, 0.10, 0.10),   # 面积 0.01
        (3, 0.50, 0.50, 0.40, 0.50),   # 面积 0.20 → 应排第 0 槽
        (6, 0.90, 0.90, 0.20, 0.20),   # 面积 0.04 → 第 1 槽
    ])
    feat = build_boxset_frame_features(path).reshape(BOXSET_N_SLOTS, BOXSET_SLOT_DIM)
    assert feat[0][3] == pytest.approx(1.0)          # 第 0 槽是类 3
    assert feat[0][8:12] == pytest.approx([0.50, 0.50, 0.40, 0.50])
    assert feat[1][6] == pytest.approx(1.0)          # 第 1 槽是类 6
    assert feat[1][8:12] == pytest.approx([0.90, 0.90, 0.20, 0.20])
    assert feat[2][0] == pytest.approx(1.0)          # 第 2 槽是类 0
    assert not feat[3:].any()                        # 其余零填充


def test_out_of_range_class_ignored(tmp_path):
    path = tmp_path / "f.txt"
    _write_frame(path, [(99, 0.5, 0.5, 0.2, 0.2), (1, 0.5, 0.5, 0.2, 0.2)])
    feat = build_boxset_frame_features(path).reshape(BOXSET_N_SLOTS, BOXSET_SLOT_DIM)
    assert feat[0][1] == pytest.approx(1.0)   # 只剩类 1
    assert not feat[1].any()


def test_slot_cap(tmp_path):
    """超过槽位的框被丢弃（只保留面积最大的前 K 个）。"""

    path = tmp_path / "f.txt"
    rows = [(0, 0.1, 0.1, 0.01 * (i + 1), 0.1) for i in range(BOXSET_N_SLOTS + 3)]
    _write_frame(path, rows)
    feat = build_boxset_frame_features(path).reshape(BOXSET_N_SLOTS, BOXSET_SLOT_DIM)
    # 面积随 i 递增 → 最大的 8 个应保留下来（i = K+2 .. 3）
    assert int(feat[:, :8].sum()) == BOXSET_N_SLOTS
    # 槽内 8:12 是 (cx, cy, w, h)，面积最大的那个框 w=0.01×(K+3)
    assert feat[0][10] == pytest.approx(0.01 * (BOXSET_N_SLOTS + 3), abs=1e-6)


def _model() -> torch.nn.Module:
    return build_model({"type": "boxset_mstcn2", "input_dim": BOXSET_DIM, "num_classes": 6,
                        "hidden": 32, "num_stages": 2, "num_layers": 3, "num_slots": BOXSET_N_SLOTS,
                        "slot_dim": BOXSET_SLOT_DIM, "slot_embed": 16})


def test_model_forward_shape():
    model = _model().eval()
    with torch.no_grad():
        out = model(torch.randn(2, 25, BOXSET_DIM))
    assert out.shape == (2, 25, 6)


def test_model_permutation_invariant():
    """把同一帧内的**槽顺序打乱**，输出必须逐元素相同（集合语义）。"""

    model = _model().eval()
    torch.manual_seed(0)
    frame = torch.zeros(BOXSET_DIM)
    # 三个有效槽（类 0/3/6），带不同坐标
    for slot, (cls, box) in enumerate([(0, (0.1, 0.2, 0.3, 0.4)),
                                       (3, (0.5, 0.6, 0.3, 0.4)),
                                       (6, (0.7, 0.8, 0.2, 0.2))]):
        frame[slot * BOXSET_SLOT_DIM + cls] = 1.0
        frame[slot * BOXSET_SLOT_DIM + 8: slot * BOXSET_SLOT_DIM + 12] = torch.tensor(box)
    perm = frame.reshape(BOXSET_N_SLOTS, BOXSET_SLOT_DIM)[[2, 0, 5, 1, 7, 3, 6, 4]].reshape(-1)
    # 两帧必须放在 **batch** 维而不是时间维——否则会被 MS-TCN 的时序卷积混合，
    # 测到的就不是集合编码器的性质了（本测试初版就踩了这个坑）。
    x = torch.stack([frame, perm]).unsqueeze(1)          # [2, 1, 96]
    with torch.no_grad():
        out = model(x)
    assert torch.allclose(out[0, 0], out[1, 0], atol=1e-6)


def test_model_empty_frame_is_finite():
    """整帧无框（全零）不得产生 NaN/Inf。"""

    model = _model().eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 5, BOXSET_DIM))
    assert torch.isfinite(out).all()


def test_load_split_real_data_gives_96_dims():
    root = Path(__file__).resolve().parents[2] / "datasets/cleansight-ActionMixed-auto-lhh"
    if not root.is_dir():
        pytest.skip("本地缺少 datasets/cleansight-ActionMixed-auto-lhh")

    data_cfg = {"root": str(root), "dataset_ref": None, "split_train": "train",
                "split_val": "val", "split_eval": "test"}
    feats, truths, id2name = load_split(
        data_cfg, "test", feature_schema={"dim": BOXSET_DIM, "version": BOXSET_VERSION})
    assert len(feats) == 8
    assert all(f.shape[1] == BOXSET_DIM for f in feats)
    assert sum(len(t) for t in truths) == 2639
    assert len(id2name) == 6
