"""训练视频子采样（``data.train_video_fraction``）：学习曲线横轴的口径与保护。

口径要求：
- 只作用于 **train** split，val/test 恒全量（否则评测口径会随训练预算漂移）；
- 取清单**前 k 个**视频（k = max(1, round(n × fraction))），同一配置每次切同一批；
- 非法比例立即报错，不静默退回全量；
- 子采样口径写进 checkpoint 数据溯源，且 resume 时不一致要拒绝（防止无声混训）。
"""

from __future__ import annotations

import numpy as np

from framework.cleansight_eval.temporal.data import (
    build_dataset_provenance,
    count_split_videos,
    load_split,
    resolve_train_video_fraction,
    resolve_train_video_limit,
    split_video_names,
)

ACTIONS = ["idle", "air_injection", "flush", "long_brush_insert", "long_brush_withdraw", "short_brush"]


def _make_dataset(root, videos: int = 6, frames: int = 24):
    """造一个最小的非 legacy ActionMixed 目录（每视频一段带动作的采样帧）。"""

    rng = np.random.default_rng(0)
    (root / "labels" / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
    (root / "labels" / "data.yaml").write_text(
        "nc: 6\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(ACTIONS)), encoding="utf-8"
    )
    # 检测映射（build_dataset_provenance 会对它取 sha256）
    (root / "frames" / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
    (root / "frames" / "data.yaml").write_text(
        "nc: 8\nnames:\n" + "".join(f"  {i}: det{i}\n" for i in range(8)), encoding="utf-8"
    )
    names = []
    for split in ("train", "test"):
        n_videos = videos if split == "train" else 2
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (root / "frames" / split).mkdir(parents=True, exist_ok=True)
        for index in range(n_videos):
            stem = f"{split}_v{index}"
            names.append(stem)
            frame_ids = list(range(1, frames * 4, 4))
            (root / "labels" / split / f"{stem}.mp4.txt").write_text(
                "\n".join(f"{fid} {int(rng.integers(0, 6))}" for fid in frame_ids) + "\n",
                encoding="utf-8",
            )
            for fid in frame_ids:
                lines = [
                    f"{int(rng.integers(0, 8))} {rng.random():.4f} {rng.random():.4f} "
                    f"{rng.random() * 0.3:.4f} {rng.random() * 0.3:.4f}"
                    for _ in range(int(rng.integers(1, 4)))
                ]
                (root / "frames" / split / f"{stem}.mp4-{fid:06d}.txt").write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )
    return names


def _data_cfg(root, **extra):
    return {"root": str(root), "split_train": "train", "split_eval": "test", **extra}


def test_video_fraction_selects_first_k_videos_deterministically(tmp_path):
    root = tmp_path / "ds"
    _make_dataset(root, videos=6)
    cfg = _data_cfg(root)
    assert count_split_videos(cfg, "train") == 6

    assert resolve_train_video_limit(cfg) is None  # 缺省 = 全量
    assert resolve_train_video_limit({**cfg, "train_video_fraction": 1.0}) is None  # 1.0 = 全量
    assert resolve_train_video_limit({**cfg, "train_video_fraction": 0.5}) == 3
    assert resolve_train_video_limit({**cfg, "train_video_fraction": 0.25}) == 2  # round(1.5)=2
    assert resolve_train_video_limit({**cfg, "train_video_fraction": 0.01}) == 1  # 至少 1 个视频

    full_names = split_video_names(cfg, "train")
    half_names = split_video_names(cfg, "train", video_limit=3)
    assert half_names == full_names[:3]

    full_feats, _, _ = load_split(cfg, "train", feature_schema={"version": "actionmixed-bbox-8cls-v1", "dim": 40})
    half_feats, half_truths, _ = load_split(
        cfg, "train", feature_schema={"version": "actionmixed-bbox-8cls-v1", "dim": 40}, video_limit=3
    )
    assert len(full_feats) == 6 and len(half_feats) == 3
    for index in range(3):
        assert np.array_equal(half_feats[index], full_feats[index]), "子采样必须取同一批视频且顺序一致"
    # val/test 不受该旋钮影响
    assert len(load_split(cfg, "test", feature_schema={"version": "actionmixed-bbox-8cls-v1", "dim": 40})[0]) == 2


def test_video_fraction_validation_rejects_bad_values(tmp_path):
    root = tmp_path / "ds"
    _make_dataset(root, videos=4, frames=12)
    cfg = _data_cfg(root)
    assert resolve_train_video_fraction({**cfg, "train_video_fraction": 0.5}) == 0.5
    for bad in (0, -0.1, 1.5, "half"):
        try:
            resolve_train_video_fraction({**cfg, "train_video_fraction": bad})
        except ValueError:
            continue
        raise AssertionError(f"非法 train_video_fraction 竟然通过: {bad!r}")


def test_video_fraction_recorded_in_provenance_and_guards_resume(tmp_path):
    root = tmp_path / "ds"
    _make_dataset(root, videos=4, frames=12)
    plain = build_dataset_provenance(_data_cfg(root), None)
    sub = build_dataset_provenance(_data_cfg(root, train_video_fraction=0.5), None)
    assert "train_video_fraction" not in plain
    assert sub["train_video_fraction"] == 0.5

    # 已登记数据集分支同样要记录（真实训练走的就是这条）。
    registered = {"dataset_ref": "temporal.actionmixed-auto-roi-v1", "split_train": "train",
                  "split_val": "val", "split_eval": "test", "root": str(root)}
    assert "train_video_fraction" not in build_dataset_provenance(registered, None)
    assert build_dataset_provenance(
        {**registered, "train_video_fraction": 0.25}, None
    )["train_video_fraction"] == 0.25

    from framework.cleansight_eval.temporal.data import assert_resume_dataset_compatible

    # 未登记数据集（合成/临时）不做版本断言，子采样口径不一致也必须拦住。
    try:
        assert_resume_dataset_compatible({"dataset": sub}, plain)
    except ValueError as exc:
        assert "train_video_fraction" in str(exc)
        return
    raise AssertionError("子采样口径不一致的 resume 必须被拒绝")
