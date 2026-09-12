#!/usr/bin/env python3
"""按数据集标签采样的帧号从源视频抽取帧图（S2/S3 特征管线第 1 步）。

见 docs/FEATURE_SCHEME_S2_S3_SPEC.md §3.1。对每个视频读取
labels/<split>/<视频>.mp4.txt 的 1-based frame_id 集合，用 OpenCV 精确抽取对应帧，
存 images/<split>/<stem>-<帧号:06d>.jpg（与 extract_embeddings.py 的帧图查找约定
一致）；产出 images_manifest.json（逐视频 帧数/缺失清单）。帧图目录不入库。

用法：
    python tools/extract_sampled_frames.py \
        --videos-dir outputs/videos-p16 \
        --root datasets/cleansight-ActionMixed-auto \
        --splits train val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

sys_inserted = False
if True:
    import sys
    root = Path(__file__).resolve().parents[1]
    for p in (str(root), str(root / "framework")):
        if p not in sys.path:
            sys.path.insert(0, p)

import cv2  # 重依赖在函数内 import（仓库惯例）


def label_frame_ids(labels_file: Path) -> list[int]:
    ids = []
    for line in labels_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            ids.append(int(parts[0]))
    return sorted(set(ids))


def extract_video(
    video_path: Path, frame_ids: list[int], images_split_dir: Path, stem: str
) -> tuple[int, list[int]]:
    """抽取缺失帧；返回 (已就绪帧数, 仍缺失的帧号列表)。"""

    images_split_dir.mkdir(parents=True, exist_ok=True)

    def out_path(frame_id: int) -> Path:
        return images_split_dir / f"{stem}-{frame_id:06d}.jpg"

    todo = [fid for fid in frame_ids if not out_path(fid).exists()]
    if not todo:
        return len(frame_ids), []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    wanted = set(todo)
    extracted = []
    max_id = max(wanted)
    index = 0
    while index <= max_id:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1  # 帧号 1-based，与标签 frame_id 对齐
        if index in wanted:
            cv2.imwrite(str(out_path(index)), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted.append(index)
    cap.release()
    return len(frame_ids) - len(wanted) + len(extracted), sorted(wanted - set(extracted))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--root", required=True, help="数据集 root（含 labels/，写 images/）")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()

    root = Path(args.root)
    videos_dir = Path(args.videos_dir)
    manifest_path = root / "images_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    for split in args.splits:
        for labels_file in sorted((root / "labels" / split).glob("*.txt")):
            name = labels_file.stem  # "<视频>.mp4"
            video_path = videos_dir / name
            if not video_path.exists():
                manifest[name] = {"split": split, "error": "video missing"}
                print(f"[miss] {name}")
                continue
            ids = label_frame_ids(labels_file)
            done, missing = extract_video(video_path, ids, root / "images" / split, name)
            manifest[name] = {
                "split": split,
                "label_frames": len(ids),
                "extracted": done,
                "missing": missing,
            }
            print(f"[ok] {name}: {done}/{len(ids)} frames, missing={len(missing)}")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
