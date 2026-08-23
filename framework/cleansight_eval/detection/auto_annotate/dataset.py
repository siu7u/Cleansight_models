"""图像帧序列数据集 → 时序训练数据（图片自动标注，服务时序模型训练链）。

输入：数据集根目录（Ultralytics 布局）——
- ``images/<split>/<序列>-<帧号:06d>.<ext>``：有序帧序列（帧号 6 位，序列名
  含 ``.mp4`` 时与视频链命名一致，如 ``05ba4406-xxx.mp4-000141.jpg``）
- ``labels/<split>/<序列>.txt``：动作标签，每行 ``"frame_id action_id"``
  （时序训练必需，来自人工标注；图片本身无法生成动作标签）

输出：与 ``convert``（视频链）完全同构的时序训练数据布局，可直接被
``temporal/data.py`` 消费：
- ``<out>/frames/<split>/<序列>-<帧号:06d>.txt``：逐帧 YOLO bbox
  ``"class_id cx cy w h"``（8 类全局编号，仅覆盖有动作标签的帧）
- ``<out>/labels/<split>/<序列>.txt``：动作标签原样复制
- ``<out>/labels/data.yaml`` + ``<out>/frames/data.yaml``：类别映射（缺省补写）

模型加载与批量推理复用 ``run``（``_load_models`` / ``_infer_batch`` /
``_normalize_conf``），不直接 import ultralytics。
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import cv2
import yaml

from ._constants import ACTION_CLASSES, DETECTION_CLASSES
from .run import _infer_batch, _load_models, _normalize_conf

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_FRAME_SUFFIX = re.compile(r"^(.*)-(\d{6})$")


def _parse_seq_frame(stem: str) -> tuple[str, int]:
    """从图片文件名解析 ``(序列名, 帧号)``：末尾 ``-<6 位帧号>`` 之前的部分是序列名。

    帧号不足 6 位或缺失时报错（有序帧序列必须带帧号，保证帧序与时间轴一致）。
    """

    match = _FRAME_SUFFIX.fullmatch(stem)
    if match is None:
        raise ValueError(
            f"图片文件名缺少 '-<帧号:06d>' 后缀（如 demo.mp4-000001.jpg）: {stem}"
        )
    return match.group(1), int(match.group(2))


def _write_mapping_if_missing(out_root: Path) -> None:
    """补写 labels/data.yaml（6 类动作）与 frames/data.yaml（8 类检测），已有不覆盖。"""

    labels_yaml = out_root / "labels" / "data.yaml"
    if not labels_yaml.is_file():
        labels_yaml.parent.mkdir(parents=True, exist_ok=True)
        labels_yaml.write_text(
            yaml.safe_dump(
                {"nc": len(ACTION_CLASSES), "names": {i: n for i, n in enumerate(ACTION_CLASSES)}},
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
    frames_yaml = out_root / "frames" / "data.yaml"
    if not frames_yaml.is_file():
        frames_yaml.parent.mkdir(parents=True, exist_ok=True)
        frames_yaml.write_text(
            yaml.safe_dump(
                {"nc": len(DETECTION_CLASSES), "names": {i: n for i, n in enumerate(DETECTION_CLASSES)}},
                allow_unicode=True,
            ),
            encoding="utf-8",
        )


def run_dataset_annotate(
    dataset_root: Path,
    checkpoint_specs: list[dict],
    out_root: Path,
    *,
    imgsz: int = 640,
    conf: float | dict = 0.25,
    batch_size: int = 16,
    runs_dir: Path | None = None,
    resume: bool = False,
) -> list[Path]:
    """主入口：图片帧序列数据集 + checkpoint 配置 → 时序训练数据（frames/ + labels/）。

    按 ``images/<split>/`` 的每个 split，把图片按文件名解析为 ``(序列, 帧号)``，
    只对**有动作标签的帧**做 YOLO 检测并写出 ``frames/<split>/<序列>-<帧号:06d>.txt``
    （bbox 行 ``class_id cx cy w h``，class_id 为 8 类全局编号，类别表外检测丢弃）；
    动作标签文件原样复制到 ``<out>/labels/<split>/``。``conf`` 支持标量或
    ``{类别名: 阈值}``；``resume=True`` 时跳过已完成序列/帧。返回产出文件列表。
    """

    dataset_root = Path(dataset_root)
    out_root = Path(out_root)
    images_root = dataset_root / "images"
    labels_root = dataset_root / "labels"
    if not images_root.is_dir():
        raise FileNotFoundError(f"数据集缺少 images/ 目录: {images_root}")
    if not labels_root.is_dir():
        raise FileNotFoundError(f"数据集缺少 labels/ 目录（时序训练必需动作标签）: {labels_root}")
    infer_conf, class_conf = _normalize_conf(conf)
    models = _load_models(checkpoint_specs, imgsz=imgsz, runs_dir=runs_dir)
    class_to_id = {name: cid for cid, name in enumerate(DETECTION_CLASSES)}

    splits = sorted(p.name for p in images_root.iterdir() if p.is_dir())
    if not splits:
        raise ValueError(f"images/ 下没有 split 目录: {images_root}")

    outputs: list[Path] = []
    for split in splits:
        labels_split = labels_root / split
        if not labels_split.is_dir():
            raise FileNotFoundError(
                f"labels/ 缺少 split 目录（无动作标签无法训练时序模型）: {labels_split}"
            )

        # 图片按 (序列, 帧号) 索引
        by_seq: dict[str, dict[int, Path]] = {}
        for image in sorted(images_root.glob(f"{split}/*")):
            if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            seq, frame = _parse_seq_frame(image.stem)
            by_seq.setdefault(seq, {})[frame] = image

        for seq in sorted(by_seq):
            label_file = labels_split / f"{seq}.txt"
            if not label_file.is_file():
                raise FileNotFoundError(f"序列 {seq} 缺少动作标签: {label_file}")
            frame_ids = []
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 2:
                    frame_ids.append(int(parts[0]))
            if not frame_ids:
                continue  # 与 data.py 一致：无有效标签行的空文件跳过
            images = {}
            for frame in frame_ids:
                image = by_seq.get(seq, {}).get(frame)
                if image is None:
                    raise FileNotFoundError(
                        f"{seq} 标签帧 {frame} 在 images/{split}/ 中无对应图片"
                        f"（需命名 <序列>-<帧号:06d>.<ext>）"
                    )
                images[frame] = image

            frames_dir = out_root / "frames" / split
            if resume and all(
                (frames_dir / f"{seq}-{frame:06d}.txt").is_file() for frame in frame_ids
            ):
                print(f"[auto-annotate] 跳过（已存在）: {split}/{seq}")
                continue

            # 按标签帧序批量推理（只覆盖有动作标签的帧，与 convert 的 frames/ 语义一致）
            pending: list[tuple[int, object]] = []
            results: dict[int, list[dict]] = {}

            def flush() -> None:
                if not pending:
                    return
                indices = [index for index, _ in pending]
                frames_bgr = [frame for _, frame in pending]
                batch_results = _infer_batch(
                    models,
                    frames_bgr,
                    imgsz=imgsz,
                    conf=infer_conf,
                    track=False,
                    class_conf=class_conf,
                )
                for index, detections in zip(indices, batch_results):
                    results[index] = detections
                pending.clear()

            for index, frame in enumerate(frame_ids):
                target = frames_dir / f"{seq}-{frame:06d}.txt"
                if resume and target.is_file():
                    outputs.append(target)
                    continue
                image = cv2.imread(str(images[frame]))
                if image is None:
                    print(f"[auto-annotate] 跳过（无法读取）: {images[frame].name}")
                    continue
                pending.append((index, image))
                if len(pending) >= batch_size:
                    flush()
            flush()

            for index, frame in enumerate(frame_ids):
                detections = results.get(index)
                if detections is None:
                    continue
                lines = []
                for detection in detections:
                    class_id = class_to_id.get(detection["class"])
                    if class_id is None:
                        continue  # 8 类全局表外的检测丢弃（与 convert 的 class_to_id 语义一致）
                    cx, cy, width, height = detection["xywhn"]
                    lines.append(f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}\n")
                frame_path = frames_dir / f"{seq}-{frame:06d}.txt"
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                frame_path.write_text("".join(lines), encoding="utf-8")
                outputs.append(frame_path)
            print(f"[auto-annotate] {split}/{seq}: {len(frame_ids)} 帧 → frames/{split}/")

        # 动作标签复制到输出根（data.py 从输出根读 labels/）；原地输出时跳过自复制
        for label_file in sorted(labels_split.glob("*.txt")):
            dest = out_root / "labels" / split / label_file.name
            if dest == label_file:
                continue  # --out 默认数据集根：标签已在原地
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label_file, dest)
            outputs.append(dest)

    _write_mapping_if_missing(out_root)
    return outputs
