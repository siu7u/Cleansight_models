"""检测数据集操作：按淘汰决策生成裁剪数据集。

``build_trimmed_dataset`` 从原始 YOLO 分组数据集中移除淘汰类别：只过滤 labels
（重新映射保留类的 class id），images 用软链接复用，并写出新的 ``data.yaml``。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml


def build_trimmed_dataset(
    group_dir: Path,
    keep_classes: List[str],
    output_dir: Path,
) -> Optional[Path]:
    """
    从原始 group 数据集中移除淘汰类别，生成新的 data.yaml 和 labels。

    只修改 labels（过滤掉淘汰类的标注），images 直接软链接。返回新 data.yaml 路径。
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    raw_yaml = group_dir / "data.yaml"
    if not raw_yaml.is_file():
        raise FileNotFoundError(f"data.yaml 缺失: {raw_yaml}")
    cfg = yaml.safe_load(raw_yaml.read_text(encoding="utf-8")) or {}
    raw_names = cfg["names"]

    # data.yaml 的 names 可能是 list 或 dict{id: name}；统一成 [(old_id, name)]
    if isinstance(raw_names, dict):
        ordered = [(int(key), str(name)) for key, name in sorted(raw_names.items(), key=lambda kv: int(kv[0]))]
    else:
        ordered = [(i, str(name)) for i, name in enumerate(raw_names)]

    old_to_new: dict[int, int] = {}
    new_names: dict[int, str] = {}
    new_id = 0
    for old_id, name in ordered:
        if name in keep_classes:
            old_to_new[old_id] = new_id
            new_names[new_id] = name
            new_id += 1

    if not new_names:
        raise ValueError(f"无保留类别，无法构建裁剪数据集: {keep_classes}")

    new_yaml_data = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(new_names),
        "names": new_names,
    }
    new_yaml_path = output_dir / "data.yaml"
    new_yaml_path.write_text(
        yaml.dump(new_yaml_data, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )

    for split in ("train", "val", "test"):
        src_img_dir = group_dir / "images" / split
        src_lbl_dir = group_dir / "labels" / split
        dst_img_dir = output_dir / "images" / split
        dst_lbl_dir = output_dir / "labels" / split
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        if not src_img_dir.is_dir():
            continue

        for img_file in sorted(src_img_dir.iterdir()):
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            link = dst_img_dir / img_file.name
            if not link.exists():
                link.symlink_to(img_file.resolve())

            lbl_file = src_lbl_dir / f"{img_file.stem}.txt"
            if lbl_file.is_file():
                kept_lines = []
                for line in lbl_file.read_text(encoding="utf-8").strip().splitlines():
                    if not line.strip():
                        continue
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    old_cid = int(parts[0])
                    if old_cid in old_to_new:
                        parts[0] = str(old_to_new[old_cid])
                        kept_lines.append(" ".join(parts))
                if kept_lines:
                    (dst_lbl_dir / lbl_file.name).write_text(
                        "\n".join(kept_lines) + "\n", encoding="utf-8"
                    )

    print(f"裁剪数据集已生成: {output_dir}")
    print(f"  保留类别: {list(new_names.values())}")
    print(f"  新 data.yaml: {new_yaml_path}")
    return new_yaml_path
