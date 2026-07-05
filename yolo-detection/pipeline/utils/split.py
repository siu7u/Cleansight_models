#!/usr/bin/env python3
"""
稳定切分:splits.yaml 是 视频->split 的唯一真源。

设计目标(用户明确要求"必须是稳定切分"):
  - 同一视频永远同一 split,可复现。
  - 新增视频不打乱已有分配(增量友好)。
  - 一个视频的所有帧只进它的 split,绝不跨 split -> 杜绝时间相邻泄漏。
  - 人工可覆盖(手动改 splits.yaml 里的值,永不被自动重排)。

未归属视频用 hash(seed:stem) 确定性落到 train/val,由 assign() 显式回填并写回 yaml
(不在 build 里静默改动)。
"""
import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent  # yolo_pipeline/(自包含)
SPLITS_PATH = Path(__file__).resolve().parent.parent / "splits.yaml"

VALID_SPLITS = {"train", "val", "test", "e2e_test"}
# 参与训练/验证的 split(test / e2e_test 留给端到端评测,不进 YOLO 数据集)
TRAINVAL_SPLITS = {"train", "val"}


def stem_of(name: str) -> str:
    """视频文件名 -> splits.yaml 的 key(去扩展名)。"""
    return Path(name).stem


def load(splits_path: Path = SPLITS_PATH) -> dict:
    if not splits_path.exists():
        return {"val_ratio": 0.2, "seed": 1337, "assignments": {}}
    data = yaml.safe_load(splits_path.read_text(encoding="utf-8")) or {}
    data.setdefault("assignments", {})
    data.setdefault("val_ratio", 0.2)
    data.setdefault("seed", 1337)
    return data


def save(data: dict, splits_path: Path = SPLITS_PATH) -> None:
    """写回,保留顶部注释块(以 # 开头的行)。"""
    header = []
    if splits_path.exists():
        for line in splits_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or line.strip() == "":
                header.append(line)
            else:
                break
    assignments = data.get("assignments", {})
    body = [
        f"val_ratio: {data.get('val_ratio', 0.2)}",
        f"seed: {data.get('seed', 1337)}",
        "",
        "assignments:",
    ]
    for stem in sorted(assignments):
        body.append(f"  {stem}: {assignments[stem]}")
    text = "\n".join(header + body) + "\n"
    splits_path.write_text(text, encoding="utf-8")


def deterministic_split(stem: str, seed: int, val_ratio: float) -> str:
    """hash(seed:stem) -> train/val,确定性、稳定。"""
    h = hashlib.sha1(f"{seed}:{stem}".encode("utf-8")).hexdigest()
    bucket = int(h, 16) % 100
    return "val" if bucket < round(val_ratio * 100) else "train"


def get_split(stem: str, data: dict):
    """返回已登记的 split;未登记返回 None(不隐式分配)。"""
    return data.get("assignments", {}).get(stem)


def assign(stems, data: dict) -> list:
    """
    给未归属的 stems 确定性回填 split(写进 data['assignments'],调用方负责 save)。
    返回新增分配 [(stem, split), ...]。已归属的保持不动。
    """
    seed = data.get("seed", 1337)
    val_ratio = data.get("val_ratio", 0.2)
    assignments = data.setdefault("assignments", {})
    added = []
    for stem in stems:
        if stem in assignments:
            continue
        split = deterministic_split(stem, seed, val_ratio)
        assignments[stem] = split
        added.append((stem, split))
    return added
