#!/usr/bin/env python3
"""
组员数据工具：一键下载训练所需数据集 + 就绪校验。

用法（仓库根执行）:
    python tools/team_dataset.py --list-presets   # 查看可下载的数据源与位置
    python tools/team_dataset.py --preset all     # 下载训练所需的全部数据集
    python tools/team_dataset.py --preset yolo    # 只下载 YOLO 数据集
    python tools/team_dataset.py --check          # 校验已下载数据是否就绪

下载与校验逻辑复用根目录 download_modelscope_dataset.py；本脚本只提供
组员友好的参数入口，并导出 check_required_datasets 供 team_train 使用。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 训练所需数据集 → 校验文件
REQUIRED_FILES: dict[str, list[Path]] = {
    "yolo": [
        Path("datasets/cleansight-yolo/group1_large/data.yaml"),
        Path("datasets/cleansight-yolo/group2_small/data.yaml"),
    ],
    "actionmixed": [
        Path("datasets/cleansight-ActionMixed/labels/data.yaml"),
        Path("datasets/cleansight-ActionMixed/frames/data.yaml"),
    ],
}
DOWNLOAD_CMD = {
    "yolo": "python tools/team_dataset.py --preset yolo",
    "actionmixed": "python tools/team_dataset.py --preset actionmixed",
}


def check_required_datasets(keys: list[str] | None = None) -> list[str]:
    """校验指定数据集（默认全部）是否就绪，返回缺失的数据集 key 列表。"""

    keys = keys or list(REQUIRED_FILES)
    missing = []
    for key in keys:
        files = REQUIRED_FILES.get(key, [])
        if not all((ROOT / rel).is_file() for rel in files):
            missing.append(key)
    return missing


def run_download(preset: str) -> None:
    """调用根目录下载脚本下载单个 preset。"""

    sys.path.insert(0, str(ROOT))
    from download_modelscope_dataset import main as download_main

    sys.argv = ["download_modelscope_dataset.py", "--preset", preset]
    download_main()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="组员数据下载与校验工具")
    p.add_argument("--preset", choices=["yolo", "actionmixed", "all"], default=None,
                   help="下载指定数据集；all = yolo + actionmixed")
    p.add_argument("--check", action="store_true", help="校验已下载数据是否就绪")
    p.add_argument("--list-presets", action="store_true", help="列出可下载的数据源")
    args = p.parse_args(argv)

    if args.list_presets or not (args.preset or args.check):
        sys.path.insert(0, str(ROOT))
        from download_modelscope_dataset import list_presets

        list_presets()
        return 0

    if args.check:
        missing = check_required_datasets()
        print("数据就绪检查：")
        for key, files in REQUIRED_FILES.items():
            status = "✅" if key not in missing else "❌"
            print(f"  [{status}] {key}: {', '.join(str(f) for f in files)}")
        print()
        if missing:
            print("缺失数据集，请下载：")
            for key in missing:
                print(f"  {DOWNLOAD_CMD[key]}")
            print("\n（或在训练时加 --force 跳过检查）")
            return 1
        print("全部就绪 ✅")
        return 0

    if args.preset == "all":
        for key in ("yolo", "actionmixed"):
            run_download(key)
        print("\n全部数据集下载完成。可运行校验：python tools/team_dataset.py --check")
        return 0

    run_download(args.preset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
