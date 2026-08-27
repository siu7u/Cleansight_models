#!/usr/bin/env python3
"""
通过 ModelScope git clone + git-lfs 拉取数据集到本地 datasets/ 目录（薄封装）。

实现已迁移到 framework 数据契约层（``framework.cleansight_eval.core.dataset_download``），
本脚本保留为向后兼容入口；新命令请使用：

    python -m framework.cleansight_eval.cli.dataset --preset all
    python -m framework.cleansight_eval.cli.dataset --check
    python -m framework.cleansight_eval.cli.dataset --list-presets
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "framework"))


def main() -> None:
    from framework.cleansight_eval.cli.dataset import main as dataset_main

    # 透传参数：--preset/--dataset/--output/--depth/--branch/--skip-lfs
    dataset_main()


if __name__ == "__main__":
    main()
