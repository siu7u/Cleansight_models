"""数据管理 CLI：python -m framework.cleansight_eval.cli.dataset。

数据契约层（core/dataset_download.py）的下载与就绪校验入口：
  --preset all         一键下载训练所需全部数据集（yolo + actionmixed）
  --preset <key>       下载单个数据集到 datasets/ 正确位置
  --check              校验已下载数据是否就绪（缺失返回非零）
  --list-presets       列出全部数据源与目标位置
"""

from __future__ import annotations

import argparse
import sys

from ..core import dataset_download as dd


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CleanSight 数据集下载与校验")
    p.add_argument("--preset", choices=sorted(dd.DATASET_PRESETS) + ["all"], default=None,
                   help="下载指定数据集；all = 训练所需全部（yolo + actionmixed + actionmixed-auto）")
    p.add_argument("--check", action="store_true", help="校验已下载数据是否就绪")
    p.add_argument("--list-presets", action="store_true", help="列出全部数据源与目标位置")
    p.add_argument("--dataset", help="手动指定 ModelScope dataset id（覆盖 preset 的默认源）")
    p.add_argument("--output", help="本地输出目录（默认按 preset）")
    p.add_argument("--depth", type=int, default=1, help="git clone --depth（默认 1）")
    p.add_argument("--branch", default="master", help="克隆分支（默认 master）")
    p.add_argument("--skip-lfs", action="store_true", help="跳过 LFS 大文件下载")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dd.load_env(dd.REPO_ROOT / ".env")

    if args.list_presets or not (args.preset or args.check):
        print(dd.list_presets())
        return 0

    if args.check:
        missing = dd.check_data()
        if missing:
            print(f"缺失 {len(missing)} 项，请下载：")
            for key, cmd in missing:
                print(f"  {cmd}")
            return 1
        print("全部就绪 ✅")
        return 0

    if args.preset == "all":
        for key in ("yolo", "actionmixed", "actionmixed-auto"):
            dd.download_one(key, dataset=args.dataset, output=args.output,
                            branch=args.branch, depth=args.depth, skip_lfs=args.skip_lfs)
        print("\n全部数据集下载完成。可运行校验：")
        print("  python -m framework.cleansight_eval.cli.dataset --check")
        return 0

    dd.download_one(args.preset, dataset=args.dataset, output=args.output,
                    branch=args.branch, depth=args.depth, skip_lfs=args.skip_lfs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
