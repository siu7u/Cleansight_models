"""评估矩阵入口：python -m cleansight_eval.cli.matrix --runs <dir>。

汇总 run 目录下所有信封，产出 matrix.json（机读）+ matrix.md（人读），异构
指标列并保留 N/A / MISSING / 已计算三态（需求 §9）。
"""

from __future__ import annotations

import argparse

from ..core.matrix import write_matrix


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="cleansight_eval 评估矩阵")
    p.add_argument("--runs", required=True, help="包含 *.envelope.json 的运行目录")
    p.add_argument("--out", default=None, help="矩阵输出目录，默认与 --runs 相同")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    json_path, md_path = write_matrix(args.runs, args.out)
    print(f"[matrix] {json_path}")
    print(f"[matrix] {md_path}")
    return str(json_path)


if __name__ == "__main__":
    main()
