"""检测优化实验编排 CLI：python -m framework.cleansight_eval.cli.sweep。

用法:
    python -m framework.cleansight_eval.cli.sweep --group group1_large --preset large_baseline large_s
    python -m framework.cleansight_eval.cli.sweep --group group1_large --grid models resolutions --dry-run
    python -m framework.cleansight_eval.cli.sweep --group group2_small --preset small_s_1280_p2 --device 0
    python -m framework.cleansight_eval.cli.sweep --group group1_large \
        --preset large_s large_s_960 large_m_960 --smoke
"""

from __future__ import annotations

import argparse
import sys

from ..detection.sweep import (
    PRESETS,
    REPORTS_DIR,
    print_summary,
    run_experiment,
    run_grid,
    save_report,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CleanSight YOLO 优化实验编排")
    p.add_argument("--group", required=True, choices=("group1_large", "group2_small"))
    p.add_argument("--preset", nargs="*", help="预设名(可多个), 如 large_baseline large_s")
    p.add_argument("--grid", nargs="*", help="grid search 维度: models, resolutions, augments; 或 all")
    p.add_argument("--dry-run", action="store_true", help="仅打印计划，不执行训练")
    p.add_argument("--smoke", action="store_true",
                   help="快速探针：截断 epochs/patience + fraction 子采样，结果仅用于方向对比")
    p.add_argument("--device", default="auto", help="auto / cpu / cuda:0 等")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.smoke:
        print("⚠️ SMOKE 探针模式：截断训练 + 数据子采样，结果仅用于预设方向对比，不代表正式指标。")

    experiments = []
    if args.preset:
        for preset_name in args.preset:
            if preset_name in PRESETS:
                experiments.append((preset_name, dict(PRESETS[preset_name])))
            else:
                print(f"[WARN] 未知预设 '{preset_name}'，跳过。可用: {sorted(PRESETS)}")

    all_results = []
    for preset_name, cfg in experiments:
        all_results.append(run_experiment(args.group, preset_name, cfg,
                                          dry_run=args.dry_run, smoke=args.smoke,
                                          device=args.device))

    if args.grid:
        grid_dims = args.grid if args.grid != ["all"] else ["models", "resolutions", "augments"]
        all_results.extend(run_grid(args.group, grid_dims,
                                    dry_run=args.dry_run, smoke=args.smoke,
                                    device=args.device))

    if not all_results:
        print("[ERROR] 请指定 --preset 或 --grid", file=sys.stderr)
        print(f"可用 preset: {sorted(PRESETS)}", file=sys.stderr)
        return 1

    print_summary(all_results)
    if not args.dry_run:
        save_report(all_results, args.group)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
