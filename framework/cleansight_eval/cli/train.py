"""训练入口：python -m framework.cleansight_eval.cli.train。

两种用法：
  a) 配置驱动（默认）：``--config <yaml>``，适合进阶/精确控制；
  b) 模型别名（组员友好）：``--model <名> [--group <组>]``，由
     ``core/model_aliases.py`` 解析到实验配置，自动附加 YOLO 权重/分组覆盖。

本入口只做**分派**：加载配置、选设备、应用 CLI 覆盖项，然后交给
``get_pipeline(cfg["pipeline"]).train(...)``。训练主体（时序的 forward/loss 循环、检测的
ultralytics 封装）各由所属流水线实现。

临时调参用通用的 ``-S/--set 点路径=值``（可多次），不改配置文件；核心 CLI 不预设任何纵的
调参名，各纵按自己的超参词汇寻址，如 ``-S train.epochs=5``（两纵通用）、``-S train.batch=8``
（检测/ultralytics）、``-S train.window=32``（时序滑窗）。

--model 模式训练前自动做数据就绪检查（core/dataset_download.check_required_datasets），
缺失时打印下载命令并退出；``--force`` 跳过检查。
"""

from __future__ import annotations

import argparse
from typing import Any

from ..core.config import apply_overrides, load_config


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="cleansight_eval 训练入口")
    p.add_argument("--config", default=None, help="实验配置 YAML（与 --model 二选一）")
    p.add_argument("--model", default=None, help="模型别名（--list-models 查看；与 --config 二选一）")
    p.add_argument("--group", default=None, help="YOLO 分组（group1_large / group2_small），仅 --model 模式")
    p.add_argument("--list-models", action="store_true", help="列出所有可训练模型别名")
    p.add_argument("--force", action="store_true", help="--model 模式跳过数据就绪检查")
    p.add_argument("--runs-dir", default="runs", help="运行输出根目录")
    p.add_argument("--run-id", default=None, help="显式指定 run 目录名，例如 yolo11s-ft768-lowlr-lowaug")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", help="从完整训练 checkpoint（通常是 checkpoints/last.pt）恢复")
    p.add_argument(
        "-S",
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="overrides",
        help="通用配置覆盖，点路径寻址，可多次；如 -S train.epochs=5 -S train.batch=8",
    )
    return p.parse_args(argv)


def _coerce(s: str) -> Any:
    """把 --set 的字符串值转成 bool/None/int/float，无法转则原样保留字符串。"""
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def _parse_overrides(items: list[str]) -> list[tuple[str, Any]]:
    out = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set 需为 KEY=VALUE 形式（点路径寻址）: {item!r}")
        key, value = item.split("=", 1)
        out.append((key.strip(), _coerce(value.strip())))
    return out


def _resolve_model_config(args) -> str:
    """--model 模式：解析别名 → 配置路径，附加权重/分组覆盖，做数据就绪检查。"""

    from ..core.dataset_download import check_required_datasets
    from ..core.model_aliases import model_config_path, resolve_model

    info = resolve_model(args.model, args.group)
    config_path = model_config_path(info)

    if not args.force:
        missing = check_required_datasets([info["dataset"]])
        if missing:
            print("[train] 训练所需数据未就绪，请先下载：")
            for key in missing:
                print(f"  python -m framework.cleansight_eval.cli.dataset --preset {key}")
            raise SystemExit(f"缺少数据集: {', '.join(missing)}（或用 --force 跳过检查）")

    # 模型别名附加覆盖项：YOLO 权重与分组（用户 -S 优先，后附加的覆盖在 apply 时被 -S 覆盖，
    # 因此这里先放别名覆盖、用户在 --set 里的同名项会覆盖它）
    return str(config_path)


def main(argv=None) -> str:
    args = parse_args(argv)

    if args.list_models:
        from ..core.model_aliases import list_models

        print(list_models())
        return ""

    if args.config and args.model:
        raise SystemExit("--config 与 --model 只能二选一")

    overrides = _parse_overrides(args.overrides)
    # resume 语义：ultralytics 的 resume 只接受 True（从 self.ckpt_path 续训），
    # 因此 resume 时必须把 model.weights 指向 last.pt，并传 train.resume=True。
    resume_ckpt = None
    if args.resume:
        resume_ckpt = str(args.resume)
        overrides.append(("train.resume", True))

    if args.model:
        config_path = _resolve_model_config(args)
        from ..core.model_aliases import resolve_model

        info = resolve_model(args.model, args.group)
        # 别名覆盖（YOLO 权重/分组）先应用，用户 -S 后应用并优先
        alias_overrides = list(info.get("overrides", {}).items())
        if info.get("group"):
            alias_overrides.append(("data.name", info["group"]))
        print(f"[train] 模型: {args.model}  配置: {config_path}")
        if resume_ckpt:
            alias_overrides.append(("model.weights", resume_ckpt))
            print(f"[train] resume: 从 {resume_ckpt} 续训")
        if alias_overrides or overrides:
            print(f"[train] 覆盖: 别名{alias_overrides} + 用户{overrides}")
        cfg = apply_overrides(load_config(config_path), alias_overrides + overrides)
    else:
        if not args.config:
            raise SystemExit("请指定 --config <yaml> 或 --model <别名>（--list-models 查看）")
        cfg = apply_overrides(load_config(args.config), overrides)
        if resume_ckpt:
            # --config 模式 resume：把 model.weights 指向 last.pt 再 train.resume=True
            cfg = apply_overrides(cfg, [("model.weights", resume_ckpt)])

    from ..core.environment import pick_device
    from ..core.registry import get_pipeline

    device = pick_device()
    pipeline = get_pipeline(cfg["pipeline"])
    pipeline.validate_config(cfg)  # 流水线专属校验（core 不再代劳）
    return pipeline.train(cfg, runs_dir=args.runs_dir, seed=args.seed, device=device, run_id=args.run_id)


if __name__ == "__main__":
    main()
