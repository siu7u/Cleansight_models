"""评估入口：python -m cleansight_eval.cli.eval --config <yaml> --ckpt <path>。

只做**分派**：在本实验的喂入模式（``cfg["feeding"]``，与训练同一个）下调用
``get_task(cfg["task"]).evaluate(...)``，得到一份三态信封并落盘。重建模型、指标口径、
喂入语义等由所属任务实现（§4.2）。训练怎么喂，评估就怎么喂，不做多模式扫描。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..core.config import load_config
from ..core.environment import now_stamp, pick_device
from ..tasks import get_task


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="cleansight_eval 评估入口")
    p.add_argument("--config", required=True, help="实验配置 YAML")
    p.add_argument("--ckpt", required=True, help="checkpoint 路径（需存在同名 .meta.json）")
    p.add_argument("--out-dir", default=None, help="信封输出目录，默认写到 ckpt 所在 run 的 evals/")
    return p.parse_args(argv)


def _resolve_out_dir(ckpt: str, override: str | None) -> Path:
    if override:
        return Path(override)
    # ckpt 通常在 <run>/checkpoints/... 下，评估写到同 run 的 evals/。
    # 时序权重直接在 checkpoints/；检测权重在 checkpoints/<name>/weights/，故向上找。
    ckpt_path = Path(ckpt)
    for anc in ckpt_path.parents:
        if anc.name == "checkpoints":
            return anc.parent / "evals"
    return ckpt_path.parent / "evals"


def main(argv=None) -> list[str]:
    args = parse_args(argv)
    cfg = load_config(args.config)
    device = pick_device()
    task = get_task(cfg["task"])

    out_dir = _resolve_out_dir(args.ckpt, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feeding_name = cfg["feeding"]
    envelope = task.evaluate(cfg, args.ckpt, feeding_name, device)
    path = out_dir / f"{envelope.family}-{feeding_name}-{now_stamp()}.envelope.json"
    envelope.write(path)
    print(f"[eval] {feeding_name}: {path}")
    return [str(path)]


if __name__ == "__main__":
    main()
