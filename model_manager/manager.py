#!/usr/bin/env python3
"""统一管理 YOLO 与时序模型训练、评测和 benchmark 的轻量 CLI。

本脚本不替代各模型已有训练逻辑，只读取 `models.yaml` 中的模型清单，
再通过统一接口调用现有脚本。这样可以在不重构 YOLO pipeline 和
temporal-* 仓库的前提下，集中管理不同模型的输入、输出和命令。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CATALOG = Path(__file__).with_name("models.yaml")


@dataclass(frozen=True)
class ModelSpec:
    """模型清单中的单个模型条目。

    `raw` 保留 YAML 原始字段，供后续扩展更多模型族或输出检查规则。
    """

    id: str
    family: str
    adapter: str
    workdir: Path
    target: str
    raw: dict[str, Any]

    @property
    def checkpoint(self) -> Path | None:
        """返回模型登记的 checkpoint 路径；未声明时返回 None。"""

        value = self.raw.get("output", {}).get("checkpoint")
        return self.workdir / value if value else None

    @property
    def report(self) -> Path | None:
        """返回模型登记的评估报告路径；未声明时返回 None。"""

        value = self.raw.get("output", {}).get("report")
        return self.workdir / value if value else None


def load_catalog(path: Path = CATALOG) -> dict[str, Any]:
    """读取模型管理清单，并返回 YAML 对象。"""

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_models(path: Path = CATALOG) -> dict[str, ModelSpec]:
    """读取所有模型条目，并按模型 id 建立索引。"""

    catalog = load_catalog(path)
    models: dict[str, ModelSpec] = {}
    for item in catalog.get("models", []):
        workdir = ROOT / item["workdir"]
        spec = ModelSpec(
            id=item["id"],
            family=item["family"],
            adapter=item["adapter"],
            workdir=workdir,
            target=item["target"],
            raw=item,
        )
        models[spec.id] = spec
    return models


def build_python_command(args: list[str]) -> list[str]:
    """把清单中的脚本参数转换为当前 Python 解释器可执行命令。"""

    return [sys.executable, *args]


def command_for(spec: ModelSpec, action: str) -> tuple[list[str], Path]:
    """返回某个模型某个动作的命令和工作目录。

    `train` 对时序模型在各自仓库下运行；`eval` 如果调用跨仓库工具，则在
    模型集根目录下运行，避免相对路径找不到 `tools/`。
    """

    commands = spec.raw.get("commands", {})
    if action not in commands:
        raise KeyError(f"{spec.id} 不支持动作: {action}")

    cmd = build_python_command([str(part) for part in commands[action]])
    cwd = spec.workdir
    if spec.family == "temporal" and action == "eval":
        cwd = ROOT
    return cmd, cwd


def list_models(models: dict[str, ModelSpec]) -> None:
    """打印所有已登记模型的关键输入输出信息。"""

    for spec in models.values():
        input_cfg = spec.raw.get("input", {})
        output_cfg = spec.raw.get("output", {})
        print(f"{spec.id}")
        print(f"  family: {spec.family}")
        print(f"  adapter: {spec.adapter}")
        print(f"  workdir: {spec.workdir.relative_to(ROOT)}")
        if "input_dim" in input_cfg:
            print(f"  input_dim: {input_cfg['input_dim']}")
        if "window" in input_cfg:
            print(f"  window: {input_cfg['window']}")
        if "feature_mapping" in input_cfg:
            print(f"  feature_mapping: {input_cfg['feature_mapping']}")
        if output_cfg.get("checkpoint"):
            print(f"  checkpoint: {output_cfg['checkpoint']}")
        print()


def status_models(models: dict[str, ModelSpec]) -> int:
    """检查模型清单声明的关键产物是否存在。"""

    missing = 0
    for spec in models.values():
        print(f"{spec.id}")
        for label, path in [("checkpoint", spec.checkpoint), ("report", spec.report)]:
            if path is None:
                continue
            ok = path.exists()
            print(f"  {label}: {'OK' if ok else 'MISSING'} {path.relative_to(ROOT)}")
            if not ok:
                missing += 1
    return 0 if missing == 0 else 2


def run_action(spec: ModelSpec, action: str, dry_run: bool) -> int:
    """执行或预览某个模型的训练/评测动作。"""

    cmd, cwd = command_for(spec, action)
    print(f"[{spec.id}] {action}")
    print(f"  cwd: {cwd.relative_to(ROOT)}")
    print(f"  cmd: {' '.join(cmd)}")
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=cwd).returncode


def run_benchmark(name: str, dry_run: bool) -> int:
    """执行或预览清单中的集中 benchmark。"""

    catalog = load_catalog()
    benchmarks = catalog.get("benchmarks", {})
    if name not in benchmarks:
        raise KeyError(f"未知 benchmark: {name}")
    cmd = build_python_command([str(part) for part in benchmarks[name]["command"]])
    print(f"[benchmark] {name}")
    print("  cwd: .")
    print(f"  cmd: {' '.join(cmd)}")
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=ROOT).returncode


def select_models(models: dict[str, ModelSpec], model_id: str | None, family: str | None) -> list[ModelSpec]:
    """按模型 id 或 family 选择要操作的模型集合。"""

    selected = list(models.values())
    if model_id:
        if model_id not in models:
            raise KeyError(f"未知模型 id: {model_id}")
        selected = [models[model_id]]
    if family:
        selected = [item for item in selected if item.family == family]
    return selected


def main() -> int:
    """解析命令行参数，并执行统一模型管理动作。"""

    parser = argparse.ArgumentParser(description="CleanSight 模型集统一管理入口")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="列出模型清单")
    sub.add_parser("status", help="检查 checkpoint/report 等登记产物")

    for name in ("train", "eval"):
        p = sub.add_parser(name, help=f"运行或预览 {name} 动作")
        p.add_argument("--model", help="指定模型 id，例如 temporal.gru")
        p.add_argument("--family", choices=["yolo", "temporal"], help="按模型族筛选")
        p.add_argument("--run", action="store_true", help="真正执行；不传则只打印命令")

    bench = sub.add_parser("benchmark", help="运行或预览集中 benchmark")
    bench.add_argument("name", choices=["single_model_yolo", "single_model_temporal", "temporal_feed_mode"])
    bench.add_argument("--run", action="store_true", help="真正执行；不传则只打印命令")

    args = parser.parse_args()
    models = load_models()

    if args.command == "list":
        list_models(models)
        return 0
    if args.command == "status":
        return status_models(models)
    if args.command in {"train", "eval"}:
        selected = select_models(models, args.model, args.family)
        if not selected:
            raise SystemExit("没有匹配的模型")
        codes = [run_action(spec, args.command, dry_run=not args.run) for spec in selected]
        return 0 if all(code == 0 for code in codes) else 2
    if args.command == "benchmark":
        return run_benchmark(args.name, dry_run=not args.run)

    raise SystemExit(f"未知命令: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
