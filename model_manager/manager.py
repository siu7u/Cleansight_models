#!/usr/bin/env python3
"""统一管理 YOLO 与时序模型训练、评测和 benchmark 的轻量 CLI。

本脚本不替代各模型已有训练逻辑，只读取 `models.yaml` 中的模型清单，
再通过统一接口调用现有脚本。这样可以在不重构 YOLO pipeline 和
temporal-* 仓库的前提下，集中管理不同模型的输入、输出和命令。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CATALOG = Path(__file__).with_name("models.yaml")
ENV_FILES = [ROOT / ".env"]


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


def parse_env_file(path: Path) -> dict[str, str]:
    """读取简单 .env 文件，返回可传给子进程的环境变量。"""

    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        if text.startswith("export "):
            text = text[len("export "):].strip()
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


def build_subprocess_env() -> dict[str, str]:
    """合并当前环境和仓库 .env；已有系统环境变量优先。"""

    env = os.environ.copy()
    loaded_keys: set[str] = set()
    for path in ENV_FILES:
        for key, value in parse_env_file(path).items():
            if key not in env:
                env[key] = value
                loaded_keys.add(key)
    if loaded_keys:
        names = [key if key != "LS_TOKEN" else "LS_TOKEN=<hidden>" for key in sorted(loaded_keys)]
        print(f"[env] loaded from .env: {', '.join(names)}")
    return env


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


def commands_for(spec: ModelSpec, action: str) -> list[tuple[list[str], Path]]:
    """返回某个动作的一组命令；单步动作也规范为单元素列表。"""

    commands = spec.raw.get("commands", {})
    if action not in commands:
        raise KeyError(f"{spec.id} 不支持动作: {action}")

    raw = commands[action]
    if raw and all(isinstance(item, list) for item in raw):
        return [(build_python_command([str(part) for part in step]), spec.workdir) for step in raw]
    return [command_for(spec, action)]


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
    """执行或预览某个模型的单步或多步动作。"""

    print(f"[{spec.id}] {action}")
    env = None if dry_run else build_subprocess_env()
    for index, (cmd, cwd) in enumerate(commands_for(spec, action), start=1):
        prefix = f"  step {index}: " if action == "pipeline" else "  "
        print(f"{prefix}cwd: {cwd.relative_to(ROOT)}")
        print(f"{prefix}cmd: {' '.join(cmd)}")
        if dry_run:
            continue
        code = subprocess.run(cmd, cwd=cwd, env=env).returncode
        if code != 0:
            print(f"  failed at step {index} with exit code {code}")
            return code
    return 0


def ensure_action_supported(specs: list[ModelSpec], action: str) -> None:
    """提前检查所选模型是否都支持目标动作。"""

    unsupported = [spec.id for spec in specs if action not in spec.raw.get("commands", {})]
    if unsupported:
        raise SystemExit(f"{action} 不支持这些模型: {', '.join(unsupported)}")


def run_benchmark(
    name: str,
    dry_run: bool,
    version: str | None = None,
    model: str | None = None,
    split: str | None = None,
    weights: str | None = None,
    summaries: list[str] | None = None,
    card: str | None = None,
    latency_ms: float | None = None,
    causality: str | None = None,
    num_params: int | None = None,
) -> int:
    """执行或预览清单中的集中 benchmark。"""

    catalog = load_catalog()
    benchmarks = catalog.get("benchmarks", {})
    if name not in benchmarks:
        raise KeyError(f"未知 benchmark: {name}")
    cmd = build_python_command([str(part) for part in benchmarks[name]["command"]])
    if name == "single_model_yolo" and (model or split or weights):
        cmd = [part for part in cmd if part != "--skip-run"]
    if version:
        cmd.extend(["--version", version])
    if model:
        cmd.extend(["--model", model])
    if split:
        cmd.extend(["--split", split])
    if weights:
        cmd.extend(["--weights", weights])
    for summary in summaries or []:
        cmd.extend(["--summary", summary])
    if card:
        cmd.extend(["--card", card])
    if latency_ms is not None:
        cmd.extend(["--latency-ms", str(latency_ms)])
    if causality:
        cmd.extend(["--causality", causality])
    if num_params is not None:
        cmd.extend(["--num-params", str(num_params)])
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

    for name in ("train", "eval", "pipeline"):
        p = sub.add_parser(name, help=f"运行或预览 {name} 动作")
        p.add_argument("--model", help="指定模型 id，例如 temporal.gru")
        p.add_argument("--family", choices=["yolo", "temporal"], help="按模型族筛选")
        p.add_argument("--run", action="store_true", help="真正执行；不传则只打印命令")

    bench = sub.add_parser("benchmark", help="运行或预览集中 benchmark")
    bench.add_argument("name", choices=["single_model_yolo", "single_model_temporal", "temporal_feed_mode", "release_gate"])
    bench.add_argument("--run", action="store_true", help="真正执行；不传则只打印命令")
    bench.add_argument("--version", help="为 benchmark summary 指定版本名，例如 yolo-large-v2")
    bench.add_argument("--model", help="传给 benchmark runner 的模型 id 或模型名")
    bench.add_argument("--split", help="传给 benchmark runner 的数据 split,例如 val/test")
    bench.add_argument("--weights", help="传给 benchmark runner 的权重路径")
    bench.add_argument("--summary", action="append", help="传给 release_gate 的 benchmark summary JSON;可传多次")
    bench.add_argument("--card", help="传给 release_gate 的 CARD.md 路径")
    bench.add_argument("--latency-ms", type=float, help="传给 release_gate 的部署机实测延迟")
    bench.add_argument("--causality", help="传给 release_gate 的因果性/感受域声明")
    bench.add_argument("--num-params", type=int, help="传给 release_gate 的模型参数量")

    args = parser.parse_args()
    models = load_models()

    if args.command == "list":
        list_models(models)
        return 0
    if args.command == "status":
        return status_models(models)
    if args.command in {"train", "eval", "pipeline"}:
        selected = select_models(models, args.model, args.family)
        if not selected:
            raise SystemExit("没有匹配的模型")
        ensure_action_supported(selected, args.command)
        codes = [run_action(spec, args.command, dry_run=not args.run) for spec in selected]
        return 0 if all(code == 0 for code in codes) else 2
    if args.command == "benchmark":
        return run_benchmark(
            args.name,
            dry_run=not args.run,
            version=args.version,
            model=args.model,
            split=args.split,
            weights=args.weights,
            summaries=args.summary,
            card=args.card,
            latency_ms=args.latency_ms,
            causality=args.causality,
            num_params=args.num_params,
        )

    raise SystemExit(f"未知命令: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
