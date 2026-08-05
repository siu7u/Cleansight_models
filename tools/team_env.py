#!/usr/bin/env python3
"""
组员环境工具：检查模型训练所需依赖，可选一键安装。

用法（仓库根执行）:
    python tools/team_env.py          # 检查 python/torch/ultralytics 等依赖
    python tools/team_env.py --setup  # 在当前 python 环境安装 framework/requirements.txt
    python tools/team_env.py --setup-venv   # 创建 .venv 并安装依赖（推荐给没有环境的新机器）

说明：
- 无 torch/ultralytics 时也能运行本脚本（检查项逐个尝试导入）。
- YOLO 训练需要 ultralytics（体积大，含 torch）；纯时序训练只需 torch + numpy + PyYAML。
- --setup 只安装依赖，不创建虚拟环境；--setup-venv 会新建仓库内 .venv。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "framework" / "requirements.txt"

# (包名, 导入名, 训练是否需要, 说明)
CHECKS = [
    ("python", None, True, f"Python ≥ 3.10（当前 {sys.version.split()[0]}）"),
    ("torch", "torch", True, "深度学习框架（YOLO 与时序都需要）"),
    ("PyYAML", "yaml", True, "配置解析"),
    ("numpy", "numpy", True, "数值计算"),
    ("ultralytics", "ultralytics", False, "仅 YOLO 训练需要（含 opencv，体积大）"),
    ("pytorch-tcn", "torch_tcn", False, "仅 MS-TCN 时序需要"),
    ("matplotlib", "matplotlib", False, "训练曲线可视化"),
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="组员环境检查与安装")
    p.add_argument("--setup", action="store_true", help="在当前 python 环境安装依赖")
    p.add_argument("--setup-venv", action="store_true", help="创建仓库内 .venv 并安装依赖")
    return p.parse_args(argv)


def _importable(name: str | None) -> bool:
    if name is None:
        return True
    try:
        __import__(name)
        return True
    except Exception:
        return False


def check_env() -> list[tuple[str, bool, str, bool]]:
    """逐个检查依赖，返回 (包名, 是否就绪, 说明, 是否必需)。"""

    results = []
    for pkg, import_name, required, desc in CHECKS:
        if pkg == "python":
            major, minor = sys.version_info[:2]
            ok = (major, minor) >= (3, 10)
        else:
            ok = _importable(import_name)
        results.append((pkg, ok, desc, required))
    return results


def print_report(results) -> None:
    print("环境检查：")
    print()
    for pkg, ok, desc, required in results:
        mark = "✅" if ok else ("⚠️" if not required else "❌")
        note = "" if ok else ("（可选）" if not required else "（必需）")
        print(f"  [{mark}] {pkg:<12} {desc} {note}")
    print()
    missing_required = [pkg for pkg, ok, _, required in results if not ok and required]
    if missing_required:
        print(f"缺少必需依赖: {', '.join(missing_required)}")
        print("安装: python tools/team_env.py --setup")
    else:
        print("必需依赖已就绪 ✅")


def run_pip(*args: str) -> int:
    return subprocess.call([sys.executable, "-m", "pip", "install", *args])


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.setup_venv:
        venv_dir = ROOT / ".venv"
        if not (venv_dir / "bin" / "python").is_file():
            print(f"创建虚拟环境: {venv_dir}")
            if subprocess.call([sys.executable, "-m", "venv", str(venv_dir)]) != 0:
                print("venv 创建失败（可能需要 python3-venv 包）。", file=sys.stderr)
                return 1
        python = venv_dir / "bin" / "python"
        print(f"安装依赖到 {venv_dir} ...")
        code = subprocess.call([str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
        if code != 0:
            print("依赖安装失败。", file=sys.stderr)
            return code
        print(f"完成。使用: source {venv_dir}/bin/activate")
        return 0

    if args.setup:
        if not REQUIREMENTS.is_file():
            print(f"找不到 {REQUIREMENTS}", file=sys.stderr)
            return 1
        print(f"安装依赖: {REQUIREMENTS}")
        return run_pip("-r", str(REQUIREMENTS))

    results = check_env()
    print_report(results)
    if any(not ok and required for _, ok, _, required in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
