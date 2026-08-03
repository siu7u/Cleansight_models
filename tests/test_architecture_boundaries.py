"""framework 与 benchmark 的职责和依赖边界。"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_PACKAGE = ROOT / "framework" / "cleansight_eval"
ACTIVE_MODEL_EXECUTION_ROOTS = (
    ROOT / "benchmark",
    ROOT / "tools",
)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_framework_does_not_own_evaluation_entry_or_outputs():
    forbidden = [
        FRAMEWORK_PACKAGE / "cli" / "eval.py",
        FRAMEWORK_PACKAGE / "cli" / "matrix.py",
        FRAMEWORK_PACKAGE / "core" / "report.py",
        FRAMEWORK_PACKAGE / "core" / "matrix.py",
        FRAMEWORK_PACKAGE / "core" / "envelope.py",
        FRAMEWORK_PACKAGE / "core" / "artifacts.py",
    ]
    assert not [path for path in forbidden if path.exists()]


def test_framework_does_not_import_benchmark_result_or_evaluators():
    """训练可复用纯 metrics/testset，但不能反向拥有 evaluator 或结果产物。"""

    forbidden_prefixes = (
        "benchmark.evaluators",
        "benchmark.core.result",
        "benchmark.core.report",
        "benchmark.core.matrix",
        "benchmark.core.artifact_io",
        "benchmark.core.delivery",
    )
    violations = []
    for path in FRAMEWORK_PACKAGE.rglob("*.py"):
        for name in _imports(path):
            if name.startswith(forbidden_prefixes):
                violations.append(f"{path.relative_to(ROOT)} -> {name}")
    assert violations == []


def test_benchmark_owns_public_evaluation_entry():
    assert (ROOT / "benchmark" / "cli" / "eval.py").is_file()
    assert (ROOT / "benchmark" / "core" / "result.py").is_file()
    assert (ROOT / "benchmark" / "core" / "report.py").is_file()
    assert (ROOT / "benchmark" / "core" / "matrix.py").is_file()


def test_active_non_framework_code_does_not_execute_models():
    """benchmark/tools 不得重新加载 checkpoint 或直接调用模型库。"""

    forbidden_imports = ("torch", "ultralytics")
    forbidden_calls = {
        "torch.load",
        "torch.jit.load",
        "YOLO",
        "load_state_dict",
    }
    violations = []
    for package in ACTIVE_MODEL_EXECUTION_ROOTS:
        for path in package.rglob("*.py"):
            relative = path.relative_to(ROOT)
            for imported in _imports(path):
                if imported in forbidden_imports or imported.startswith(
                    tuple(f"{name}." for name in forbidden_imports)
                ):
                    violations.append(f"{relative} imports {imported}")
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _call_name(node.func)
                if name in forbidden_calls or name.endswith(".load_state_dict"):
                    violations.append(f"{relative} calls {name}")
    assert violations == []


def _call_name(node: ast.expr) -> str:
    """把简单调用表达式转换为点路径，供边界检查使用。"""

    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def test_active_code_does_not_depend_on_legacy_layout():
    """legacy 只能被人工审计，不得成为活跃 Python 包的运行依赖。"""

    violations = []
    for package in (ROOT / "framework", ROOT / "benchmark", ROOT / "tools"):
        for path in package.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "legacy/" in text or "legacy." in text:
                violations.append(str(path.relative_to(ROOT)))
    assert violations == []
