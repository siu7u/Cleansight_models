"""framework 与 benchmark 的职责和依赖边界。"""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_PACKAGE = ROOT / "framework" / "cleansight_eval"


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
