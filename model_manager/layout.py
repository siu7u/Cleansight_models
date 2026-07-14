"""检查模型代码与权重产物是否按目录职责隔离。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CODE_DIRECTORY_NAMES = frozenset({"model", "scripts", "tools"})
WEIGHT_DIRECTORY_NAMES = frozenset({"checkpoints", "weights", "runs", "registry"})
WEIGHT_SUFFIXES = frozenset({".pt", ".pth", ".onnx", ".engine", ".ckpt", ".safetensors"})


@dataclass(frozen=True)
class LayoutViolation:
    """描述一项目录隔离违规及其仓库相对路径。"""

    path: Path
    rule: str
    message: str

    def __str__(self) -> str:
        return f"{self.path.as_posix()}: {self.message}"


class LayoutError(ValueError):
    """目录隔离校验失败，并携带全部违规项。"""

    def __init__(self, violations: list[LayoutViolation]):
        self.violations = tuple(violations)
        details = "\n".join(f"- {item}" for item in self.violations)
        super().__init__(f"模型目录隔离校验失败：\n{details}")


def _named_directories(root: Path, names: frozenset[str]) -> list[Path]:
    """查找根目录本身及其后代中名称匹配的目录。"""

    found = {path for path in root.rglob("*") if path.is_dir() and path.name in names}
    if root.is_dir() and root.name in names:
        found.add(root)
    return sorted(found)


def _relative(path: Path, root: Path) -> Path:
    """返回便于报告的仓库相对路径。"""

    try:
        return path.relative_to(root)
    except ValueError:
        return path


def scan_layout(root: str | Path) -> list[LayoutViolation]:
    """扫描代码与权重目录，返回全部隔离违规。

    `model/scripts/tools` 中禁止出现模型权重后缀；
    `checkpoints/weights/runs/registry` 中禁止出现 Python 源码；正式
    `registry` 只保存元数据引用，因此也禁止直接存放权重文件。
    """

    base = Path(root)
    if not base.is_dir():
        raise NotADirectoryError(base)

    violations: list[LayoutViolation] = []
    seen: set[tuple[Path, str]] = set()

    for directory in _named_directories(base, CODE_DIRECTORY_NAMES):
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES:
                relative = _relative(path, base)
                key = (relative, "weight_in_code")
                if key not in seen:
                    seen.add(key)
                    violations.append(
                        LayoutViolation(relative, key[1], "代码目录中禁止存放模型权重")
                    )

    for directory in _named_directories(base, WEIGHT_DIRECTORY_NAMES):
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            relative = _relative(path, base)
            if path.suffix.lower() == ".py":
                key = (relative, "python_in_weight")
                if key not in seen:
                    seen.add(key)
                    violations.append(
                        LayoutViolation(relative, key[1], "权重目录中禁止存放 Python 源码")
                    )
            if directory.name == "registry" and path.suffix.lower() in WEIGHT_SUFFIXES:
                key = (relative, "weight_in_registry")
                if key not in seen:
                    seen.add(key)
                    violations.append(
                        LayoutViolation(relative, key[1], "正式 registry 只保留元数据引用，禁止直接存放权重")
                    )

    return sorted(violations, key=lambda item: (item.path.as_posix(), item.rule))


def validate_layout(root: str | Path) -> None:
    """校验目录隔离；存在任一违规时抛出 ``LayoutError``。"""

    violations = scan_layout(root)
    if violations:
        raise LayoutError(violations)
