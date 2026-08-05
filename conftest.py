"""让仓库根目录直接执行 ``pytest`` 时可以导入活跃源码包。

仓库同时保留两种现有导入形式：根级模块使用 ``benchmark`` / ``tools``，framework
测试使用 ``cleansight_eval``。因此测试收集前需要把仓库根目录和 ``framework/``
都加入模块搜索路径；这只影响 pytest，不改变生产运行时依赖。
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
for source_root in (ROOT, ROOT / "framework"):
    path = str(source_root)
    if path not in sys.path:
        sys.path.insert(0, path)
