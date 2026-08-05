"""脚本共用:定位仓库根、加载 config.yaml、白名单判断。"""
from pathlib import Path

import yaml

PKG_ROOT = Path(__file__).resolve().parent.parent   # yolo_pipeline/
ROOT = PKG_ROOT                                      # 自包含:raw/datasets/runs 全在 yolo_pipeline/ 内
CONFIG_PATH = PKG_ROOT / "config.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def is_whitelisted(name: str, only_videos) -> bool:
    """only_videos 为空 = 全部通过;否则文件名前缀匹配任一。"""
    if not only_videos:
        return True
    return any(name.startswith(p) for p in only_videos)
