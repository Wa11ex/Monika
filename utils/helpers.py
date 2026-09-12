'''
通用工具函数，供所有模块共享
'''

import os
from pathlib import Path

_PROJECT_ROOT: str | None = None


def _get_project_root() -> str:
    '''用于惰性计算'''
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 便携版: 如果代码在 src/ 子目录，项目根再往上一级
        if os.path.basename(root) == "src":
            root = os.path.dirname(root)
        _PROJECT_ROOT = root
    return _PROJECT_ROOT


def resolve_path(path: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_get_project_root(), path))