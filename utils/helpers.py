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
        _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return _PROJECT_ROOT


def resolve_path(path: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_get_project_root(), path))