'''
模型注册表

管理 assets/model/registry.json，追踪所有已导入的 GGUF 模型
启动时自动扫描 assets/model/*.gguf 将未注册的文件补录入表
'''

import os
import json
import glob
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REGISTRY_PATH = os.path.join(_PROJECT_ROOT, "assets", "model", "registry.json")


# -- 内部 I/O --------------------------------------------------

def _load() -> dict:
    if os.path.exists(_REGISTRY_PATH):
        try:
            with open(_REGISTRY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"models": []}


def _save(data: dict) -> None:
    os.makedirs(os.path.dirname(_REGISTRY_PATH), exist_ok=True)
    with open(_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# -- 启动迁移：扫描现有 .gguf 文件 ----------------------------

def _scan_and_migrate() -> None:
    '''把 assets/model/*.gguf 中尚未注册的文件自动补录'''
    data = _load()
    existing_paths = {os.path.normpath(m["path"]) for m in data["models"]}
    model_dir = os.path.join(_PROJECT_ROOT, "assets", "model")
    changed = False
    for gguf_path in glob.glob(os.path.join(model_dir, "*.gguf")):
        norm = os.path.normpath(gguf_path)
        if norm not in existing_paths:
            name = os.path.splitext(os.path.basename(norm))[0]
            try:
                size_gb = round(os.path.getsize(norm) / (1024 ** 3), 2)
            except OSError:
                size_gb = 0.0
            data["models"].append({
                "name": name,
                "path": norm,
                "format": "gguf",
                "arch": "unknown",
                "size_gb": size_gb,
                "added_at": datetime.now().isoformat(timespec="seconds"),
            })
            changed = True
    if changed:
        _save(data)


# -- 公共调用区 --------------------------------------------------

def list_models() -> list[dict]:
    '''返回所有已注册模型的列表
    自动扫描目录，补录'''
    _scan_and_migrate()
    return _load()["models"]


def add_model(name: str, path: str, fmt: str = "gguf", arch: str = "unknown") -> None:
    '''注册或更新一个模型条目（同名则覆盖）'''
    data = _load()
    norm_path = os.path.normpath(os.path.abspath(path))
    # 同名覆盖
    data["models"] = [m for m in data["models"] if m["name"] != name]
    try:
        size_gb = round(os.path.getsize(norm_path) / (1024 ** 3), 2)
    except OSError:
        size_gb = 0.0
    data["models"].append({
        "name": name,
        "path": norm_path,
        "format": fmt,
        "arch": arch,
        "size_gb": size_gb,
        "added_at": datetime.now().isoformat(timespec="seconds"),
    })
    _save(data)


def remove_model(name: str) -> None:
    '''从注册表中删除指定名称的模型'''
    data = _load()
    data["models"] = [m for m in data["models"] if m["name"] != name]
    _save(data)


def get_model_path(name: str) -> str | None:
    '''通过名称查找模型绝对路径，否则返回 None'''
    for m in list_models():
        if m["name"] == name:
            return m["path"]
    return None
