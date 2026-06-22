'''
从 config.yaml 读取全局配置，供各组件使用
'''

import os
import threading
import yaml
from typing import Any, Dict


_config: Dict = None
_config_path: str = None
_lock = threading.Lock()


def load_config(path: str = None) -> Dict:

    global _config, _config_path

    if path is None:
        # 从项目根目录查找
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "config.yaml")

    if not os.path.exists(path):
        print(f">>> [Config] 未找到配置文件: {path}，使用默认配置")
        _config = {}
        return _config

    with open(path, 'r', encoding='utf-8') as f:
        _config = yaml.safe_load(f) or {}

    _config_path = path
    print(f">>> [Config] 配置已加载: {path}")
    return _config


def get_config(section: str = None, default: Any = None) -> Any:
    '''支持点号分隔的嵌套键'''
    global _config
    if _config is None:
        load_config()

    if section is None:
        return _config

    keys = section.split(".")
    value = _config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def set_config(section: str, value: Any) -> None:
    '''
    设置配置项并持久化到 config.yaml支持点号分隔的嵌套键
    线程安全

    Examples:
        set_config("ui.live2d_scale", 1.5)
        set_config("ui.live2d_x", 320)

    Args:
        section: 配置键路径
        value:   要写入的值
    '''
    global _config, _config_path
    if _config is None:
        load_config()

    keys = section.split(".")
    with _lock:
        node = _config
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = value

        if _config_path:
            with open(_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
