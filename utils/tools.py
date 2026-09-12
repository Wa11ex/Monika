"""
通用工具函数 — 时间格式化 + 数据加解密。
"""

import os
import sys
import json
import base64
import hashlib
import secrets
from datetime import datetime
from typing import Optional
from utils.config_loader import get_config

_KEY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "assets", ".monika_key")

def _ensure_key() -> bytes:
    """读取或生成 32 字节随机密钥（持久化到 assets/.monika_key）。
    
    密钥随便携目录一起分发，换机器也能解密。安全性：防文件浏览，不防整体窃取。
    """
    os.makedirs(os.path.dirname(_KEY_FILE), exist_ok=True)
    if os.path.isfile(_KEY_FILE):
        with open(_KEY_FILE, "rb") as f:
            return f.read()
    key = secrets.token_bytes(32)
    with open(_KEY_FILE, "wb") as f:
        f.write(key)
    return key

def _xor_encrypt(data: bytes, key: bytes) -> bytes:
    """XOR 加密（配合随机 nonce），对非关键数据足够。"""
    # 生成 16 字节随机 nonce 作为前缀
    nonce = secrets.token_bytes(16)
    # 用 nonce + key 生成密钥流
    keystream = hashlib.sha256(nonce + key).digest()
    # 扩展密钥流到数据长度
    full_keystream = keystream
    while len(full_keystream) < len(data):
        full_keystream += hashlib.sha256(full_keystream[-32:] + key).digest()
    encrypted = bytes(a ^ b for a, b in zip(data, full_keystream[:len(data)]))
    return nonce + encrypted

def _xor_decrypt(encrypted: bytes, key: bytes) -> bytes:
    """XOR 解密，还原原始数据。"""
    nonce = encrypted[:16]
    data = encrypted[16:]
    keystream = hashlib.sha256(nonce + key).digest()
    full_keystream = keystream
    while len(full_keystream) < len(data):
        full_keystream += hashlib.sha256(full_keystream[-32:] + key).digest()
    return bytes(a ^ b for a, b in zip(data, full_keystream[:len(data)]))

def encrypt_data(plaintext: str) -> str:
    """加密字符串，返回 base64 编码的密文。"""
    key = _ensure_key()
    encrypted = _xor_encrypt(plaintext.encode("utf-8"), key)
    return base64.urlsafe_b64encode(encrypted).decode("ascii")

def decrypt_data(ciphertext: str) -> Optional[str]:
    """解密 base64 密文，失败返回 None。"""
    try:
        key = _ensure_key()
        encrypted = base64.urlsafe_b64decode(ciphertext.encode("ascii"))
        return _xor_decrypt(encrypted, key).decode("utf-8")
    except Exception:
        return None

def encrypt_json(data: dict | list) -> str:
    """加密 JSON 可序列化数据为 base64 字符串。"""
    return encrypt_data(json.dumps(data, ensure_ascii=False))

def decrypt_json(ciphertext: str) -> Optional[dict | list]:
    """解密 base64 字符串为 JSON 对象，失败返回 None。"""
    plain = decrypt_data(ciphertext)
    if plain is None:
        return None
    try:
        return json.loads(plain)
    except json.JSONDecodeError:
        return None


class Tools:
    def __init__(self):
        pass
    
    def get_current_time_str(self):
        '''获取当前时间的格式化字符串'''
        try:
            time_format = get_config("context.time.format", "%Y/%m/%d 周一 %H:%M")
            current_time = datetime.now()
            weekday_cn = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][current_time.weekday()] # 翻译

            if "%A" in time_format:
                # 暂时用占位符替换 %A，进行时间格式化
                temp_format = time_format.replace("%A", "__WEEKDAY__")
                time_str = current_time.strftime(temp_format)
                time_str = time_str.replace("__WEEKDAY__", weekday_cn)
            else:
                time_str = current_time.strftime(time_format)
            return time_str
        except Exception as e:
            print(f"!!! [Tools] 获取时间字符串失败: {e}")
            return datetime.now().strftime("%Y-%m-%d %H:%M")