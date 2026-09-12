'''
PTT (Push-To-Talk) 全局热键管理器
基于 keyboard 库的后台线程监听，与 MonikaGUI 解耦
'''

import asyncio
import threading
import logging

try:
    import keyboard as _keyboard_lib
    _KEYBOARD_AVAILABLE = True
except ImportError:
    _keyboard_lib = None
    _KEYBOARD_AVAILABLE = False

from utils.config_loader import get_config

logger = logging.getLogger(__name__)


class PttHandler:
    '''PTT 全局热键管理

    用法:
        handler = PttHandler(on_press_coro, on_release_coro)
        handler.start()
        ...
        handler.stop()
    '''

    def __init__(self, on_press_coro, on_release_coro):
        '''
        Args:
            on_press_coro:   async callable() — 按下热键时调用
            on_release_coro: async callable() — 释放热键时调用
        '''
        self._on_press = on_press_coro
        self._on_release = on_release_coro
        self._thread = None
        self._hotkey_name = None
        self._key_held = False
        self._shutdown_flag = False

    # -- 静态工具 --------------------------------------------

    @staticmethod
    def normalize_key(key_str: str) -> str | None:
        '''将 Qt 键名转换为 keyboard 库识别的名字'''
        k = key_str.strip().lower()
        mapping = {
            'space': 'space', 'return': 'enter', 'enter': 'enter',
            'escape': 'esc', 'esc': 'esc', 'tab': 'tab',
            'backspace': 'backspace', 'delete': 'delete', 'del': 'delete',
            'insert': 'insert', 'ins': 'insert',
            'home': 'home', 'end': 'end',
            'pageup': 'page up', 'pagedown': 'page down',
            'up': 'up', 'down': 'down', 'left': 'left', 'right': 'right',
            'shift': 'shift', 'ctrl': 'ctrl', 'alt': 'alt',
        }
        if k in mapping:
            return mapping[k]
        if len(k) == 1:
            return k
        if k.startswith('f') and k[1:].isdigit():
            return k
        return None

    # -- 生命周期 --------------------------------------------

    def start(self, hotkey: str | None = None):
        '''启动后台线程监听全局热键

        Args:
            hotkey: 热键字符串（如 "Space"），None 则从 config 读取
        '''
        if not _KEYBOARD_AVAILABLE:
            logger.warning("[PTT] keyboard 库未安装，全局热键不可用pip install keyboard")
            return

        key_str = hotkey or get_config("audio.ptt_key", "Space")
        normalized = self.normalize_key(key_str)
        if not normalized:
            logger.warning("[PTT] 无法识别热键: %s", key_str)
            return

        if normalized == self._hotkey_name and self._thread and self._thread.is_alive():
            return

        self.stop()
        self._hotkey_name = normalized
        self._key_held = False
        self._shutdown_flag = False
        loop = asyncio.get_event_loop()

        def _listener():
            def on_press(event):
                if self._shutdown_flag:
                    return
                if not self._key_held:
                    self._key_held = True
                    loop.call_soon_threadsafe(
                        lambda: asyncio.ensure_future(self._on_press())
                    )

            def on_release(event):
                if self._shutdown_flag:
                    return
                if self._key_held:
                    self._key_held = False
                    loop.call_soon_threadsafe(
                        lambda: asyncio.ensure_future(self._on_release())
                    )

            try:
                _keyboard_lib.on_press_key(normalized, on_press, suppress=False)
                _keyboard_lib.on_release_key(normalized, on_release, suppress=False)
                logger.info("[PTT] 全局热键已启动: [%s]", normalized)
                while not self._shutdown_flag:
                    threading.Event().wait(timeout=0.5)
            except Exception as e:
                logger.error("[PTT] 热键监听失败: %s", e)
            finally:
                try:
                    _keyboard_lib.unhook_all()
                except Exception:
                    pass

        self._thread = threading.Thread(target=_listener, daemon=True, name="PTT-Hotkey")
        self._thread.start()

    def stop(self):
        '''停止热键监听'''
        self._shutdown_flag = True
        self._hotkey_name = None
        self._key_held = False
        if self._thread and self._thread.is_alive():
            try:
                _keyboard_lib.unhook_all()
            except Exception:
                pass
            self._thread = None
