'''
窗口布局持久化管理器
负责保存/恢复窗口位置、尺寸、分割器比例及 Live2D 变换
'''

import os
import json
import logging

logger = logging.getLogger(__name__)


class LayoutManager:
    '''窗口布局持久化

    与 MonikaGUI 配合使用：GUI 将自身引用注入，LayoutManager 负责序列化/反序列化
    '''

    _STATE_FILE = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets", "window_state.json"
    )

    def __init__(self, window, splitter, live2d_viewer):
        '''
        Args:
            window:         QMainWindow 实例
            splitter:       QSplitter 实例
            live2d_viewer:  Live2DWidget 实例
        '''
        self._window = window
        self._splitter = splitter
        self._live2d = live2d_viewer

        # 两种模式的缓存
        self.splitter_sizes_normal = []
        self.splitter_sizes_maximized = []
        self.live2d_normal = {}
        self.live2d_maximized = {}

    # -- 公开 API --------------------------------------------

    def load(self) -> dict:
        '''从磁盘读取窗口布局状态（失败时返回空字典）'''
        try:
            if os.path.isfile(self._STATE_FILE):
                with open(self._STATE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def save(self):
        '''将窗口布局状态持久化到磁盘'''
        try:
            cur_sizes = list(self._splitter.sizes())
            if self._window.isMaximized():
                self.splitter_sizes_maximized = cur_sizes
            else:
                self.splitter_sizes_normal = cur_sizes

            geo = self._window.normalGeometry()
            state = {
                "was_maximized": self._window.isMaximized(),
                "normal": {
                    "x": geo.x(), "y": geo.y(),
                    "w": geo.width(), "h": geo.height(),
                    "splitter": self.splitter_sizes_normal,
                    "live2d":   self.live2d_normal,
                },
                "maximized": {
                    "splitter": self.splitter_sizes_maximized,
                    "live2d":   self.live2d_maximized,
                },
            }
            os.makedirs(os.path.dirname(self._STATE_FILE), exist_ok=True)
            with open(self._STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning("[Layout] 窗口布局保存失败: %s", e)

    def restore(self, state: dict, apply_splitter_fn, set_geometry_fn):
        '''根据已加载的 state dict 恢复窗口布局

        Args:
            state:             load() 返回的字典
            apply_splitter_fn: callable(sizes) -> 原子化设置分割器比例
            set_geometry_fn:   callable(x, y, w, h) -> 设置窗口几何
        '''
        if not state:
            return

        if "normal" in state:
            g = state["normal"]
            if all(k in g for k in ("x", "y", "w", "h")):
                set_geometry_fn(g["x"], g["y"], g["w"], g["h"])
            if g.get("splitter"):
                self.splitter_sizes_normal = list(g["splitter"])
            if g.get("live2d"):
                self.live2d_normal = dict(g["live2d"])

        mx = state.get("maximized", {})
        if mx.get("splitter"):
            self.splitter_sizes_maximized = list(mx["splitter"])
        if mx.get("live2d"):
            self.live2d_maximized = dict(mx["live2d"])

        if state.get("was_maximized"):
            # 延迟最大化以让窗口先完成基础布局
            from PyQt6.QtCore import QTimer
            self._window.showMaximized()
            QTimer.singleShot(100, lambda: apply_splitter_fn(self.splitter_sizes_maximized))
        else:
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, lambda: apply_splitter_fn(self.splitter_sizes_normal))

    # -- Live2D 模式切换 -------------------------------------

    def switch_live2d_mode(self, to_maximized: bool):
        '''保存旧模式的 Live2D 变换，并应用新模式的变换（JS 异步，无闪烁）'''
        if not (self._live2d.isVisible() and self._live2d._model_loaded):
            return
        old_key = "maximized" if not to_maximized else "normal"

        def _on_got(result):
            if result and isinstance(result, dict):
                try:
                    data = {
                        "scale": round(float(result.get("scale", 1.0)), 4),
                        "x":     round(float(result.get("x", 0)), 2),
                        "y":     round(float(result.get("y", 0)), 2),
                    }
                    if old_key == "normal":
                        self.live2d_normal = data
                    else:
                        self.live2d_maximized = data
                except Exception:
                    pass
            new = self.live2d_maximized if to_maximized else self.live2d_normal
            if new:
                self._live2d.apply_transform(new["scale"], new["x"], new["y"])

        self._live2d.page().runJavaScript(
            "window.getModelTransform ? window.getModelTransform() : null", _on_got
        )

    def on_splitter_moved(self):
        '''分割器拖动时更新当前模式缓存'''
        sizes = list(self._splitter.sizes())
        if self._window.isMaximized():
            self.splitter_sizes_maximized = sizes
        else:
            self.splitter_sizes_normal = sizes
