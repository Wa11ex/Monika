'''
Monika GUI 主窗口
'''

import os
import json
import asyncio
import logging
import random
import threading
try:
    import keyboard as _keyboard_lib
    _KEYBOARD_AVAILABLE = True
except ImportError:
    _keyboard_lib = None
    _KEYBOARD_AVAILABLE = False
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextBrowser, QLineEdit, QPushButton, QLabel, QToolButton, QSplitter, QSizePolicy,
    QProgressBar,
)
from PyQt6.QtCore import Qt, QEvent, QTimer, QAbstractNativeEventFilter
from PyQt6.QtGui import QFont, QCursor, QTextCursor, QTextBlockFormat, QTextFrameFormat
from qasync import asyncSlot

from main import init_monika, shutdown_monika
from utils.config_loader import get_config
from utils.logger import QtLogHandler
from .utils import strip_metadata
from .settings_dialog import SettingsDialog
from .live2d_widget import Live2DWidget
from .layout_manager import LayoutManager
from .ptt_handler import PttHandler
from .theme import get_stylesheet
import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# AIUEO 调试面板
# ══════════════════════════════════════════════════════════════

class AiueoDebugPanel(QWidget):
    '''
    实时 AIUEO 口型调试面板
    在输入框中输入 \\check --sound 即可切换显示（命令注册表见 ui/debug_commands.py）
    由 bus.lipsync_callback 驱动更新（在 asyncio/qasync 主线程中调用，安全）
    '''
    _VOWELS = ["A", "I", "U", "E", "O"]
    _COLORS = {"A": "#ff6655", "I": "#55aaff", "U": "#aa77ff", "E": "#ffcc33", "O": "#55ddaa"}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(72)
        self.setStyleSheet(
            "background-color: #1a1a2e;"
            "border-top: 1px solid #3e3e42;"
            "border-radius: 6px;"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(10)

        # 标题
        lbl = QLabel("AIUEO")
        lbl.setStyleSheet("color: #7788aa; font-size: 8pt; font-weight: bold; background: transparent;")
        layout.addWidget(lbl)

        # 5 个元音进度条
        self._bars: dict[str, QProgressBar] = {}
        for v in self._VOWELS:
            col = QVBoxLayout()
            col.setSpacing(2)
            col.setContentsMargins(0, 0, 0, 0)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedWidth(22)
            bar.setOrientation(Qt.Orientation.Vertical)
            c = self._COLORS[v]
            bar.setStyleSheet(
                f"QProgressBar {{ background: #222233; border-radius: 3px; border: none; }}"
                f"QProgressBar::chunk {{ background: {c}; border-radius: 3px; }}"
            )
            col.addWidget(bar)

            lbl_v = QLabel(v)
            lbl_v.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            lbl_v.setStyleSheet(
                f"color: {c}; font-size: 7pt; font-weight: bold; background: transparent;"
            )
            col.addWidget(lbl_v)

            layout.addLayout(col)
            self._bars[v] = bar

        # MouthOpenY 条
        layout.addSpacing(6)
        col_oy = QVBoxLayout()
        col_oy.setSpacing(2)
        col_oy.setContentsMargins(0, 0, 0, 0)
        self._oy_bar = QProgressBar()
        self._oy_bar.setRange(0, 100)
        self._oy_bar.setValue(0)
        self._oy_bar.setTextVisible(False)
        self._oy_bar.setFixedWidth(22)
        self._oy_bar.setOrientation(Qt.Orientation.Vertical)
        self._oy_bar.setStyleSheet(
            "QProgressBar { background: #222233; border-radius: 3px; border: none; }"
            "QProgressBar::chunk { background: #33dd88; border-radius: 3px; }"
        )
        col_oy.addWidget(self._oy_bar)
        lbl_oy = QLabel("Y")
        lbl_oy.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        lbl_oy.setStyleSheet("color: #33dd88; font-size: 7pt; background: transparent;")
        col_oy.addWidget(lbl_oy)
        layout.addLayout(col_oy)

        layout.addStretch()

        # 收起按钮
        close_btn = QPushButton("∧ 收起")
        close_btn.setFixedHeight(24)
        close_btn.setStyleSheet(
            "QPushButton { background: #2d2d40; color: #aaaacc; border: 1px solid #444466;"
            "  border-radius: 4px; font-size: 8pt; padding: 0 8px; }"
            "QPushButton:hover { background: #3a3a55; }"
        )
        close_btn.clicked.connect(self.hide)
        layout.addWidget(close_btn)

    def update_scores(self, open_y: float, scores: "np.ndarray") -> None:
        '''由 bus.lipsync_callback 调用（qasync 主线程，无需加锁）'''
        self._oy_bar.setValue(int(open_y * 100))
        for i, v in enumerate(self._VOWELS):
            self._bars[v].setValue(int(float(scores[i]) * 100))


class _EdgeResizeFilter(QAbstractNativeEventFilter):
    '''应用级 WM_NCHITTEST 过滤器——绕过 PyQt6 nativeEvent 虚函数的 sip 崩溃问题'''

    def __init__(self, window):
        super().__init__()
        self._win = window

    def nativeEventFilter(self, eventType, message):
        if eventType != b"windows_generic_MSG":
            return False, 0
        if self._win.isMaximized():
            return False, 0
        try:
            import ctypes
            addr = int(message)
            if not addr:
                return False, 0
            # MSG 64-bit 布局: HWND(8) + UINT message(4)
            hwnd_size = ctypes.sizeof(ctypes.c_void_p)  # 8
            # 读取消息目标 HWND（偏移 0），只处理主窗口的消息
            msg_hwnd = ctypes.c_void_p.from_address(addr).value
            if msg_hwnd != int(self._win.winId()):
                return False, 0
            msg_id = ctypes.c_uint.from_address(addr + hwnd_size).value
            if msg_id == 0x0084:  # WM_NCHITTEST
                pos = QCursor.pos()
                x, y = pos.x(), pos.y()
                geo = self._win.frameGeometry()
                e = self._win._EDGE
                on_l = x < geo.left()   + e
                on_r = x > geo.right()  - e
                on_t = y < geo.top()    + e
                on_b = y > geo.bottom() - e
                if on_t and on_l: return True, 13  # HTTOPLEFT
                if on_t and on_r: return True, 14  # HTTOPRIGHT
                if on_b and on_l: return True, 16  # HTBOTTOMLEFT
                if on_b and on_r: return True, 17  # HTBOTTOMRIGHT
                if on_l:          return True, 10  # HTLEFT
                if on_r:          return True, 11  # HTRIGHT
                if on_t:          return True, 12  # HTTOP
                if on_b:          return True, 15  # HTBOTTOM
        except Exception:
            pass
        return False, 0


class TitleBar(QWidget):
    '''自定义无边框标题栏（拖拽移动、双击最大化）'''

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self.setStyleSheet(
            "background-color: #2d2d30;"
            "border-bottom: 1px solid #3e3e42;"
        )
        self._drag_pos = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 4, 0)
        layout.setSpacing(0)

        title = QLabel("Monika V3")
        title.setStyleSheet("color: #cccccc; font-size: 9pt; background: transparent; border: none;")
        layout.addWidget(title)
        layout.addStretch()

        _btn = (
            "QPushButton { background: transparent; color: #cccccc; border: none; font-size: 11pt; }"
            "QPushButton:hover { background-color: #3e3e42; }"
        )
        _close = (
            "QPushButton { background: transparent; color: #cccccc; border: none; font-size: 10pt; }"
            "QPushButton:hover { background-color: #c42b1c; color: white; }"
        )

        for text, style, slot in [
            ("\u2500", _btn, lambda: self.window().showMinimized()),
            ("\u25a1", _btn, self._toggle_max),
            ("\u2715", _close, lambda: self.window().close()),
        ]:
            btn = QPushButton(text)
            btn.setFixedSize(40, 32)
            btn.setStyleSheet(style)
            btn.clicked.connect(slot)
            layout.addWidget(btn)

    def _toggle_max(self):
        w = self.window()
        if w.isMaximized():
            w.showNormal()
        else:
            w.showMaximized()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self._drag_pos is not None:
            if not self.window().isMaximized():
                delta = event.globalPosition().toPoint() - self._drag_pos
                self.window().move(self.window().pos() + delta)
            self._drag_pos = event.globalPosition().toPoint()
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    def mouseDoubleClickEvent(self, event):
        self._toggle_max()


class _ChatDisplay(QTextBrowser):
    '''QTextBrowser 子类：阻止自定义 scheme 触发内部导航清空文档
    
    setSource() 不是虚函数，Python 覆盖无效绕过方案：重写 mouseReleaseEvent，
    对 feedback:/confirm: 手动发射 anchorClicked 信号后直接 return，
    不调用父类（从而不走 setSource -> 不会清空文档）
    '''
    def mouseReleaseEvent(self, event):
        from PyQt6.QtCore import QUrl
        anchor = self.anchorAt(event.pos())
        if anchor:
            url = QUrl(anchor)
            if url.scheme() in ('feedback', 'confirm'):
                self.anchorClicked.emit(url)  # 只发信号，阻止默认导航
                return
        super().mouseReleaseEvent(event)


class MonikaGUI(QMainWindow):
    '''Monika V3 主窗口'''

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monika V3")
        self.resize(1020, 540)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        # 应用深色主题
        self.setStyleSheet(get_stylesheet())

        self.components = {}
        self.bus = None
        self.brain = None
        self.ears = None
        self._listen_task = None
        self._ptt_task = None         # PTT 录音任务
        self._is_shutting_down = False
        self._is_processing = False   # 正在说话吗
        self._listener_gen = 0        # 每次开麦递增，用于丢弃过期线程的结果
        self._mic_in_use = False      # True = listen_auto() 线程正持有麦克风流
        self._init_dot_task = None    # 动画小点任务
        self._po_task = None          # 主动输出（PO）定时器任务
        self._po_reset = None         # asyncio.Event：真实用户输入时 set，触发 PO 重置

        # 聊天消息日志（用于 debug 热插拔）
        # 格式： (type, html)  type = 'debug' | 'user' | 'monika'
        self._message_log: list[tuple[str, str]] = []

        # 当前 Monika 响应的 QTextFrame 容器（隔离 Monika 句子流和外部内容）
        self._monika_frame = None

        self._EDGE = 6                # 边缘热区宽度 px
        self._resize_filter = None    # 边缘缩放过滤器（showEvent 后安装）
        self._layout_restored = False # 布局已从文件恢复（仅触发一次）
        self._restoring_layout = False # 正在执行初始布局恢复（屏蔽 changeEvent 覆写）
        self._live2d_normal: dict | None = None     # 普通模式的 Live2D 变换缓存
        self._live2d_maximized: dict | None = None  # 最大化模式的 Live2D 变换缓存

        # 输入历史
        self._input_history: list[str] = []
        self._history_index: int = -1

        # 子模块管理器（延迟创建，等 UI 控件构造完毕）
        self.layout_mgr: LayoutManager | None = None
        self.ptt_handler: PttHandler | None = None

        # -- 中心布局 -------------------------------------------
        main_widget = QWidget()
        main_widget.setStyleSheet("background-color: #1e1e1e;")
        self.setCentralWidget(main_widget)

        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 自定义标题栏（无边框模式）
        main_layout.addWidget(TitleBar(self))

        # 内容区（带内边距）
        _content = QWidget()
        _content.setStyleSheet("background-color: #1e1e1e;")
        _cl = QVBoxLayout(_content)
        _cl.setContentsMargins(12, 8, 12, 12)
        _cl.setSpacing(12)

        # 创建分裂器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("background-color: #1e1e1e;")
        
        # 左侧面板 (Chat & Controls)
        left_panel = QWidget()
        left_panel.setStyleSheet("background-color: #1e1e1e;")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 状态栏
        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        self.status_label = QLabel("状态: 正在初始化...")
        self.emotion_label = QLabel("情绪: N/A")
        self.status_label.setStyleSheet("color: #858585; font-size: 10pt;")
        self.emotion_label.setStyleSheet("color: #858585; font-size: 10pt;")
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        status_row.addWidget(self.emotion_label)
        left_layout.addLayout(status_row)

        # 聊天区
        self.chat_display = _ChatDisplay()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(self._create_font("Segoe UI", 10))
        self.chat_display.setStyleSheet("QTextEdit { background-color: #252526; color: #e0e0e0; border: 1px solid #3e3e42; border-radius: 6px; padding: 8px; }")
        self.chat_display.anchorClicked.connect(self._on_feedback_link)
        left_layout.addWidget(self.chat_display, 1)

        # 底部工具行
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(8)

        # 麦克风常开切换
        self.mic_btn = QPushButton("🎙 开启麦克风")
        self.mic_btn.setCheckable(True)
        self.mic_btn.setMinimumHeight(48)
        self.mic_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.mic_btn.setFont(self._create_font("Segoe UI", 7, bold=True))
        self.mic_btn.setToolTip("开启麦克风（常开模式）")
        self.mic_btn.clicked.connect(self.on_mic_toggled)
        bottom_row.addWidget(self.mic_btn)

        # PTT 按钮（仅 push_to_talk 模式显示）
        self.ptt_btn = QPushButton("⏺ 按住说话")
        self.ptt_btn.setVisible(False)
        self.ptt_btn.setMinimumHeight(48)
        self.ptt_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.ptt_btn.setFont(self._create_font("Segoe UI", 7, bold=True))
        self.ptt_btn.setToolTip("按住说话（PTT 模式），说话中再按可打断 Monika")
        self.ptt_btn.pressed.connect(self.on_ptt_pressed)
        self.ptt_btn.released.connect(self.on_ptt_released)
        # PTT 按钮：setAutoRepeat 关闭（防止长按触发多次 pressed）
        self.ptt_btn.setAutoRepeat(False)
        bottom_row.addWidget(self.ptt_btn)

        # 输入框
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("输入消息或开启麦克风...")
        self.input_field.setMinimumHeight(48)
        self.input_field.setMinimumWidth(150)
        self.input_field.returnPressed.connect(self.on_send_clicked)
        self.input_field.installEventFilter(self)
        bottom_row.addWidget(self.input_field, 1)

        # 发送
        self.send_btn = QPushButton("发送")
        self.send_btn.setMinimumHeight(48)
        self.send_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.send_btn.setFont(self._create_font("Segoe UI", 7, bold=True))
        self.send_btn.setToolTip("发送消息")
        self.send_btn.clicked.connect(self.on_send_clicked)
        bottom_row.addWidget(self.send_btn)

        # 切换Live2D
        self.live2d_toggle_btn = QToolButton()
        self.live2d_toggle_btn.setText("👤 切换Live2D")
        self.live2d_toggle_btn.setMinimumHeight(48)
        self.live2d_toggle_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.live2d_toggle_btn.setFont(self._create_font("Segoe UI", 5, bold=True))
        self.live2d_toggle_btn.setToolTip("显示/隐藏 Live2D 小人")
        self.live2d_toggle_btn.clicked.connect(self.toggle_live2d)
        bottom_row.addWidget(self.live2d_toggle_btn)

        # 设置齿轮
        self.settings_btn = QToolButton()
        self.settings_btn.setText("⚙")
        self.settings_btn.setMinimumHeight(48)
        self.settings_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.settings_btn.setFont(self._create_font("Segoe UI", 7, bold=True))
        self.settings_btn.setToolTip("打开设置")
        self.settings_btn.clicked.connect(self.open_settings)
        bottom_row.addWidget(self.settings_btn)

        left_layout.addLayout(bottom_row)

        # AIUEO 调试面板（默认隐藏，由 \check --sound debug 命令切换）
        self._aiueo_panel = AiueoDebugPanel()
        self._aiueo_panel.setVisible(False)
        left_layout.addWidget(self._aiueo_panel)
        self.live2d_viewer = Live2DWidget()
        self.live2d_viewer.setMinimumWidth(250)
        left_panel.setMinimumWidth(300)
        
        # 组装分裂器
        splitter.addWidget(left_panel)
        splitter.addWidget(self.live2d_viewer)
        splitter.setSizes([750, 270])  # 默认占宽比
        # Live2D 面板不占用最小宽时，允许窗口一起缩小
        self.live2d_viewer.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        splitter.setStretchFactor(0, 1)  # 左侧能伸缩
        splitter.setStretchFactor(1, 0)  # 右侧不强制伸展

        self.splitter = splitter
        self._splitter_sizes_normal: list = [750, 270]    # 普通模式分割器像素尺寸
        self._splitter_sizes_maximized: list = [750, 270] # 最大化模式分割器像素尺寸
        splitter.splitterMoved.connect(self._on_splitter_moved)
        _cl.addWidget(splitter)
        main_layout.addWidget(_content, 1)

        # -- 子模块管理器 --------------------------------------
        self.layout_mgr = LayoutManager(self, splitter, self.live2d_viewer)
        self.ptt_handler = PttHandler(self.on_ptt_pressed, self.on_ptt_released)

        # -- 启动初始化 -------------------------------------
        self._log(">>> [System] 正在唤醒后台服务，请稍候...")
        self._init_dot_task = asyncio.ensure_future(self._animate_init_status())
        asyncio.ensure_future(self._init_async())

    # -- 输入历史 ------------------------------------------

    def eventFilter(self, obj, event):
        '''拦截输入框的上下方向键，实现历史记录翻页'''
        if obj is self.input_field and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key == Qt.Key.Key_Up and self._input_history:
                if self._history_index < len(self._input_history) - 1:
                    self._history_index += 1
                    self.input_field.setText(
                        self._input_history[-(self._history_index + 1)]
                    )
                return True  # 消费事件（阻止光标移动到行首）
            elif key == Qt.Key.Key_Down and self._input_history:
                if self._history_index > 0:
                    self._history_index -= 1
                    self.input_field.setText(
                        self._input_history[-(self._history_index + 1)]
                    )
                elif self._history_index == 0:
                    self._history_index = -1
                    self.input_field.clear()
                return True  # 消费事件
        # 未处理的事件交给默认处理（Enter、普通输入等）
        return False

    # -- 工具方法 --------------------------------------------
    def _create_font(self, family="Segoe UI", size=10, bold=False):
        '''创建字体对象'''
        font = QFont(family, size)
        if bold:
            font.setBold(True)
        return font

    # -- 边缘缩放（由 showEvent 安装 _EdgeResizeFilter）--------

    def showEvent(self, event):
        super().showEvent(event)
        self._apply_dwm_rounded_corners()
        # 安装应用级边缘缩放过滤器（仅安装一次）
        if self._resize_filter is None:
            from PyQt6.QtWidgets import QApplication
            self._resize_filter = _EdgeResizeFilter(self)
            QApplication.instance().installNativeEventFilter(self._resize_filter)
        # 首次显示时从文件恢复窗口布局
        if not self._layout_restored:
            self._layout_restored = True
            QTimer.singleShot(0, self._restore_layout_from_file)

    def _apply_dwm_rounded_corners(self):
        '''Win11 DWM 原生圆角（Win10 / 其他平台静默跳过）'''
        try:
            import ctypes
            DWMWA_WINDOW_CORNER_PREFERENCE = 33
            DWMWCP_ROUND = 2
            hwnd = int(self.winId())
            pref = ctypes.c_int(DWMWCP_ROUND)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_WINDOW_CORNER_PREFERENCE,
                ctypes.byref(pref), ctypes.sizeof(pref)
            )
        except Exception:
            pass  # Win10 或非 Windows 平台静默跳过

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() != QEvent.Type.WindowStateChange:
            return
        if self._restoring_layout:
            return  # 初始布局恢复期间屏蔽，防止覆写已读取的尺寸
        if self.isMaximized():
            # 普通 -> 最大化：保存普通尺寸，原子切换至最大化尺寸
            self._splitter_sizes_normal = list(self.splitter.sizes())
            self._apply_splitter_atomic(self._splitter_sizes_maximized)
            self.layout_mgr.switch_live2d_mode(to_maximized=True)
        elif not self.isMinimized():
            # 最大化 -> 普通：保存最大化尺寸，原子切换至普通尺寸
            self._splitter_sizes_maximized = list(self.splitter.sizes())
            self._apply_splitter_atomic(self._splitter_sizes_normal)
            self.layout_mgr.switch_live2d_mode(to_maximized=False)

    def _on_splitter_moved(self, pos, index):
        self.layout_mgr.on_splitter_moved()

    def _apply_splitter_atomic(self, sizes: list):
        '''在单帧内完成分割器尺寸切换，避免中间态闪烁'''
        if not sizes:
            return
        self.setUpdatesEnabled(False)
        try:
            self.splitter.setSizes(sizes)
        finally:
            self.setUpdatesEnabled(True)

    def _load_window_state(self) -> dict:
        '''委托给 LayoutManager'''
        return self.layout_mgr.load()

    def _save_window_state(self):
        '''委托给 LayoutManager'''
        self.layout_mgr.save()

    def _restore_layout_from_file(self):
        '''委托给 LayoutManager'''
        state = self.layout_mgr.load()
        if not state:
            return
        self.layout_mgr.restore(
            state,
            apply_splitter_fn=self._apply_splitter_atomic,
            set_geometry_fn=self.setGeometry,
        )
        # 同步 _splitter_sizes_* 缓存以兼容 changeEvent
        self._splitter_sizes_normal = self.layout_mgr.splitter_sizes_normal
        self._splitter_sizes_maximized = self.layout_mgr.splitter_sizes_maximized
        self._live2d_normal = self.layout_mgr.live2d_normal
        self._live2d_maximized = self.layout_mgr.live2d_maximized
        if state.get("was_maximized"):
            self._restoring_layout = True
            # showMaximized 在 restore 中已调用，这里不再重复
            self._restoring_layout = False

    async def _animate_init_status(self):
        '''初始化期间小点动画，让系统任务栏知道应用在响应'''
        dots = 0
        try:
            while True:
                dots = (dots % 3) + 1
                self.status_label.setText(f"状态: 初始化中{'.' * dots}")
                await asyncio.sleep(0.6)
        except asyncio.CancelledError:
            pass
    def _log(self, text: str):
        '''向聊天框追加系统消息'''
        html = f'<span style="color: #858585; font-size: 9pt;">{text}</span>'
        self._message_log.append(("debug", html))
        if not get_config("ui.debug_mode", True):
            self.chat_display.document().setModified(True)  # 打断 QTextBrowser setSource 导航
            return
        # 若帧活跃 -> 写入帧内（<br> 软换行，无段落间距）
        frame = self._monika_frame
        if frame is not None:
            cursor = frame.lastCursorPosition()
            if cursor.position() > frame.firstCursorPosition().position():
                cursor.insertHtml('<br>')
            cursor.insertHtml(html)
        else:
            self.chat_display.append(html)

    def _log_monika(self, raw: str):
        '''追加 Monika 的回复，过滤元数据'''
        clean = strip_metadata(raw)
        if clean:
            html = (
                f'<span style="color: #4ec9b0;"><b>Monika:</b></span> '
                f'<span style="color: #e0e0e0;">{clean}</span>'
            )
            self._message_log.append(("monika", html))
            self.chat_display.append(html)

    def _log_user(self, text: str):
        html = (
            f'<span style="color: #007acc;"><b>你:</b></span> '
            f'<span style="color: #e0e0e0;">{text}</span>'
        )
        self._message_log.append(("user", html))
        self.chat_display.append(html)

    def _on_feedback_link(self, url):
        '''处理链接点击：feedback:up/down 或 confirm:allow/deny'''
        scheme = url.scheme()
        # -- 工具确认 --
        if scheme == "confirm":
            action = url.path()  # "allow" or "deny"
            if hasattr(self, 'brain') and hasattr(self.brain, '_confirm_request'):
                req = self.brain._confirm_request
                if req and not req.is_set():
                    self.brain._confirm_result = (action == "allow")
                    self._log(f"[确认] {'已允许' if action == 'allow' else '已拒绝'}")
                    req.set()
            return
        if scheme != "feedback":
            return
        fb_type = url.path()  # "up" or "down"（feedback:up -> path="up"）
        raw = getattr(self, '_last_monika_response', "")
        if not raw:
            return
        clean = strip_metadata(raw)
        label = "点赞" if fb_type == "up" else "点踩"
        try:
            mem = self.components.get("memory_manager") if hasattr(self, 'components') else None
            if mem and hasattr(mem, 'add_conversation'):
                mem.add_conversation(
                    f"[反馈: {label}]",
                    clean,
                    metadata={"feedback": fb_type, "timestamp": __import__('datetime').datetime.now().isoformat()}
                )
            self._log(f"[反馈] 已记录: {label}")
        except Exception as e:
            logger.warning(f"[反馈] 存储失败: {e}")

    def _redraw_chat(self):
        '''根据当前 debug 模式重绘聊天框（切换调试显示时调用）'''
        show_debug = get_config("ui.debug_mode", True)
        self.chat_display.clear()
        for typ, html in self._message_log:
            if typ == "debug" and not show_debug:
                continue
            self.chat_display.append(html)

    def _set_ui_busy(self, busy: bool):
        self.input_field.setEnabled(not busy)
        self.send_btn.setEnabled(not busy)
        self.ptt_btn.setEnabled(not busy)

    # -- 初始化 --------------------------------------------
    async def _init_async(self):
        try:
            self.components = await init_monika(log_callback=self._log)
            self.bus   = self.components["bus"]
            self.brain = self.components["brain"]
            self.ears  = self.components["ears"]

            # 将内置 Live2D 控制器注入 Bus，使 LLM 输出的表情标记同步驱动 Live2D
            self.bus.live2d = self.live2d_viewer

            # 口型同步回调：将 AIUEO 数据推送到 debug 面板
            def _on_lipsync(open_y, scores):
                if self._aiueo_panel.isVisible():
                    self._aiueo_panel.update_scores(open_y, scores)
            self.bus.lipsync_callback = _on_lipsync

            # 将 logging 系统输出转发到 GUI 聊天窗口
            _qt_handler = QtLogHandler(self._log)
            logging.getLogger().addHandler(_qt_handler)

            # 停止小点动画
            if self._init_dot_task and not self._init_dot_task.done():
                self._init_dot_task.cancel()

            input_mode = get_config("audio.input_mode", "typing")

            # -- 恢复 Live2D 显示状态（记忆上次选择）------------------------
            live2d_visible = get_config("ui.live2d_visible", True)
            live2d_model = get_config("ui.live2d_model", "")
            if live2d_visible and live2d_model:
                self.status_label.setText("状态: 正在加载 Live2D...")
                self.live2d_viewer.show()
                # 注入当前模式对应的 Live2D 变换（load_model 内部优先应用）
                _t = self.layout_mgr.live2d_maximized if self.isMaximized() else self.layout_mgr.live2d_normal
                if _t:
                    self.live2d_viewer._pending_transform = (_t["scale"], _t["x"], _t["y"])
                self.live2d_viewer.load_model(live2d_model)
                await asyncio.sleep(3)  # 等 Live2D JS 加载
            else:
                self.live2d_viewer.hide()
                self._update_window_size()

            # -- 就绪提示（绿色高亮）----------------------------------------
            if input_mode == "mic_always_on":
                self.mic_btn.setChecked(True)
                self.on_mic_toggled(True)
                ready_hint = "🟢 初始化已就绪，已开始监听"
            elif input_mode == "mic_push_to_talk":
                self.ptt_btn.setVisible(True)
                self._setup_ptt_hotkey()
                ptt_key = get_config("audio.ptt_key", "Space")
                ready_hint = f"🟢 初始化已就绪，按住 [{ptt_key}] 说话"
            else:
                ready_hint = "🟢 初始化已就绪，可以打字输入"

            self.status_label.setText("状态: 待机")
            self.status_label.setStyleSheet("color: #4ec9b0; font-size: 10pt; font-weight: bold;")
            self.chat_display.append(
                f'<span style="color: #4ec9b0; font-size: 10pt; font-weight: bold;">{ready_hint}</span>'
            )

            # -- 启动主动输出（PO）定时器 -----------------------------------
            if self.components.get("po") is not None:
                self._po_reset = asyncio.Event()
                self._po_task = asyncio.ensure_future(self._po_loop())
                logger.info("[GUI] PO 定时器已启动，冷却窗口: %d", self.components["po"].total_windows)

        except Exception as e:
            logger.exception(f"[Main] 初始化失败")
            self._log(f"!!! 初始化失败: {e}")
            self.status_label.setText("状态: 初始化失败")
            if self._init_dot_task and not self._init_dot_task.done():
                self._init_dot_task.cancel()


    # -- PO定时器 ------------------------------
    async def _po_loop(self):
        '''主动输出定时器协程

        将「沉默」定义为：用户长时间未向 LLM 发送内容（与输入模式无关）
        每次真实用户输入会重置计时器；PO 触发后进入下一个冷却窗口；
        用完所有窗口后等待下一次真实输入再重新开始
        '''
        from main import PO_TRIGGER_TEXT
        po = self.components.get("po")
        if po is None:
            return

        po.reset()
        try:
            while not self._is_shutting_down:
                if po.is_exhausted:
                    # 所有冷却窗口已用完，等待下一次真实输入后重置
                    await self._po_reset.wait()
                    self._po_reset.clear()
                    po.reset()
                    continue

                timeout = po.next_timeout()
                if timeout is None:
                    await self._po_reset.wait()
                    self._po_reset.clear()
                    po.reset()
                    continue

                self._po_reset.clear()

                try:
                    await asyncio.wait_for(self._po_reset.wait(), timeout=timeout)
                    # 收到真实用户输入信号 -> 重置到第一个冷却窗口
                    po.reset()
                except asyncio.TimeoutError:
                    # 静默超时 -> 若当前未在处理则触发 PO
                    if not self._is_processing:
                        po.advance()
                        logger.info(
                            "[GUI] 主动发言触发（第 %d/%d 次冷却，等待了 %.0f 秒）",
                            po.current_index, po.total_windows, timeout,
                        )
                        await self._process_input(PO_TRIGGER_TEXT, _is_po=True)
                    # 若 Monika说话中，保持当前窗口，下轮重新等待
        except asyncio.CancelledError:
            pass

    # -- 发送文字 ------------------------------------------
    @asyncSlot()
    async def on_send_clicked(self):
        text = self.input_field.text().strip()
        if not text:
            return
        # 保存到输入历史
        if text not in self._input_history or self._input_history[-1] != text:
            self._input_history.append(text)
        self._history_index = -1
        self.input_field.clear()
        # -- 工具确认拦截 --
        if hasattr(self.brain, '_confirm_request') and self.brain._confirm_request:
            if not self.brain._confirm_request.is_set():
                text_lower = text.strip().lower()
                if text_lower in ("允许", "yes", "y", "ok", "好", "可以"):
                    self.brain._confirm_result = True
                    self._log("[确认] 已允许")
                else:
                    self.brain._confirm_result = False
                    self._log("[确认] 已拒绝")
                self.brain._confirm_request.set()
                return
        # Debug 命令拦截（以 \ 开头的输入不发给 LLM）
        if text.startswith("\\"):
            from ui.debug_commands import dispatch
            dispatch(self, text)
            return
        await self._process_input(text)

    async def _process_input(self, text: str, _is_po: bool = False):
        if not self.bus or not self.brain:
            return
        if self._is_processing:
            return  # 防重入

        # 真实用户输入 -> 通知 PO 定时器重置计时
        if not _is_po and self._po_reset is not None:
            self._po_reset.set()

        # 获取声纹识别结果（供 Lore 硬规则等使用）
        speaker_info = getattr(self.ears, "last_speaker", None) if self.ears else None

        self._is_processing = True
        self._log_user(text)
        self.status_label.setText("状态: 思考中...")
        self._set_ui_busy(True)
        self.live2d_viewer.set_talking(True)

        # -- 为本次响应创建 QTextFrame 容器（margin=0）--
        _cursor = self.chat_display.textCursor()
        _cursor.movePosition(QTextCursor.MoveOperation.End)
        _fmt = QTextFrameFormat()
        _fmt.setMargin(0)
        _fmt.setPadding(0)
        _fmt.setBorder(0)
        self._monika_frame = _cursor.insertFrame(_fmt)

        # 辅助：向 frame 末尾追加内容（首条不换行，后续用 <br> 软换行）
        def _frame_append(html: str):
            frame = self._monika_frame
            if frame is None:
                return
            cursor = frame.lastCursorPosition()
            if cursor.position() > frame.firstCursorPosition().position():
                cursor.insertHtml('<br>')
            cursor.insertHtml(html)
            scrollbar = self.chat_display.verticalScrollBar()
            if scrollbar:
                scrollbar.setValue(scrollbar.maximum())

        # -- 打断监听 ----------------------------------------------------------
        # 仅在麦克风未被后台 listen_auto() 线程占用时，才能开启独立的监听流
        # 若 mic 被占用（文字输入时后台仍在录音），则依靠 _background_listener
        # 的「检测到声音 -> 直接触发 bus.interrupt()」路径来完成打断
        _monitoring_started = False
        watcher = None

        try:
            self.bus.reset_interrupt()

            if self.ears and not self._mic_in_use and self.mic_btn.isChecked():
                self.ears.interrupt_detected = False
                self.ears.start_interrupt_monitoring()
                _monitoring_started = True

                async def _watch_interrupt():
                    # 在音频真正出声前不认为是打断——否则 LLM 思考期间的外界噪音会把尚未开始的回答直接丢弃
                    while not self.bus.is_interrupted and not self.bus.player.has_started_playing:
                        await asyncio.sleep(0.05)
                    while not self.bus.is_interrupted:
                        if self.ears.check_interrupt():
                            logger.info("[GUI] 检测到打断信号!")
                            await self.bus.interrupt()
                            return
                        await asyncio.sleep(0.05)

                watcher = asyncio.create_task(_watch_interrupt())

            # -- 工具确认 --
            import asyncio as _asyncio
            _confirm_request = _asyncio.Event()
            _confirm_request.set()
            self.brain._confirm_request = _confirm_request
            self.brain._confirm_result = False
            self.brain._confirm_info = ""

            async def _watch_confirm():
                while not self.bus.is_interrupted:
                    if not _confirm_request.is_set():
                        info = self.brain._confirm_info or "执行操作"
                        self._log(f"[确认] Monika 想要 {info}")
                        confirm_html = (
                            f'<span style="color:#e0a040;">[确认] Monika 想要: {info}</span><br>'
                            f'<a href="confirm:allow" style="color:#4ec9b0; font-weight:bold;">[ 允许 ]</a>  '
                            f'<a href="confirm:deny" style="color:#e05555; font-weight:bold;">[ 拒绝 ]</a>'
                        )
                        _frame_append(confirm_html)
                        # 等用户点击按钮
                        while not _confirm_request.is_set() and not self.bus.is_interrupted:
                            await _asyncio.sleep(0.3)
                    await _asyncio.sleep(0.5)
            _confirm_task = _asyncio.ensure_future(_watch_confirm())

            # -- TTS 背压：LLM 生成暂停等 TTS 合成完成 --
            import asyncio as _asyncio
            _tts_ready = _asyncio.Event()
            _tts_ready.set()  # 初始就绪
            self.brain._tts_ready = _tts_ready
            self.bus.tts_done_callback = _tts_ready.set

            # -- 逐句显示：所有内容在同一个 QTextFrame 内 --
            # frame 已在上方创建；debug / 确认 / 句子 / 反馈全部在 frame 内
            # 不同类型之间用 margin=0 的 insertBlock() 分隔，无空行
            _sentence_first = [True]

            def _on_sentence(sentence: str, _is_first: bool):
                clean = strip_metadata(sentence)
                if not clean:
                    return

                # -- 新 TALK 块：段间 <br> 软换行 --
                if getattr(self.brain, '_sentence_new_block', False):
                    _sentence_first[0] = True
                    self.brain._sentence_new_block = False

                frame = self._monika_frame
                if frame is None:
                    return

                cursor = frame.lastCursorPosition()

                if _sentence_first[0]:
                    # 帧非空 -> 先 <br>（处理 debug 在 Monika 之前到达的情况）
                    if cursor.position() > frame.firstCursorPosition().position():
                        cursor.insertHtml('<br>')

                    html = (
                        f'<span style="color: #4ec9b0;"><b>Monika:</b></span> '
                        f'<span style="color: #e0e0e0;">{clean}</span>'
                    )
                    self._message_log.append(("monika_sentence", html))
                    cursor.insertHtml(html)
                    _sentence_first[0] = False
                else:
                    html = f'<span style="color: #e0e0e0;">{clean}</span>'
                    self._message_log.append(("monika_sentence", html))
                    cursor.insertHtml(html)

                scrollbar = self.chat_display.verticalScrollBar()
                if scrollbar:
                    scrollbar.setValue(scrollbar.maximum())

            self.bus.sentence_callback = _on_sentence

            # -- LLM 生成 + TTS 流水线 --------------------------------------
            stream = self.brain.think_stream(text, speaker_info=speaker_info)
            raw_response = await self.bus.process_stream_input(stream)

            # -- 等待 TTS 全部播放完毕（而非只等 LLM 输出完毕）------------
            # process_stream_input 返回时 LLM 流结束，但 TTS 队列可能还有
            # 多个句子待合成/播放必须在此等待，否则 _is_processing 会过早
            # 清除，导致后台监听重新打开麦克风，与打断监听流冲突
            if not self.bus.is_interrupted:
                await self.bus.wait_for_completion()

            self.brain.commit_response(raw_response, interrupted=self.bus.is_interrupted)

            # -- 反馈按钮 --------------------------------------------------
            _fb_html = (
                '<span style="font-size:9pt; color:#858585;">'
                '<a href="feedback:up" style="color:#4ec9b0; text-decoration:none;">'
                '[^] 行！</a>  '
                '<a href="feedback:down" style="color:#e05555; text-decoration:none;">'
                '[v] 不行</a>'
                '</span>'
            )
            self._message_log.append(("feedback", _fb_html))
            _frame_append(_fb_html)
            scrollbar = self.chat_display.verticalScrollBar()
            if scrollbar:
                scrollbar.setValue(scrollbar.maximum())
            # 存储本轮 raw_response 引用，供反馈回调使用
            self._last_monika_response = raw_response

            if self.ears and hasattr(self.ears, 'last_emotion') and self.ears.last_emotion:
                self.emotion_label.setText(f"情绪: {self.ears.last_emotion}")
                self.live2d_viewer.set_emotion(self.ears.last_emotion)

        except Exception as e:
            self._log(f"!!! 通信错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # -- 清理回调 --------------------------------------------------
            self.bus.sentence_callback = None
            self.bus.tts_done_callback = None
            self._monika_frame = None
            if hasattr(self.brain, '_tts_ready'):
                self.brain._tts_ready = None
            if _confirm_task:
                _confirm_task.cancel()
            if hasattr(self.brain, '_confirm_request'):
                self.brain._confirm_request = None
            # -- 清理打断监听 ----------------------------------------------
            if watcher:
                watcher.cancel()
                try:
                    await watcher
                except asyncio.CancelledError:
                    pass
            if _monitoring_started:
                self.ears.stop_interrupt_monitoring()

            self._is_processing = False
            self.status_label.setText("状态: 聆听中..." if self.mic_btn.isChecked() else "状态: 待机")
            self.live2d_viewer.set_talking(False)
            self._set_ui_busy(False)
            self.input_field.setFocus()

    # -- 麦克风切换 ----------------------------------------
    def on_mic_toggled(self, checked: bool):
        if checked:
            if not self.ears:
                self.mic_btn.setChecked(False)
                return
            # 每次开麦生成新的 generation ID，旧线程结果会被丢弃
            self._listener_gen += 1
            self.ears._stop_listen.clear()
            self.mic_btn.setText("🔴 关闭麦克风")
            self.status_label.setText("状态: 聆听中...")
            self._listen_task = asyncio.ensure_future(self._background_listener(self._listener_gen))
        else:
            self.mic_btn.setText("🎙 开启麦克风")
            if not self._is_processing:
                self.status_label.setText("状态: 待机")
            # else: 模型正在思考/播放中，保留当前状态（如"思考中..."）
            if self.ears:
                self.ears._stop_listen.set()
                # 停止打断监听流（_process_input 有独立的 watcher）
                if self.ears.monitoring_stream is not None:
                    self.ears.stop_interrupt_monitoring()
                    self.ears.interrupt_detected = False  # 重置残留检测
            # ★ 关键：处理中不取消 _listen_task，否则 CancelledError 会
            #    沿 await 链级联到 _process_input -> 杀死 brain 生成线程
            if not self._is_processing:
                if self._listen_task and not self._listen_task.done():
                    self._listen_task.cancel()

    async def _background_listener(self, my_gen: int):
        '''在 executor 中跑阻塞式 listen_auto，避免阻塞 Qt 事件循环
        
        _mic_in_use 标记麦克风流是否被本线程持有，供 _process_input 判断
        是否可以安全开启独立的打断监听流（两流共享同一设备存在风险）
        
        当 _is_processing=True 时用户发话：
          - voice 触发路径：listen_auto 已返回、mic 已释放，_process_input
            会自行开启 start_interrupt_monitoring()，此处不需额外处理
          - text 触发路径：listen_auto 仍在运行，mic 被占用，_process_input
            无法开监听流本函数检测到声音后直接触发 bus.interrupt()
        '''
        loop = asyncio.get_event_loop()
        while True:
            if my_gen != self._listener_gen:
                break
            try:
                self._mic_in_use = True
                text = await loop.run_in_executor(None, self.ears.listen_auto)
                self._mic_in_use = False

                if my_gen != self._listener_gen:
                    break

                if text and text != "exit":
                    if not self._is_processing: #打断条件1
                        await self._process_input(text)
                        
                    elif self.bus and not self.bus.is_interrupted: #bus中且不在1中
                        logger.info("[GUI] 后台监听检测到用户发话，触发打断!")
                        await self.bus.interrupt()
                        
                    if my_gen == self._listener_gen:
                        self.status_label.setText("状态: 聆听中...")
                elif text == "exit":
                    break
                else:
                    await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                self._mic_in_use = False
                break
            except Exception as e:
                self._mic_in_use = False
                self._log(f"[监听错误] {e}")
                await asyncio.sleep(1.0)

    # -- PTT 热键监听 --------------------------------------
    def _setup_ptt_hotkey(self):
        '''启动后台线程监听全局热键（委托给 PttHandler）'''
        self.ptt_handler.start()

    def _stop_ptt_hotkey(self):
        '''停止热键监听（委托给 PttHandler）'''
        self.ptt_handler.stop()

    # -- PTT ----------------------------------------------
    @asyncSlot()
    async def on_ptt_pressed(self):
        '''按住 PTT（按钮或热键）时启动录音'''
        if not self.ears or self._ptt_task is not None:
            return  # 防重入

        # 若正在播放，先打断
        if self.bus and self._is_processing and not self.bus.is_interrupted:
            await self.bus.interrupt()

        self.status_label.setText("状态: 录音中...")
        self.ptt_btn.setText("⏹ 松开结束")
        self.ptt_btn.setStyleSheet(
            "QPushButton { background-color: #c42b1c; color: white;"
            " border: 1px solid #c42b1c; border-radius: 4px; font-weight: bold; }"
        )

        self.ears._stop_listen.clear()
        loop = asyncio.get_event_loop()
        self._ptt_task = loop.run_in_executor(None, self.ears.ptt_listen)

    @asyncSlot()
    async def on_ptt_released(self):
        '''松开 PTT 时结束录音，处理结果'''
        # 恢复按钮外观
        self.ptt_btn.setText("⏺ 按住说话")
        self.ptt_btn.setStyleSheet("")

        if not self.ears:
            return

        # 发信号让 ptt_listen 结束录音
        self.ears._stop_listen.set()

        if self._ptt_task is None:
            self.status_label.setText("状态: 待机")
            return

        try:
            text = await self._ptt_task
            self._ptt_task = None

            if text and not self._is_processing:
                self._log_user(f"[PTT] {text}")
                await self._process_input(text)
            elif not text:
                self.status_label.setText("状态: 无有效语音")
                await asyncio.sleep(0.8)
        except Exception as e:
            self._log(f"[PTT 错误] {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._ptt_task = None
            self.status_label.setText("状态: 待机" if not self.mic_btn.isChecked() else "状态: 聆听中...")

    # -- Live2D 控制 ----------------------------------------
    def _update_window_size(self):
        '''根据 Live2D 面板显隐状态自动调整窗口宽度'''
        if self.live2d_viewer.isVisible():
            self.resize(1020, self.height())
        else:
            self.resize(750, self.height())

    def toggle_live2d(self):
        '''显示/隐藏 Live2D 小人，并把选择保存到 config'''
        import yaml
        visible = not self.live2d_viewer.isVisible()
        if visible:
            self.live2d_viewer.show()
            # 保证 WebGL 已渲染（首次打开触发加载，再次打开触发 resize）
            self.live2d_viewer.ensure_rendered()
        else:
            self.live2d_viewer.hide()
        self._update_window_size()

        # 将选择写回 config.yaml
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            data.setdefault('ui', {})['live2d_visible'] = visible
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        except Exception as e:
            self._log(f"[Live2D] 保存显示状态失败: {e}")

    # -- 设置 ---------------------------------------------
    def open_settings(self):
        vts_mgr = (self.components or {}).get("vts")
        dlg = SettingsDialog(self, vts_manager=vts_mgr)
        if dlg.exec() == SettingsDialog.DialogCode.Accepted:
            self._log(">>> [System] 设置已保存")
            self._apply_live_config()
            # 处理 VTS 连接/断开请求
            if dlg.vts_requested_action == "connect":
                asyncio.ensure_future(self._do_vts_connect())
            elif dlg.vts_requested_action == "disconnect":
                asyncio.ensure_future(self._do_vts_disconnect())

    async def _do_vts_connect(self):
        '''异步连接 VTube Studio（由用户点击"同步VTS"触发）
        若 VTS 未运行则先自动启动 exe，等待就绪后再连接
        '''
        vts = (self.components or {}).get("vts")
        if vts is None:
            self._log(">>> [VTS] 组件尚未初始化，无法连接")
            return

        # 1. 尝试启动 VTS exe（已在运行时静默跳过）
        from core.launcher import launch_vts_app
        launched = launch_vts_app()
        if launched:
            self._log(">>> [VTS] 正在等待 VTube Studio 启动（约 10 秒）...")
            await asyncio.sleep(10)

        # 2. 尝试连接
        self._log(">>> [VTS] 正在连接 VTube Studio...")
        try:
            success = await vts.connect()
            if success:
                self._log(">>> [VTS] ✓ 连接成功，表情/动作同步已启用")
            else:
                self._log(">>> [VTS] ✗ 连接失败，请在 VTube Studio 中允许插件授权")
        except Exception as e:
            self._log(f">>> [VTS] ✗ 连接出错: {e}")

    async def _do_vts_disconnect(self):
        '''异步断开 VTube Studio 连接'''
        vts = (self.components or {}).get("vts")
        if vts is None:
            return
        await vts.close()
        self._log(">>> [VTS] 已断开 VTube Studio 连接")

    def _apply_live_config(self):
        '''将可即时生效的配置项应用到已运行的组件'''
        # -- 输入模式 -------------------------------------
        new_mode = get_config("audio.input_mode", "typing")
        is_mic_on = self.mic_btn.isChecked()
        if new_mode == "mic_push_to_talk":
            self.ptt_btn.setVisible(True)
            self._setup_ptt_hotkey()  # 热重载：可能改了热键
            if is_mic_on:
                self.mic_btn.setChecked(False)
                self.on_mic_toggled(False)
        elif new_mode == "mic_always_on":
            self.ptt_btn.setVisible(False)
            if not is_mic_on:
                self.mic_btn.setChecked(True)
                self.on_mic_toggled(True)
        else:
            self.ptt_btn.setVisible(False)
            if is_mic_on:
                self.mic_btn.setChecked(False)
                self.on_mic_toggled(False)

        # -- ears VAD 参数 ----
        if self.ears is not None:
            self.ears.vad_threshold           = get_config("ears.vad_threshold", 0.007)
            self.ears.interrupt_vad_threshold = get_config("ears.interrupt_vad_threshold", 0.015)
            self.ears.silence_limit           = get_config("ears.silence_limit", 0.5)
            self.ears.pre_buffer_len          = get_config("ears.pre_buffer_len", 4)

        # -- Live2D 模型路径变更 --------------------------
        new_model = get_config("ui.live2d_model", "")
        if new_model and new_model != self.live2d_viewer._current_model_path:
            if self.live2d_viewer.isVisible():
                self._log(">>> [Live2D] 模型路径已更新，重新加载...")
                self.live2d_viewer.load_model(new_model)
            else:
                # 未显示时重置加载标志，下次打开会自动加载新模型
                self.live2d_viewer._model_loaded = False
                self.live2d_viewer._current_model_path = None

    # -- 关闭处理 ------------------------------------------
    def closeEvent(self, event):
        if self._is_shutting_down:
            # 卸载边缘缩放过滤器，防止野指针
            if self._resize_filter is not None:
                from PyQt6.QtWidgets import QApplication
                QApplication.instance().removeNativeEventFilter(self._resize_filter)
                self._resize_filter = None
            event.accept()
            return

        self._is_shutting_down = True
        self._stop_ptt_hotkey()
        event.ignore()  # 先拦截，等后台清理完

        # 关闭前禁用所有控件
        self.send_btn.setEnabled(False)
        self.input_field.setEnabled(False)
        self.mic_btn.setEnabled(False)
        self.ptt_btn.setEnabled(False)
        self.settings_btn.setEnabled(False)

        # 清理正在进行中的任务
        if self.ears:
            self.ears._stop_listen.set()

        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()

        if self._ptt_task and not self._ptt_task.done():
            self._ptt_task.cancel()

        if self._po_task and not self._po_task.done():
            self._po_task.cancel()

        self.status_label.setText("状态: 正在关闭...")
        self._log(">>> [System] 正在安全关闭，请稍候...")

        async def _do_shutdown():
            try:
                await shutdown_monika(self.components, log_callback=self._log)
            except Exception as e:
                logger.warning("[GUI] 关闭时报错: %s", e)
            finally:
                # _is_shutting_down 保持 True，下次 closeEvent 会走 accept() 分支
                self.close()

        def _after_live2d(result):
            # 将最新变换写入 layout_mgr 缓存，再一并持久化
            if result and isinstance(result, dict):
                try:
                    data = {
                        "scale": round(float(result.get("scale", 1.0)), 4),
                        "x":     round(float(result.get("x", 0)), 2),
                        "y":     round(float(result.get("y", 0)), 2),
                    }
                    if self.isMaximized():
                        self.layout_mgr.live2d_maximized = data
                    else:
                        self.layout_mgr.live2d_normal = data
                    # 向后兼容：同时写入扁平键
                    from utils.config_loader import set_config as _sc
                    _sc("ui.live2d_scale", data["scale"])
                    _sc("ui.live2d_x",     data["x"])
                    _sc("ui.live2d_y",     data["y"])
                except Exception:
                    pass
            self._save_window_state()
            asyncio.ensure_future(_do_shutdown())

        if self.live2d_viewer.isVisible() and self.live2d_viewer._model_loaded:
            self.live2d_viewer.page().runJavaScript(
                "window.getModelTransform ? window.getModelTransform() : null",
                _after_live2d,
            )
        else:
            self._save_window_state()
            asyncio.ensure_future(_do_shutdown())
