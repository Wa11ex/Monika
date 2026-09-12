'''
启动加载窗口 — 显示初始化进度和日志

在主窗口构造前显示，init_monika 完成后关闭并切换到主窗口
'''

from __future__ import annotations

import asyncio
import logging

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QProgressBar,
    QPlainTextEdit, QPushButton,
)
from PyQt6.QtCore import Qt, QTimer
from qasync import asyncSlot

from ui.theme import COLORS


logger = logging.getLogger(__name__)


class SplashWindow(QDialog):
    '''启动加载窗口：进度条 + 日志列表 + 深色风格'''

    # 加载阶段定义（label, weight）
    _STAGES = [
        ("TTS 服务", 20),
        ("听觉系统", 15),
        ("语言模型", 25),
        ("记忆系统", 20),
        ("工具注册", 5),
        ("收尾", 15),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Monika — 正在唤醒...")
        self.setFixedSize(420, 320)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint
        )
        self.setStyleSheet(self._stylesheet())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 16)
        layout.setSpacing(12)

        # 标题
        title = QLabel("正在唤醒 Monika...")
        title.setStyleSheet(
            f"color: {COLORS['text_primary']}; font-size: 14pt; font-weight: bold; background: transparent;"
        )
        layout.addWidget(title)

        # 副标题（当前阶段）
        self._stage_label = QLabel("准备中...")
        self._stage_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 9pt; background: transparent;"
        )
        layout.addWidget(self._stage_label)

        # 进度条
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFixedHeight(8)
        self._progress.setStyleSheet(self._progress_style())
        layout.addWidget(self._progress)

        # 日志区域
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(200)
        self._log_view.setStyleSheet(
            f"QPlainTextEdit {{"
            f"  background-color: {COLORS['bg_primary']};"
            f"  color: {COLORS['text_secondary']};"
            f"  font-size: 8pt; font-family: Consolas, monospace;"
            f"  border: 1px solid {COLORS['border']}; border-radius: 4px;"
            f"  padding: 4px;"
            f"}}"
        )
        layout.addWidget(self._log_view, 1)

        self._current_progress = 0
        self._components = None
        self._error = None
        self._on_complete = None  # 回调：初始化完成后调用

    def set_stage(self, name: str, weight: int = 0) -> None:
        '''更新当前加载阶段'''
        self._stage_label.setText(name)
        if weight > 0:
            target = min(self._current_progress + weight, 100)
            # 平滑动画
            self._animate_to(target)
        self._log(f"[{name}]")

    def _animate_to(self, target: int) -> None:
        '''进度条平滑增长'''
        self._current_progress = target
        self._progress.setValue(target)

    def _log(self, msg: str) -> None:
        '''追加一行日志'''
        self._log_view.appendPlainText(msg)

    def on_log_callback(self, msg: str) -> None:
        '''作为 init_monika 的 log_callback'''
        self._log(msg)

    @asyncSlot()
    async def start_init(self) -> None:
        '''启动后台初始化'''
        from main import init_monika

        def _on_progress(stage: str, pct: int):
            self._stage_label.setText(stage)
            self._progress.setValue(pct)

        try:
            self._components = await init_monika(
                log_callback=self.on_log_callback,
                progress_callback=_on_progress,
            )
            self._progress.setValue(100)
            self._stage_label.setText("初始化完成！")
            # 短暂展示后切换
            QTimer.singleShot(500, self._finish)
        except Exception as e:
            self._error = e
            self._log(f"[错误] 初始化失败: {e}")
            self._stage_label.setText("初始化失败！")
            self._stage_label.setStyleSheet(
                f"color: #e05555; font-size: 9pt; background: transparent;"
            )

    def _finish(self) -> None:
        '''初始化完成，触发回调（不自动关闭，等主窗口 ready 后由外部关闭）'''
        if self._on_complete:
            self._on_complete(self._components, self._error)
        # 不在这里关闭 splash — 由 gui.py 的 _on_init_done 创建主窗口后关闭

    def _stylesheet(self) -> str:
        return f"""
        QDialog {{
            background-color: {COLORS['bg_secondary']};
        }}
        QLabel {{
            background: transparent;
        }}
        """

    def _progress_style(self) -> str:
        return f"""
        QProgressBar {{
            background-color: {COLORS['bg_primary']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            text-align: center;
            color: transparent;
            font-size: 1pt;
        }}
        QProgressBar::chunk {{
            background-color: {COLORS['accent']};
            border-radius: 3px;
        }}
        """
