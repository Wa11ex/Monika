'''
实时视觉监视窗口：显示摄像头/屏幕的实时画面 + fps/分辨率/单帧耗时。
用于调试验证帧率与画质，不做推理（推理仍是事件驱动单帧）。

VisionMonitor   : 单源监视（摄像头或屏幕）
DualVisionMonitor: 摄像头 + 屏幕 并排监视
'''
from __future__ import annotations

import time

from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap


class _VisionPanel(QWidget):
    '''单路实时画面面板（图像 + 统计 + 定时抓帧）'''

    def __init__(self, source, label: str, target_fps: float = 15.0, parent=None):
        super().__init__(parent)
        self._source = source          # 具有 capture() -> VisionFrame 的源
        self._label = label
        self._target_fps = float(target_fps)
        self._frames = 0
        self._start = time.perf_counter()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._stats = QLabel("启动中...")
        self._stats.setStyleSheet("color: #aaaacc; font-size: 9pt; background: transparent;")
        layout.addWidget(self._stats)

        self._img = QLabel()
        self._img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img.setMinimumSize(240, 180)
        self._img.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3e3e42;")
        layout.addWidget(self._img, 1)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def start(self):
        interval = max(int(1000.0 / self._target_fps), 16)
        self._timer.start(interval)

    def stop(self):
        self._timer.stop()

    def _tick(self):
        t0 = time.perf_counter()
        try:
            frame = self._source.capture()
        except Exception as e:
            self._stats.setText(f"捕获失败: {e}")
            return
        if frame is None:
            self._stats.setText("无帧")
            return

        latency_ms = (time.perf_counter() - t0) * 1000
        self._frames += 1
        elapsed = time.perf_counter() - self._start
        fps = self._frames / elapsed if elapsed > 0 else 0.0

        pix = QPixmap()
        pix.loadFromData(frame.jpeg_bytes, "JPEG")
        self._img.setPixmap(pix.scaled(
            self._img.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

        self._stats.setText(
            f"{frame.label}  {frame.width}x{frame.height}  "
            f"实际 {fps:.1f} fps（目标 {self._target_fps:.0f}）  单帧 {latency_ms:.0f} ms"
        )


class VisionMonitor(QDialog):
    '''单源监视（摄像头或屏幕）'''

    def __init__(self, source, title: str = "Vision", target_fps: float = 15.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(480, 400)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self._panel = _VisionPanel(source, title, target_fps, self)
        layout.addWidget(self._panel)

    def start(self):
        self._panel.start()

    def closeEvent(self, event):
        self._panel.stop()
        super().closeEvent(event)


class DualVisionMonitor(QDialog):
    '''摄像头 + 屏幕 并排监视'''

    def __init__(self, cam_source, scr_source,
                 cam_fps: float = 15.0, scr_fps: float = 5.0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("双流监视")
        self.resize(960, 420)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self._cam = _VisionPanel(cam_source, "摄像头", cam_fps, self)
        self._scr = _VisionPanel(scr_source, "屏幕", scr_fps, self)
        layout.addWidget(self._cam, 1)
        layout.addWidget(self._scr, 1)

    def start(self):
        self._cam.start()
        self._scr.start()

    def closeEvent(self, event):
        self._cam.stop()
        self._scr.stop()
        super().closeEvent(event)
