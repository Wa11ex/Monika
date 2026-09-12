'''
摄像头捕获源（opencv-python）
惰性打开，复用句柄，避免每次 capture 重新 open 的开销。
'''
from __future__ import annotations

from typing import Optional

from . import VisionFrame
from utils.logger import get_logger

log = get_logger(__name__)


class CameraSource:
    def __init__(self, device: int = 0, width: int = 1280, height: int = 720):
        self.device = device
        self.width = width
        self.height = height
        self._cap = None

    def _open(self):
        if self._cap is None:
            import cv2
            import time as _time
            _silence_cv2()
            # Windows 上优先用 DSHOW 后端（打开更快、更稳），失败回退默认后端
            cap = cv2.VideoCapture(self.device, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(self.device)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            # 预热：冷启动时自动曝光未稳定，前若干帧是黑帧。循环读帧直到
            # 画面亮度不再接近全黑（最多等 3 秒）。
            _deadline = _time.time() + 3.0
            while _time.time() < _deadline:
                ok, _f = cap.read()
                if ok and _f is not None and _f.mean() > 10:
                    break
            self._cap = cap
        return self._cap

    def capture(self, jpeg_quality: int = 90) -> Optional[VisionFrame]:
        '''抓一帧，JPEG 编码后返回；失败返回 None'''
        try:
            import cv2
            cap = self._open()
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                return None

            h, w = frame_bgr.shape[:2]
            # cv2.imencode 期望 BGR 输入，直接编码即可得到颜色正确的 JPEG
            # （之前先转 RGB 再编码，导致 R/B 通道互换 -> 人脸偏蓝）
            ok, buf = cv2.imencode(
                ".jpg", frame_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
            )
            if not ok:
                return None

            return VisionFrame(
                source="camera",
                jpeg_bytes=buf.tobytes(),
                width=w,
                height=h,
                label="摄像头画面",
            )
        except Exception as e:
            log.warning("[Vision] 摄像头捕获失败: %s", e)
            return None

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def _silence_cv2():
    '''关闭 OpenCV 自身的日志输出（避免 DSHOW 枚举时刷警告）'''
    import cv2
    try:
        cv2.setLogLevel(0)
    except Exception:
        pass


def list_cameras(max_devices: int = 6):
    '''枚举可用摄像头设备，返回 [(index, name), ...]'''
    import os
    import cv2
    _silence_cv2()
    result = []
    # OpenCV DSHOW 对不存在的索引会向 stderr 刷 C 层警告，这里静默掉
    _devnull = os.open(os.devnull, os.O_WRONLY)
    _old_err = os.dup(2)
    os.dup2(_devnull, 2)
    try:
        for idx in range(max_devices):
            cap = None
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    result.append((idx, f"摄像头 {idx}"))
            except Exception:
                pass
            finally:
                if cap is not None:
                    cap.release()
    finally:
        os.dup2(_old_err, 2)
        os.close(_old_err)
        os.close(_devnull)
    return result
