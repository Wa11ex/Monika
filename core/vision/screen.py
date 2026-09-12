'''
屏幕捕获源（mss + opencv 缩放）
抓主显示器，等比缩到 max_width，JPEG 编码。
'''
from __future__ import annotations

from typing import Optional

from . import VisionFrame
from utils.logger import get_logger

log = get_logger(__name__)


class ScreenSource:
    def __init__(self, max_width: int = 1280):
        self.max_width = max_width

    def capture(self, jpeg_quality: int = 90) -> Optional[VisionFrame]:
        '''抓一帧主显示器，缩放后 JPEG 编码；失败返回 None'''
        try:
            import cv2
            import mss
            import numpy as np

            with mss.MSS() as sct:
                shot = sct.grab(sct.monitors[1])  # 主显示器

            # mss 10.x: bgra 是 4 通道 BGRA（rgb 是 3 通道 RGB）
            frame_bgra = np.frombuffer(
                shot.bgra, dtype=np.uint8
            ).reshape(shot.height, shot.width, 4).copy()
            frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

            h, w = frame_bgr.shape[:2]
            if w > self.max_width:
                scale = self.max_width / w
                new_h = max(1, int(h * scale))
                frame_bgr = cv2.resize(
                    frame_bgr, (self.max_width, new_h),
                    interpolation=cv2.INTER_AREA,
                )
                h, w = new_h, self.max_width

            ok, buf = cv2.imencode(
                ".jpg", frame_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
            )
            if not ok:
                return None

            return VisionFrame(
                source="screen",
                jpeg_bytes=buf.tobytes(),
                width=w,
                height=h,
                label="屏幕截图",
            )
        except Exception as e:
            log.warning("[Vision] 屏幕捕获失败: %s", e)
            return None
