'''
Monika-V3 视觉捕获层（package）

事件驱动：只在需要时抓单帧，不做连续视频流。
统一输出 JPEG 字节（VisionFrame），直接对接 VLM（data URL）。

外挂视觉编码器（encoder）：主脑不支持视觉（supports_vision=False）时的降级路线，
用小型 VLM/OCR 把图像转成文本后再注入对话。
'''
from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import List, Optional

from utils.config_loader import get_config
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class VisionFrame:
    '''一帧视觉快照'''
    source: str                # "camera" | "screen"
    jpeg_bytes: bytes          # JPEG 编码字节
    width: int
    height: int
    captured_at: float = field(default_factory=time.time)
    label: str = ""            # 人类可读描述，如 "屏幕截图" / "摄像头画面"

    def to_data_url(self) -> str:
        b64 = base64.b64encode(self.jpeg_bytes).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"


# -- 开关（运行时读取，支持热重载） --------------------------------------------

def camera_enabled() -> bool:
    return bool(get_config("multimodal.vision.enabled", False))


def screen_enabled() -> bool:
    return bool(get_config("multimodal.screen.enabled", False))


# -- 单例源 -------------------------------------------------------------------

_camera = None
_screen = None


def get_camera():
    '''摄像头单例（惰性初始化）'''
    global _camera
    if _camera is None:
        from .camera import CameraSource
        _camera = CameraSource(
            device=int(get_config("multimodal.vision.capture_device", 0)),
            width=int(get_config("multimodal.vision.width", 1280)),
            height=int(get_config("multimodal.vision.height", 720)),
        )
    return _camera


def get_screen():
    '''屏幕单例（惰性初始化）'''
    global _screen
    if _screen is None:
        from .screen import ScreenSource
        _screen = ScreenSource(
            max_width=int(get_config("multimodal.screen.max_width", 1280)),
        )
    return _screen


def capture_from_config() -> List[VisionFrame]:
    '''按当前开关抓取本轮要附带的帧（空列表 = 不附带）'''
    frames: List[VisionFrame] = []
    if camera_enabled():
        f = get_camera().capture()
        if f is not None:
            frames.append(f)
    if screen_enabled():
        f = get_screen().capture()
        if f is not None:
            frames.append(f)
    return frames


# -- 视觉编码器（外挂 VLM/OCR，主脑不支持视觉时的降级路线） -------------------

_encoder = None


def get_vision_encoder():
    '''返回外挂视觉编码器；未配置模型时返回 NullEncoder

    配置键（需在 config.yaml 的 multimodal.vision 下补齐）：
      model_path    : 视觉 VLM 的文本 GGUF 路径
      mmproj_path   : 配对的 mmproj 投影器路径
      family        : 模型家族（默认 moondream；见 encoder._VISION_HANDLERS）
      encoder_prompt: 编码提示词（默认 "请描述这张图片。"）
    '''
    global _encoder
    if _encoder is None:
        from utils.helpers import resolve_path
        from .encoder import ExternalVisionEncoder, NullEncoder
        model_path = get_config("multimodal.vision.model_path", "")
        mmproj_path = get_config("multimodal.vision.mmproj_path", "")
        family = get_config("multimodal.vision.family", "moondream")
        prompt = get_config("multimodal.vision.encoder_prompt", "请描述这张图片。")
        if model_path:
            model_path = resolve_path(model_path)
        if mmproj_path:
            mmproj_path = resolve_path(mmproj_path)
        if model_path:
            _encoder = ExternalVisionEncoder(
                model_path=model_path,
                mmproj_path=mmproj_path,
                family=family,
                prompt=prompt,
                n_ctx=int(get_config("multimodal.vision.encoder_n_ctx", 2048)),
                n_gpu_layers=int(get_config("multimodal.vision.encoder_n_gpu_layers", -1)),
            )
        else:
            _encoder = NullEncoder()
    return _encoder
