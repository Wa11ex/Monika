'''
外挂视觉编码器：把 VisionFrame 转成文本描述 / OCR 文本。
主脑不支持视觉（supports_vision=False）时的降级路线。
'''
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Optional

import contextlib
import ctypes

from . import VisionFrame
from utils.logger import get_logger

log = get_logger(__name__)


@contextlib.contextmanager
def _suppress_c_output():
    '''彻底静默 llama.cpp 的 C 层 stdout/stderr

    suppress_stdout_stderr 用 dup2 重定向 fd，但 C 运行时 stdout 是缓冲的，
    缓冲内容会在进程退出时才冲刷，从而绕过重定向。这里在恢复 fd 之前
    强制 fflush(NULL)，把缓冲内容冲刷进（已指向 devnull 的）fd。
    '''
    from llama_cpp import suppress_stdout_stderr
    with suppress_stdout_stderr():
        try:
            yield
        finally:
            try:
                _msvcrt = ctypes.CDLL("msvcrt")
                _msvcrt.fflush.argtypes = [ctypes.c_void_p]
                _msvcrt.fflush.restype = ctypes.c_int
                _msvcrt.fflush(None)
            except Exception:
                pass


class BaseVisionEncoder(ABC):
    @abstractmethod
    async def encode(self, frame: VisionFrame) -> Optional[str]:
        '''把一帧图像编码为文本（描述/OCR），失败返回 None'''
        ...

    async def close(self):
        pass


class NullEncoder(BaseVisionEncoder):
    '''未配置外挂模型时的空实现'''
    async def encode(self, frame: VisionFrame) -> Optional[str]:
        log.debug("[Vision] 未配置外挂视觉模型，跳过编码")
        return None


# 支持的视觉模型 family -> llama_cpp 的 chat handler 类名
_VISION_HANDLERS = {
    "moondream":    "MoondreamChatHandler",
    "llava15":      "Llava15ChatHandler",
    "llava16":      "Llava16ChatHandler",
    "qwen2.5vl":    "Qwen25VLChatHandler",
    "minicpmv26":   "MiniCPMv26ChatHandler",
    "nanolava":     "NanoLlavaChatHandler",
    "obsidian":     "ObsidianChatHandler",
    "llama3vision": "Llama3VisionAlphaChatHandler",
}


class ExternalVisionEncoder(BaseVisionEncoder):
    '''用 llama-cpp-python 加载本地 VLM（文本 GGUF + mmproj），做 image -> text'''

    def __init__(self, model_path: str, mmproj_path: str = "",
                 family: str = "moondream",
                 prompt: str = "请详细描述这张图片。",
                 n_ctx: int = 2048, n_gpu_layers: int = -1):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.family = family
        self.prompt = prompt
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._llm = None

    def _ensure_model(self):
        if self._llm is None:
            from llama_cpp import Llama, suppress_stdout_stderr
            from llama_cpp import llama_chat_format as _lcf

            kwargs = dict(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )
            if self.mmproj_path:
                cls_name = _VISION_HANDLERS.get(self.family)
                if not cls_name:
                    raise ValueError(f"未知视觉模型 family: {self.family}")
                handler_cls = getattr(_lcf, cls_name)
                kwargs["chat_handler"] = handler_cls(
                    clip_model_path=self.mmproj_path, verbose=False)
                log.info("[Vision] 加载外挂视觉模型: %s (family=%s, mmproj=%s)",
                         self.model_path, self.family, self.mmproj_path)
            else:
                log.info("[Vision] 加载外挂视觉模型: %s", self.model_path)
            with _suppress_c_output():
                self._llm = Llama(**kwargs)
        return self._llm

    async def encode(self, frame: VisionFrame) -> Optional[str]:
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._encode_sync, frame)
        except Exception as e:
            log.warning("[Vision] 外挂编码失败: %s", e)
            return None

    def _encode_sync(self, frame: VisionFrame) -> Optional[str]:
        llm = self._ensure_model()
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": frame.to_data_url()}},
                {"type": "text", "text": self.prompt},
            ],
        }]
        with _suppress_c_output():
            resp = llm.create_chat_completion(
                messages=messages,
                max_tokens=256,
                temperature=0.2,
                stream=False,
            )
        choices = resp.get("choices") or []
        if not choices:
            return None
        msg = choices[0].get("message") or {}
        content = msg.get("content") or ""
        from core.event import emit as _emit_token
        _emit_token(f"\n<<< VLM [{frame.label}] >>>\n{content}\n")
        return content.strip() or None

    async def close(self):
        # 卸载 VLM 释放显存给主脑 LLM（串行原则）
        # llama-cpp-python 0.3.x 的 Llama 有显式 close()，直接调用释放底层
        # llama_context / llama_model；置空 + GC 仅作兜底。
        if self._llm is not None:
            try:
                self._llm.close()
            except Exception as e:
                log.warning("[Vision] 关闭视觉模型异常: %s", e)
            finally:
                self._llm = None
                import gc
                gc.collect()
            log.info("[Vision] 视觉模型已卸载")
