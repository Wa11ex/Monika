'''
Monika-V3 TTS 客户端
'''

import os
import asyncio
import aiohttp

from core.interfaces import BaseTTS
from utils.config_loader import get_config
from utils.helpers import resolve_path
from utils.logger import get_logger

log = get_logger(__name__)



class GPTSoVITSTTS(BaseTTS):
    '''流式 HTTP 接收'''

    def __init__(self):
        self._server_url = get_config("tts.server_url", "http://127.0.0.1:5000/tts_stream")
        self._sample_rate = get_config("tts.sample_rate", 24000)
        self._ref_audio_path = resolve_path(get_config("tts.ref_audio_path", ""))
        self._ref_text = get_config("tts.ref_text", "")
        self._if_sr = get_config("tts.if_sr", False)
        self._speed = get_config("tts.speed", 1.0)
        self._session: aiohttp.ClientSession | None = None
        self._cancelled = False

    def _get_session(self) -> aiohttp.ClientSession:
        '''懒加载并复用 aiohttp session'''
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def synthesize(self, text: str):
        '''调用 TTS 服务，失败是None'''
        self._cancelled = False
        payload = {
            "text": text,
            "text_lang": "zh",
            "ref_audio_path": self._ref_audio_path,
            "ref_text": self._ref_text,
            "ref_lang": "zh",
            "sample_rate": self._sample_rate,
            "speed": self._speed,
            "if_sr": self._if_sr,
        }
        audio_buffer = bytearray()
        try:
            session = self._get_session()
            async with session.post(
                self._server_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    log.warning("[TTS] 服务返回异常状态: %s", resp.status)
                    return None
                while True:
                    if self._cancelled:
                        return None
                    chunk = await resp.content.read(4096)
                    if not chunk:
                        break
                    audio_buffer.extend(chunk)
            return bytes(audio_buffer)
        except asyncio.CancelledError:
            return None
        except Exception as e:
            if not self._cancelled:
                log.error("[TTS] 合成异常: %s", e)
        return None

    async def cancel(self):
        '''取消时关闭HTTP请求'''
        self._cancelled = True
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        log.debug("[TTS] 请求已取消，session 已关闭")

    async def close(self):
        '''清理缓存'''
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
