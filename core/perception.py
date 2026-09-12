'''
Monika-V3 Perception 层（感知）
这个主要是为了封装不同的输入，然后给一个统一的 get_next_event() 接口，返回一个事件dict
'''

import asyncio
import time
from typing import Optional

from core.interfaces import BasePerception
from core.ears import SILENCE_TIMEOUT
from core.event import PoStateMachine
from utils.logger import get_logger

log = get_logger(__name__)


class MicPerception(BasePerception):
    '''
    麦克风，常开或ptt
    '''

    def __init__(self, ears, po: Optional[PoStateMachine] = None):
        '''
        输入参考：
        Args:
            ears: Ears 实例
            po: PO 状态机（None 时禁用主动发言）
        '''
        self._ears = ears
        self._po = po

    async def get_next_event(self) -> dict:
        '''
        mic的get_next_event
        '''
        # 计算本轮 PO 超时时间
        po_timeout: Optional[float] = None
        if self._po is not None:
            po_timeout = self._po.next_timeout()

        raw = await asyncio.to_thread(self._ears.listen_auto, po_timeout)

        now = time.time()

        if raw == "exit":
            return {"type": "exit", "data": "", "source": "ears", "timestamp": now}

        if raw == SILENCE_TIMEOUT:
            if self._po is not None:
                self._po.advance()
            po_count = self._po.current_index if self._po else 0
            return {
                "type": "silence",
                "data": "",
                "source": "ears",
                "timestamp": now,
                "po_count": po_count,
            }

        # 若成功输入，重置 PO 状态机
        if self._po is not None:
            self._po.reset()
        return {"type": "text", "data": raw or "", "source": "ears", "timestamp": now}

    # def reset_po_count(self):
    #     '''重置 PO 触发计数器'''
    #     if self._po is not None:
    #         self._po.reset()


class KeyboardPerception(BasePerception):
    '''键盘/控制台输入'''

    async def get_next_event(self) -> dict:
        '''
        get_next_event 读一行
        '''
        print("\n>>> [User] 请在控制台输入对话 (输入 'exit' 退出):")
        text = await asyncio.to_thread(input, "> ")
        now = time.time()

        stripped = text.strip()
        if not stripped or stripped.lower() == "exit":
            if stripped.lower() == "exit":
                return {"type": "exit", "data": "", "source": "keyboard", "timestamp": now}
            # 空输入视为无效，返回空 text 事件，主 loop 会 continue
            return {"type": "text", "data": "", "source": "keyboard", "timestamp": now}

        return {"type": "text", "data": stripped, "source": "keyboard", "timestamp": now}
