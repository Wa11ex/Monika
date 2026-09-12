'''
流式 token 输出缓冲，供调试面板实时显示每个模型的原始输出。

原理是模型侧（任意线程）调用 emit() 追加；UI 侧用 QTimer 轮询 drain()
取出待显示片段。纯线程安全，不依赖 Qt。
'''

from __future__ import annotations

import queue
from typing import List

_buffer: "queue.Queue[str]" = queue.Queue()


def emit(text: str) -> None:
    '''模型输出 token 片段时调用（线程安全）'''
    if text:
        _buffer.put(text)


def drain(max_items: int = 2000) -> List[str]:
    '''取出当前缓冲里所有待显示的片段（非阻塞）'''
    items: List[str] = []
    try:
        while len(items) < max_items:
            items.append(_buffer.get_nowait())
    except queue.Empty:
        pass
    return items
