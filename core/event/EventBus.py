'''
轻量事件总线 — 组件间 pub/sub 通信，解耦回调地狱。

用法：
    bus = EventBus()
    bus.subscribe("sentence_ready", lambda sentence: ...)
    bus.emit("sentence_ready", sentence="你好")

订阅者异常会被吞掉（记录日志），不影响其他订阅者。
'''

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any, Callable, Dict, List

log = logging.getLogger(__name__)


class EventBus:
    '''线程安全的事件总线（同步派发）'''

    def __init__(self):
        self._subs: Dict[str, List[Callable[..., None]]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, event: str, handler: Callable[..., None]) -> None:
        '''订阅事件。handler 接收 emit 时传入的关键字参数。'''
        with self._lock:
            if handler not in self._subs[event]:
                self._subs[event].append(handler)

    def unsubscribe(self, event: str, handler: Callable[..., None]) -> None:
        '''取消订阅。'''
        with self._lock:
            if handler in self._subs[event]:
                self._subs[event].remove(handler)

    def emit(self, event: str, **payload: Any) -> None:
        '''发布事件，同步调用所有订阅者。'''
        handlers = list(self._subs.get(event, []))
        for h in handlers:
            try:
                h(**payload)
            except Exception as e:
                log.warning("[EventBus] 订阅者处理 '%s' 异常: %s", event, e)

    def listener_count(self, event: str) -> int:
        '''某事件的订阅者数量（调试用）。'''
        return len(self._subs.get(event, []))


# 全局单例，供跨模块通信使用
_default_bus: "EventBus | None" = None


def get_event_bus() -> EventBus:
    '''获取全局 EventBus 单例。'''
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
    return _default_bus
