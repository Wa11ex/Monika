'''
PO（主动发言）状态机
管理冷却窗口，控制 PO 时间间隔
'''

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class PoWindow:
    '''冷却窗口，单位秒，在这之间随机'''
    lo: float # 最小值
    hi: float # 最大值


class PoStateMachine:
    '''PO 状态机

    用户每次说完话就 reset()，然后按顺序走冷却窗口。
    窗口用完之前每轮 next_timeout() 给一个随机倒计时，
    时间到了还没人说话就触发主动发言。
    '''

    def __init__(self, windows: List[Tuple[float, float]]):
        self._windows = [PoWindow(lo, hi) for lo, hi in windows]
        self._index = 0

    @classmethod
    def from_config(cls, cfg: dict) -> "PoStateMachine":
        '''从 proactive_output 配置节创建实例'''
        enabled = cfg.get("enabled", True) if isinstance(cfg, dict) else True
        raw = cfg.get("cooling_windows", [[240, 360], [360, 600]]) if isinstance(cfg, dict) else [[240, 360], [360, 600]]
        windows = [(float(w[0]), float(w[1])) for w in raw if len(w) == 2]
        return cls(windows) if enabled and windows else cls([])

    def reset(self) -> None:
        '''回到第一个冷却窗口（开口时调用）'''
        self._index = 0

    def next_timeout(self) -> Optional[float]:
        '''当前窗口内随机取一个超时秒数，窗口用完返回 None'''
        if self._index >= len(self._windows):
            return None
        w = self._windows[self._index]
        return random.uniform(w.lo, w.hi)

    def advance(self) -> None:
        '''切到下一个冷却窗口'''
        self._index += 1

    @property
    def is_exhausted(self) -> bool:
        '''窗口是不是全用完了'''
        return self._index >= len(self._windows)

    @property
    def current_index(self) -> int:
        return self._index

    @property
    def total_windows(self) -> int:
        return len(self._windows)
