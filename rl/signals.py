"""
隐式反馈信号定义 — 不要求用户手动标注，从行为模式推断满意度

设计原则：
  - 信号权重独立于 RewardModel，是可人工调节的 baseline
  - RewardModel 学到的是对这个 baseline 的修正
  - 所有信号在 InteractionLog 中留下原始标记（不合并），方便后续重新评估
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Any, Dict, List, Optional
import uuid
import time


class SignalType(Enum):
    """隐式反馈信号类型"""

    # 强正向
    USER_CONTINUED = "user_continued"     # 用户继续说下一轮
    EXPLICIT_UP = "explicit_up"           # 点赞

    # 弱正向
    LONG_TURN = "long_turn"               # 对话轮次 > 10
    TOOL_SUCCEEDED = "tool_succeeded"     # 工具调用成功

    # 强负向
    INTERRUPTED = "interrupted"           # 用户打断
    EXPLICIT_DOWN = "explicit_down"       # 踩

    # 弱负向
    SILENCE_TIMEOUT = "silence_timeout"   # PO 触发（用户沉默）
    SHORT_SESSION = "short_session"       # 对话不足 3 轮就退出
    TOOL_FAILED = "tool_failed"           # 工具调用失败


SIGNAL_WEIGHTS: Dict[SignalType, float] = {
    SignalType.USER_CONTINUED:   0.15,
    SignalType.EXPLICIT_UP:      0.25,
    SignalType.LONG_TURN:        0.05,
    SignalType.TOOL_SUCCEEDED:   0.05,
    SignalType.INTERRUPTED:     -0.25,
    SignalType.EXPLICIT_DOWN:   -0.25,
    SignalType.SILENCE_TIMEOUT: -0.10,
    SignalType.SHORT_SESSION:   -0.15,
    SignalType.TOOL_FAILED:     -0.05,
}


@dataclass
class InteractionLog:
    """单轮对话的完整记录 — 离线训练的基本单元"""

    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)

    # ---------------------------------------------------------------
    # 输入侧
    # ---------------------------------------------------------------
    user_text: str = ""
    user_emotion: Optional[str] = None   # SenseVoiceSmall 输出的情感标签
    speaker_name: str = ""

    # ---------------------------------------------------------------
    # 系统状态 (RL 的 observation)
    # ---------------------------------------------------------------
    trait_snapshot: Dict[str, float] = field(default_factory=lambda: {
        "mood": 0.5,
        "loneliness": 0.0,
        "curiosity": 0.5,
        "affection": 0.5,
    })

    # ---------------------------------------------------------------
    # 输出侧
    # ---------------------------------------------------------------
    assistant_text: str = ""
    tool_calls: int = 0
    tool_success: int = 0
    tool_fail: int = 0
    total_duration_ms: float = 0.0

    # ---------------------------------------------------------------
    # 信号与标签
    # ---------------------------------------------------------------
    signals: List[str] = field(default_factory=list)   # SignalType 名称列表
    interrupted: bool = False
    composite_reward: float = 0.5   # 由 compute_reward 计算，[0, 1]
    labeled_as: Optional[str] = None   # "good" | "bad" | None (训练时派生)


def compute_reward(log: InteractionLog) -> float:
    """从信号列表中计算综合奖励分数 [0, 1]

    baseline = 0.5 (中性)，正向信号加分、负向信号扣分，clamp 到 [0, 1]
    """
    reward = 0.5
    for sig_name in log.signals:
        try:
            sig = SignalType[sig_name]
            reward += SIGNAL_WEIGHTS.get(sig, 0.0)
        except KeyError:
            pass
    return max(0.0, min(1.0, reward))
