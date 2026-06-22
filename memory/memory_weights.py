'''
记忆重要度评分与时间衰减 — 独立模块，方便后续编辑评分逻辑

设计原则：
  - 评分函数接受 (text, metadata) -> float (0.0~1.0)
  - 衰减函数接受 (importance, age_seconds) -> float (0.0~1.0)
  - 通过 config.yaml 控制参数，可在运行时调整
  - 所有启发式规则集中在 _heuristic_rules()，方便增删

Usage:
    weights = MemoryWeights()
    score = weights.score("用户说记住这个", {"has_tool_call": False})
    decayed = weights.decay(score, age_seconds=86400 * 7)
'''

from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional

from utils.config_loader import get_config
from utils.logger import get_logger

log = get_logger(__name__)

# -- 参数（默认 ----------------------------------------------

_DEFAULT_HALF_LIFE_DAYS = 30.0       # 记忆半衰期（天）
_DEFAULT_MIN_IMPORTANCE = 0.1        # 最低重要度
_DEFAULT_BOOST_KEYWORDS = [          # 提升权重的关键词
    "记住", "重要", "别忘了", "关键", "一定",
    "remember", "important", "don't forget",
]
_DEFAULT_TOOL_BOOST = 0.15           # 包含工具调用的对话额外加分
_DEFAULT_LONG_MSG_THRESHOLD = 80     # 长消息字符阈值
_DEFAULT_LONG_MSG_BOOST = 0.1        # 长消息额外加分
_DEFAULT_VERY_LONG_THRESHOLD = 500   # 过长消息惩罚阈值（char数）
_DEFAULT_VERY_LONG_PENALTY = -0.15   # 过长消息惩罚分
_DEFAULT_FEEDBACK_UP_BOOST = 0.2     # 用户点赞加分
_DEFAULT_FEEDBACK_DOWN_PENALTY = -0.2 # 用户踩惩罚


class MemoryWeights:
    '''记忆重要度评分与衰减
    外部调用：score() 和 decay()
    '''

    def __init__(
        self,
        half_life_days: Optional[float] = None,
        min_importance: Optional[float] = None,
        boost_keywords: Optional[list[str]] = None,
        tool_boost: Optional[float] = None,
        long_msg_threshold: Optional[int] = None,
        long_msg_boost: Optional[float] = None,
        very_long_threshold: Optional[int] = None,
        very_long_penalty: Optional[float] = None,
        feedback_up_boost: Optional[float] = None,
        feedback_down_penalty: Optional[float] = None,
    ):
        self.half_life_days = (
            half_life_days
            or get_config("memory.weights.half_life_days", _DEFAULT_HALF_LIFE_DAYS)
        )
        self.min_importance = (
            min_importance
            or get_config("memory.weights.min_importance", _DEFAULT_MIN_IMPORTANCE)
        )
        self.boost_keywords = boost_keywords or _DEFAULT_BOOST_KEYWORDS
        self.tool_boost = tool_boost or _DEFAULT_TOOL_BOOST
        self.long_msg_threshold = (
            long_msg_threshold or _DEFAULT_LONG_MSG_THRESHOLD
        )
        self.long_msg_boost = long_msg_boost or _DEFAULT_LONG_MSG_BOOST
        self.very_long_threshold = (
            very_long_threshold
            or get_config("memory.weights.very_long_threshold", _DEFAULT_VERY_LONG_THRESHOLD)
        )
        self.very_long_penalty = (
            very_long_penalty
            or get_config("memory.weights.very_long_penalty", _DEFAULT_VERY_LONG_PENALTY)
        )
        self.feedback_up_boost = (
            feedback_up_boost
            or get_config("memory.weights.feedback_up_boost", _DEFAULT_FEEDBACK_UP_BOOST)
        )
        self.feedback_down_penalty = (
            feedback_down_penalty
            or get_config("memory.weights.feedback_down_penalty", _DEFAULT_FEEDBACK_DOWN_PENALTY)
        )

    # -- 公开调用区 ----------------------------------------

    def score(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        '''计算记忆的重要度分数 (0.0~1.0)'''
        metadata = metadata or {}
        base = 0.5  # 起始中性分数
        base += self._heuristic_rules(text, metadata)
        return max(0.0, min(1.0, base))

    def decay(self, importance: float, age_seconds: float) -> float:
        '''对重要度应用时间衰减'''
        if age_seconds <= 0 or self.half_life_days <= 0:
            return importance

        half_life_seconds = self.half_life_days * 86400.0
        decay_factor = math.pow(0.5, age_seconds / half_life_seconds)
        decayed = importance * decay_factor
        
        return max(self.min_importance, decayed)

    def combined_score(
        self, text: str, age_seconds: float, metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        '''一步搞定'''
        imp = self.score(text, metadata)
        return self.decay(imp, age_seconds)


    # -- 启发式规则（集中管理，方便增删） -----------------

    def _heuristic_rules(self, text: str, metadata: Dict[str, Any]) -> float:
        '''所有启发式规则的累加分数
        自由增删
        '''
        delta = 0.0

        # 1. 关键词命中
        delta += self._keyword_boost(text)

        # 2. 包含工具调用
        if metadata.get("has_tool_call"):
            delta += self.tool_boost

        # 3. 中等长度消息
        if len(text) > self.long_msg_threshold:
            delta += self.long_msg_boost

        # 4. 过长消息
        if len(text) > self.very_long_threshold:
            delta += self.very_long_penalty

        # 5. 点赞/踩
        feedback = metadata.get("feedback")
        if feedback == "up":
            delta += self.feedback_up_boost
        elif feedback == "down":
            delta += self.feedback_down_penalty

        # 6. 用户主动说"记住"
        if self._is_explicit_remember(text):
            delta += 0.25

        return delta

    # -- 内部子函数 ---------------------------------------

    def _keyword_boost(self, text: str) -> float:
        '''关键词命中加分'''
        hits = sum(1 for kw in self.boost_keywords if kw.lower() in text.lower())
        return min(0.2, hits * 0.05)

    @staticmethod
    def _is_explicit_remember(text: str) -> bool:
        '''记住关键词比对'''
        patterns = [
            r"记住[这此那]",
            r"别忘了",
            r"记下来",
            r"保存.*记忆",
            r"remember\s+this",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in patterns)
