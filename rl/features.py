"""
特征构建器 — 将 trait 状态 + 对话上下文 → 15 维特征向量

PolicyModel 的输入布局（15 维，全部归一化到 [0, 1]）：
  [0]  mood       — 当前心情
  [1]  loneliness — 寂寞感
  [2]  curiosity  — 好奇心
  [3]  affection  — 好感度
  [4]  turn_norm  — 对话轮次 / 100
  [5]  tool_ratio — 本轮已用工具数 / 5（max rounds）
  [6]  _reserved  — 预留
  [7]  _reserved  — 预留
  [8]  _reserved  — 预留
  [9]  tool_called_norm — 累计调用次数 / 10
  [10] tool_ok_norm     — 累计成功次数 / 10
  [11] tool_fail_norm   — 累计失败次数 / 10
  [12] hour       — 当前小时 / 24
  [13] weekday    — 星期几 / 7
  [14] minute     — 当前分钟 / 60
"""

from __future__ import annotations

import time as _time
from typing import Any, Dict

import numpy as np


def build_features(
    trait_snapshot: Dict[str, float],
    turn_count: int = 0,
    tool_called: int = 0,
    tool_success: int = 0,
    tool_fail: int = 0,
    tool_this_turn: int = 0,
    timestamp: float = 0.0,
) -> np.ndarray:
    """构建 15 维特征向量

    Args:
        trait_snapshot: {"mood": 0.5, "loneliness": 0.0, "curiosity": 0.5, "affection": 0.5}
        turn_count: 当前对话轮次（从 0 开始）
        tool_called: 累计工具调用次数
        tool_success: 累计工具调用成功次数
        tool_fail: 累计工具调用失败次数
        tool_this_turn: 本轮已使用的工具数（用于 tool_ratio）
        timestamp: 当前时间戳（0 表示用真实时间）

    Returns:
        (15,) float32 numpy array，所有值 ∈ [0, 1]
    """
    now = timestamp if timestamp > 0 else _time.time()
    t = _time.localtime(now)

    feats = np.array([
        # traits (0-3)
        float(trait_snapshot.get("mood", 0.5)),
        float(trait_snapshot.get("loneliness", 0.0)),
        float(trait_snapshot.get("curiosity", 0.5)),
        float(trait_snapshot.get("affection", 0.5)),
        # conversation stats (4-8)
        min(float(turn_count), 100.0) / 100.0,
        min(float(tool_this_turn), 5.0) / 5.0,
        0.0,  # reserved
        0.0,  # reserved
        0.0,  # reserved
        # tool history (9-11)
        min(float(tool_called), 10.0) / 10.0,
        min(float(tool_success), 10.0) / 10.0,
        min(float(tool_fail), 10.0) / 10.0,
        # time (12-14)
        float(t.tm_hour) / 24.0,
        float(t.tm_wday) / 7.0,
        float(t.tm_min) / 60.0,
    ], dtype=np.float32)

    return feats
