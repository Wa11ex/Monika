"""
RL 策略引擎 — C++ PolicyModel 的 Python 包装层

职责：
  1. 加载/管理 C++ PolicyModel 生命周期
  2. 接收 trait 状态 + 对话统计 → 调用 C++ forward() → 返回决策
  3. 实现"残差策略"：手写规则 = 基线参数，RL = 微调 delta

用法：
    engine = PolicyEngine()
    if engine.try_load("rl/weights/policy.bin"):
        decision = engine.decide(features)
        temp = engine.apply_residual(baseline_temp, decision["temperature"])
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from utils.config_loader import get_config
from utils.logger import get_logger

log = get_logger(__name__)


class PolicyEngine:
    """C++ PolicyModel 的 Python 前端"""

    def __init__(self):
        self._model = None
        self._loaded = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def try_load(self, weights_path: Optional[str] = None) -> bool:
        """尝试从文件加载训练好的权重；失败则保留未加载状态（安全退化）"""
        if weights_path is None:
            weights_path = get_config("rl.policy.weights_path", "rl/weights/policy.bin")

        try:
            from rl._rl_engine import PolicyModel
            self._model = PolicyModel()
            self._model.init()
            if not self._model.load(weights_path):
                log.warning("[RL-Engine] 权重文件不存在或损坏: %s，使用基线策略", weights_path)
                return False
            self._loaded = True
            log.info("[RL-Engine] C++ 策略模型已加载: %s", weights_path)
            return True
        except Exception as e:
            log.warning("[RL-Engine] C++ 策略模型加载失败: %s，使用基线策略", e)
            return False

    @property
    def is_ready(self) -> bool:
        return self._loaded and self._model is not None

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def decide(self, features: np.ndarray) -> dict:
        """前向推理，返回决策字典（所有值在各自有效范围内）

        features: (15,) float32 — 由 rl.features.build_features() 构建

        Returns:
            {
                "memory_count":   float [1, 5],
                "use_tool_prob":  float [0, 1],
                "temperature":    float [0.5, 1.5],
                "max_tokens":     float [128, 1024],
                "tool_logits":    list[float] (6 dims),
                "mode_logits":    list[float] (3 dims),
            }
        """
        if not self.is_ready:
            return _default_decisions()

        try:
            out = self._model.forward(features.astype(np.float32))
            # Python dict 已经是 pybind11 转换后的结果
            return out
        except Exception as e:
            log.warning("[RL-Engine] C++ 推理异常: %s，退回默认值", e)
            return _default_decisions()

    # ------------------------------------------------------------------
    # 残差策略：手写规则基线 + RL delta
    # ------------------------------------------------------------------

    @staticmethod
    def apply_residual(baseline: float, rl_value: float,
                       lo: float, hi: float) -> float:
        """残差模式：baseline + (rl_value - neutral) * scale，clamp 到 [lo, hi]

        neutral = (lo + hi) / 2，即 sigmoid(W·x) 在权重为 0 时的输出

        例如：温度 baseline=1.0, rl_value=1.2, range [0.5, 1.5]
        → neutral = 1.0, delta = 1.2 - 1.0 = 0.2, result = 1.2
        """
        neutral = (lo + hi) / 2.0
        delta = rl_value - neutral
        result = baseline + delta
        return max(lo, min(hi, result))


def _default_decisions() -> dict:
    """未加载 RL 模型时的中性决策（不改变任何参数）"""
    return {
        "memory_count": 3.0,
        "use_tool_prob": 0.5,
        "temperature": 1.0,
        "max_tokens": 512.0,
        "tool_logits": [1.0 / 6.0] * 6,
        "mode_logits": [1.0 / 3.0] * 3,
    }
