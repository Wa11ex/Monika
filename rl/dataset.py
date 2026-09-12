"""
数据集加载 — 从 rl/logs/*.jsonl 构建 (features, reward, label) 训练样本

数据流：
  rl/logs/2026-07-25.jsonl  →  逐行解析 InteractionLog
      →  提取 trait_snapshot + conv stats → 15 维特征
      →  composite_reward 作为 RM 的 target
      →  labeled_as: "good" (reward > 0.55) / "bad" (reward < 0.45) / None

用法：
    ds = RLDataset("rl/logs/")
    features, rewards, labels = ds.load()
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


class RLDataset:
    """从 jsonl 日志加载 RL 训练数据"""

    def __init__(self, log_dir: str = "rl/logs"):
        self._log_dir = Path(log_dir)

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """加载所有日志，返回 (features, rewards, labels)

        Returns:
            features: (N, 15) float32 — 每轮的特征向量
            rewards:  (N,) float32 — composite_reward [0, 1]
            labels:   (N,) int8 — 1=good, 0=bad, -1=neutral(丢弃)
        """
        records = []
        for fpath in sorted(self._log_dir.glob("*.jsonl")):
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not records:
            return (
                np.zeros((0, 15), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int8),
            )

        features = np.array(
            [self._record_to_features(r) for r in records],
            dtype=np.float32,
        )
        rewards = np.array(
            [float(r.get("composite_reward", 0.5)) for r in records],
            dtype=np.float32,
        )
        # 标签：reward > 0.55 = good(1), < 0.45 = bad(0), 中间丢弃(-1)
        labels = np.full(len(records), -1, dtype=np.int8)
        labels[rewards > 0.55] = 1
        labels[rewards < 0.45] = 0

        return features, rewards, labels

    @staticmethod
    def _record_to_features(r: dict) -> list:
        """从 InteractionLog dict 提取 15 维特征（与 rl/features.py 一致）"""
        import time as _time
        ts = float(r.get("timestamp", 0))
        t = _time.localtime(ts) if ts > 0 else _time.localtime()

        trait = r.get("trait_snapshot", {})
        return [
            float(trait.get("mood", 0.5)),
            float(trait.get("loneliness", 0.0)),
            float(trait.get("curiosity", 0.5)),
            float(trait.get("affection", 0.5)),
            0.0,  # turn_norm (日志里没存，后续补)
            0.0,  # tool_this_turn
            0.0, 0.0, 0.0,  # reserved
            min(float(r.get("tool_calls", 0)), 10.0) / 10.0,
            min(float(r.get("tool_success", 0)), 10.0) / 10.0,
            min(float(r.get("tool_fail", 0)), 10.0) / 10.0,
            float(t.tm_hour) / 24.0,
            float(t.tm_wday) / 7.0,
            float(t.tm_min) / 60.0,
        ]

    def stats(self) -> dict:
        """返回数据集统计信息"""
        features, rewards, labels = self.load()
        n_good = int((labels == 1).sum())
        n_bad = int((labels == 0).sum())
        return {
            "total": len(features),
            "good": n_good,
            "bad": n_bad,
            "neutral": len(features) - n_good - n_bad,
            "mean_reward": float(rewards.mean()) if len(rewards) > 0 else 0.0,
        }
