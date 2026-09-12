"""
Monika Trait 系统 — 心情、寂寞等内部状态。

详情见 plans/trait-system.md
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from utils.config_loader import get_config
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class TraitSnapshot:
    curiosity: float = 0.5
    affection: float = 0.5
    mood: float = 0.5
    loneliness: float = 0.0
    timestamp: float = 0.0


@dataclass
class TriggeredAction:
    action: str         # "web_search" | "po_talk" | ...
    urgency: float      # 0~1
    context: dict       # 附加数据
    trait_source: str   # 触发来源


class TraitManager:
    """Monika 内部状态。

    tm = TraitManager()
    tm.update("user_spoke", {"text": "..."})
    tm.tick(300)
    hints = tm.get_prompt_hints()
    """

    def __init__(self, save_path: str | None = None):
        self._save_path = save_path or os.path.join(
            os.path.dirname(__file__), "..", "memory", "traits.json"
        )

        # -- 数值 --
        self._curiosity: float = 0.5
        self._affection: float = 0.5
        self._mood: float = 0.5
        self._loneliness: float = 0.0

        # -- 冷却 -- 
        self._cooldowns: Dict[str, float] = {}
        self._last_tick: float = time.time()

    # -- 只读属性 ----------

    @property
    def mood(self) -> float:
        return self._mood

    @property
    def loneliness(self) -> float:
        return self._loneliness

    @property
    def curiosity(self) -> float:
        return self._curiosity

    @property
    def affection(self) -> float:
        return self._affection

    def snapshot(self) -> TraitSnapshot:
        return TraitSnapshot(
            curiosity=self._curiosity,
            affection=self._affection,
            mood=self._mood,
            loneliness=self._loneliness,
            timestamp=time.time(),
        )

    # -- 事件 ----------

    def update(self, event: str, ctx: Dict[str, Any] | None = None) -> None:
        """接事件，改数值。支持的事件见 _on_* 方法。"""
        ctx = ctx or {}
        handler = getattr(self, f"_on_{event}", None)
        if handler:
            handler(ctx)
            self._clamp()
        else:
            log.debug("[Trait] 未知事件: %s", event)

    # -- 时间流逝 ----------

    def tick(self, dt_seconds: float) -> None:
        """每 dt_seconds 秒调用一次。"""
        # 每 5 分钟
        steps = dt_seconds / 300.0
        self._mood += 0.02 * steps if self._mood < 0.5 else -0.02 * steps
        self._loneliness += 0.015 * steps
        self._last_tick = time.time()
        self._clamp()

    # -- 触发 ----------

    def check_triggers(self) -> List[TriggeredAction]:
        '''根据当前 trait 值返回待执行的主动行为列表，每个行为有自己的冷却'''
        actions = []
        now = time.time()

        # loneliness > 0.7 → po_talk (cooldown: 5 min)
        if self._loneliness > 0.7 and self._cooldown_ok("po_loneliness", 300, now):
            urgency = (self._loneliness - 0.7) / 0.3
            actions.append(TriggeredAction(
                action="po_talk", urgency=min(urgency, 1.0),
                context={"reason": "loneliness"}, trait_source="loneliness"))

        # curiosity > 0.6 → proactive_web_search (cooldown: 10 min)
        if self._curiosity > 0.6 and self._cooldown_ok("curiosity_search", 600, now):
            urgency = (self._curiosity - 0.6) / 0.4
            actions.append(TriggeredAction(
                action="web_search", urgency=min(urgency, 1.0),
                context={"reason": "curiosity"}, trait_source="curiosity"))

        # affection > 0.8 AND loneliness > 0.3 → care_po (cooldown: 20 min)
        if self._affection > 0.8 and self._loneliness > 0.3 and self._cooldown_ok("care_po", 1200, now):
            actions.append(TriggeredAction(
                action="care_po", urgency=0.5,
                context={"reason": "affection+loneliness"}, trait_source="affection"))

        # curiosity > 0.5 AND mood > 0.5 → suggest_activity (cooldown: 15 min)
        if self._curiosity > 0.5 and self._mood > 0.5 and self._cooldown_ok("suggest_activity", 900, now):
            actions.append(TriggeredAction(
                action="suggest_activity", urgency=0.3,
                context={"reason": "curious+happy"}, trait_source="curiosity"))

        return actions

    def _cooldown_ok(self, key: str, seconds: float, now: float) -> bool:
        '''冷却检查：没过冷却返回 False；否则设新冷却返回 True'''
        if key not in self._cooldowns or self._cooldowns[key] <= now:
            self._cooldowns[key] = now + seconds
            return True
        return False

    def get_prompt_hints(self) -> str:
        """当数值极端时返回一行自然语言提示，否则空串。"""
        hints = []
        if self._mood < 0.2:
            hints.append("你现在心情有些低落。")
        elif self._mood > 0.8:
            hints.append("你现在心情很愉悦，感觉一切都很美好。")
        if self._loneliness > 0.7:
            hints.append("你现在有点想念用户，感到有些孤单。")
        return "\n".join(hints) if hints else ""

    # -- 持久化 ----------

    def load(self) -> bool:
        try:
            if not os.path.exists(self._save_path):
                return False
            with open(self._save_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._mood = float(data.get("mood", 0.5))
            self._loneliness = float(data.get("loneliness", 0.0))
            self._curiosity = float(data.get("curiosity", 0.5))
            self._affection = float(data.get("affection", 0.5))
            self._last_tick = float(data.get("last_tick", time.time()))
            self._clamp()

            elapsed = time.time() - self._last_tick
            if elapsed > 60:
                self.tick(elapsed)
            log.info("[Trait] 加载完成，离线 %.0f 秒", elapsed)
            return True
        except Exception as e:
            log.warning("[Trait] 加载失败: %s", e)
            return False

    def save(self) -> bool:
        try:
            os.makedirs(os.path.dirname(self._save_path), exist_ok=True)
            with open(self._save_path, "w", encoding="utf-8") as f:
                json.dump({
                    "mood": self._mood,
                    "loneliness": self._loneliness,
                    "curiosity": self._curiosity,
                    "affection": self._affection,
                    "last_tick": self._last_tick,
                }, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            log.warning("[Trait] 保存失败: %s", e)
            return False

    # -- 事件实现 ----------

    def _on_user_spoke(self, ctx: dict) -> None:
        text = ctx.get("text", "")
        self._mood += 0.05
        self._loneliness = 0.0
        # TODO: curiosity / affection

    def _on_tool_called(self, ctx: dict) -> None:
        '''调用工具说明 Monika 有好奇心'''
        self._curiosity += 0.03

    def _on_tool_success(self, ctx: dict) -> None:
        self._mood += 0.03

    def _on_tool_failure(self, ctx: dict) -> None:
        self._mood -= 0.05

    def _on_feedback_up(self, ctx: dict) -> None:
        self._mood += 0.10

    def _on_feedback_down(self, ctx: dict) -> None:
        self._mood -= 0.15

    def _on_user_interrupt(self, ctx: dict) -> None:
        self._mood -= 0.08

    def _on_po_timeout(self, ctx: dict) -> None:
        self._loneliness += 0.05

    def _on_proactive_search_done(self, ctx: dict) -> None:
        pass  # TODO: curiosity

    def _on_proactive_po_sent(self, ctx: dict) -> None:
        self._loneliness -= 0.05

    def _on_proactive_care_sent(self, ctx: dict) -> None:
        pass  # TODO: affection

    # -- 内部 ----------

    def _clamp(self) -> None:
        self._curiosity = max(0.0, min(1.0, self._curiosity))
        self._affection = max(0.0, min(1.0, self._affection))
        self._mood = max(0.0, min(1.0, self._mood))
        self._loneliness = max(0.0, min(1.0, self._loneliness))

    def __repr__(self) -> str:
        return (f"TraitManager(mood={self._mood:.2f}, "
                f"loneliness={self._loneliness:.2f})")
