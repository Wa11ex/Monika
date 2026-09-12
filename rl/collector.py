"""
SignalCollector — 寄生在现有系统上的信号收集器

用法 (在 main.py 主循环中):
    collector = SignalCollector(log_dir="rl/logs")
    collector.on_turn_start(user_text, trait_snapshot, speaker_name, emotion)
    # ... 一轮对话 ...
    collector.on_turn_end(assistant_text, interrupted, tool_calls, tool_success, tool_fail, duration_ms)
    collector.on_feedback("up")   # 可选：用户点赞/踩
    collector.on_session_end(total_turns)  # 会话结束时调用

每个 InteractionLog 以 jsonl 格式写入 rl/logs/YYYY-MM-DD.jsonl
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# 延迟导入，避免循环依赖
# ---------------------------------------------------------------------------
def _import_signals():
    from rl.signals import SignalType, InteractionLog, compute_reward
    return SignalType, InteractionLog, compute_reward


class SignalCollector:
    """事件钩子集合 — 对接 main.py 和 ui/main_window.py"""

    def __init__(self, log_dir: str = "rl/logs"):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        self._current_log = None   # InteractionLog
        self._prev_log = None      # 用于填充延迟信号（USER_CONTINUED）
        self._turn_count = 0
        self._session_start = None

    # ------------------------------------------------------------------
    # 生命周期钩子
    # ------------------------------------------------------------------

    # PO 触发时的 user_text 标记（main.py 的 PO_TRIGGER_TEXT）
    _PO_MARKERS = {"[沉默]", "[silence]", "[PO]"}

    def on_turn_start(
        self,
        user_text: str,
        trait_snapshot: Optional[Dict[str, float]] = None,
        speaker_name: str = "",
        emotion: Optional[str] = None,
    ) -> None:
        """一轮对话开始时调用（在 think_stream 之前）"""
        SignalType, InteractionLog, _ = _import_signals()

        self._turn_count += 1
        if self._session_start is None:
            self._session_start = __import__("time").time()

        # 检测是否为 PO（主动发言）触发
        is_proactive = user_text.strip() in self._PO_MARKERS

        self._current_log = InteractionLog(
            user_text=user_text,
            user_emotion=emotion,
            speaker_name=speaker_name,
            trait_snapshot=trait_snapshot or {},
        )
        # PO 触发：记录为用户沉默
        if is_proactive:
            self._current_log.signals.append(SignalType.SILENCE_TIMEOUT.name)

    def on_turn_end(
        self,
        assistant_text: str = "",
        interrupted: bool = False,
        tool_calls: int = 0,
        tool_success: int = 0,
        tool_fail: int = 0,
        duration_ms: float = 0.0,
    ) -> None:
        """一轮对话结束时调用（在 commit_response 之后）"""
        if self._current_log is None:
            return

        SignalType, InteractionLog, compute_reward = _import_signals()
        log_entry = self._current_log

        log_entry.assistant_text = assistant_text
        log_entry.interrupted = interrupted
        log_entry.tool_calls = tool_calls
        log_entry.tool_success = tool_success
        log_entry.tool_fail = tool_fail
        log_entry.total_duration_ms = duration_ms

        # -- 采集本轮的直接信号 --
        if interrupted:
            log_entry.signals.append(SignalType.INTERRUPTED.name)

        if tool_success > 0:
            log_entry.signals.append(SignalType.TOOL_SUCCEEDED.name)
        if tool_fail > 0:
            log_entry.signals.append(SignalType.TOOL_FAILED.name)

        # -- 每轮立即写盘（即使非正常退出也有数据） --
        log_entry.composite_reward = compute_reward(log_entry)
        self._save_log(log_entry)

        # 如果上一轮还没标 USER_CONTINUED，现在标上并更新已写入的记录
        if self._prev_log is not None:
            if not interrupted:
                self._prev_log.signals.append(SignalType.USER_CONTINUED.name)
            if self._turn_count > 10:
                self._prev_log.signals.append(SignalType.LONG_TURN.name)
            self._prev_log.composite_reward = compute_reward(self._prev_log)
            self._save_log(self._prev_log)  # 用 turn_id 更新已有行

        self._prev_log = log_entry
        self._current_log = None

    def on_feedback(self, up_or_down: str) -> None:
        """用户点赞 (up) 或踩 (down) 时调用"""
        if self._prev_log is None:
            return

        SignalType, _, _ = _import_signals()
        if up_or_down == "up":
            self._prev_log.signals.append(SignalType.EXPLICIT_UP.name)
        elif up_or_down == "down":
            self._prev_log.signals.append(SignalType.EXPLICIT_DOWN.name)

    def on_session_end(self) -> None:
        """会话结束时调用（关闭 / Ctrl+C）"""
        if self._prev_log is None:
            return

        SignalType, _, compute_reward = _import_signals()

        if self._turn_count < 3:
            self._prev_log.signals.append(SignalType.SHORT_SESSION.name)

        self._prev_log.composite_reward = compute_reward(self._prev_log)
        self._save_log(self._prev_log)
        self._prev_log = None

        log.info("[RL-Collector] 会话结束，共 %d 轮", self._turn_count)

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def _save_log(self, log_entry) -> None:
        """写入/更新一条日志（按 turn_id 去重，存在则替换该行）"""
        from dataclasses import asdict

        fname = f"{date.today()}.jsonl"
        fpath = self._log_dir / fname
        record = asdict(log_entry)
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        tid = record.get("turn_id", "")

        # 读取已有行，按 turn_id 替换
        existing = {}
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        rec = json.loads(ln)
                        existing[rec.get("turn_id", "")] = ln
                    except json.JSONDecodeError:
                        continue

        existing[tid] = line.strip()
        with open(fpath, "w", encoding="utf-8") as f:
            for ln in existing.values():
                f.write(ln + "\n")
