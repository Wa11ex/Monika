'''
轻量 Benchmark 收集器 — 在关键路径埋点，方便之后添加新指标

from utils.benchmark import bench
bench.tick()                          # 开始计时
bench.tock("think_ms")                # 记录耗时
bench.record("format_ok", True)       # 记录布尔指标
bench.record("tool_calls", 1)         # 累加计数
bench.commit()                        # 提交本轮数据
bench.report()                        # 生成报告
'''

from __future__ import annotations

import time
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class TurnMetrics:
    '''单轮对话的时序与正确性指标'''
    # 时间
    asr_ms: float = 0.0            # ASR 识别耗时
    think_ms: float = 0.0          # LLM 思考阶段耗时（首 token 到 </think>）
    talk_ms: float = 0.0           # LLM 生成回复耗时
    tts_synth_ms: float = 0.0      # TTS 合成总耗时
    audio_play_ms: float = 0.0     # 音频播放总耗时
    total_ms: float = 0.0          # 端到端耗时（用户输入->播放完毕）
    
    # 计数
    tool_calls: int = 0            # 工具调用次数
    tool_success: int = 0          # 工具调用成功次数
    tool_fail: int = 0             # 工具调用失败次数
    sentence_count: int = 0        # TTS 句子数
    
    # T/F
    format_ok: bool = True         # 回复格式是否符合 (表情)[动作]… 规范
    interrupted: bool = False      # 是否被用户打断
    think_truncated: bool = False  # 思考阶段是否被截断（提前 <|im_end|>）
    
    # 元数据
    turn_index: int = 0            # 本轮序号（从 0 开始）
    speaker_name: str = ""         # 说话人名称
    timestamp: str = ""            # ISO 时间戳


class Benchmark:
    '''全局 Benchmark 单例'''

    def __init__(self):
        self._turns: List[TurnMetrics] = []
        self._current = TurnMetrics()
        self._start: float = 0.0
        self._phase_starts: Dict[str, float] = {}
        self._turn_start: float = 0.0

    # -- 计时 API --------------------------------------------

    def tick(self) -> None:
        '''标记当前时刻（后续 tock 的起点）'''
        self._start = time.perf_counter()

    def tock(self, attr: str) -> float:
        '''将自上次 tick 以来的耗时（ms）写入当前 TurnMetrics 的 attr 字段
        返回毫秒
        '''
        elapsed = (time.perf_counter() - self._start) * 1000.0
        if hasattr(self._current, attr):
            setattr(self._current, attr, elapsed)
        return elapsed

    def phase_start(self, name: str) -> None:
        '''开始一个命名的计时'''
        self._phase_starts[name] = time.perf_counter()

    def phase_end(self, name: str, attr: str) -> float:
        '''结束命名计时，耗时写入 attr'''
        start = self._phase_starts.get(name, time.perf_counter())
        elapsed = (time.perf_counter() - start) * 1000.0
        if hasattr(self._current, attr):
            setattr(self._current, attr, elapsed)
        return elapsed

    def turn_start(self) -> None:
        '''标记一轮对话的开始'''
        self._turn_start = time.perf_counter()

    def turn_end(self) -> float:
        '''计算端到端耗时'''
        elapsed = (time.perf_counter() - self._turn_start) * 1000.0
        self._current.total_ms = elapsed
        return elapsed

    # -- 记录用 API --------------------------------------------

    def record(self, attr: str, value: Any) -> None:
        '''写入任意字段（计数、布尔等）'''
        if hasattr(self._current, attr):
            current = getattr(self._current, attr)
            if isinstance(current, (int, float)) and isinstance(value, (int, float)):
                setattr(self._current, attr, current + value)
            else:
                setattr(self._current, attr, value)

    # -- 生命周期 --------------------------------------------

    def commit(self, speaker_name: str = "") -> None:
        '''提交当前轮次数据到历史列表，重置 current'''
        self._current.turn_index = len(self._turns)
        self._current.speaker_name = speaker_name
        from datetime import datetime
        self._current.timestamp = datetime.now().isoformat()
        
        self._turns.append(self._current)
        self._current = TurnMetrics()
        self._phase_starts.clear()

    # -- log ------------------------------------------------

    def report(self) -> str:
        '''生成 Markdown 格式的汇总报告'''
        if not self._turns:
            return "（暂无 Benchmark 数据）"
        
        n = len(self._turns)

        def _avg(attr: str) -> float:
            return sum(getattr(t, attr, 0) for t in self._turns) / max(n, 1)
        
        lines = [
            "## Benchmark 报告",
            f"",
            f"**轮次**: {n}",
            f"",
            f"### A 类：管道性能",
            f"| 指标 | 平均 | 最小 | 最大 |",
            f"|------|------|------|------|",
        ]
        for attr, label in [
            ("total_ms", "端到端延迟"),
            ("think_ms", "Think 阶段"),
            ("talk_ms", "Talk 阶段"),
            ("tts_synth_ms", "TTS 合成"),
            ("audio_play_ms", "音频播放"),
        ]:
            vals = [getattr(t, attr, 0) for t in self._turns if getattr(t, attr, 0) > 0]
            if vals:
                lines.append(f"| {label} | {sum(vals)/len(vals):.0f}ms | {min(vals):.0f}ms | {max(vals):.0f}ms |")

        lines.append(f"")
        lines.append(f"### B 类：功能正确性")
        lines.append(f"| 指标 | 值 |")
        lines.append(f"|------|----|")
        
        format_ok_n = sum(1 for t in self._turns if t.format_ok)
        lines.append(f"| 格式遵循率 | {format_ok_n}/{n} ({format_ok_n/n*100:.0f}%) |")
        
        tool_total = sum(t.tool_calls for t in self._turns)
        tool_ok = sum(t.tool_success for t in self._turns)
        if tool_total > 0:
            lines.append(f"| 工具调用成功率 | {tool_ok}/{tool_total} ({tool_ok/tool_total*100:.0f}%) |")
        
        interrupted_n = sum(1 for t in self._turns if t.interrupted)
        if interrupted_n:
            lines.append(f"| 被打断轮次 | {interrupted_n}/{n} |")
        
        truncated_n = sum(1 for t in self._turns if t.think_truncated)
        if truncated_n:
            lines.append(f"| Think 截断次数 | {truncated_n} |")
        
        if tool_total > 0:
            lines.append(f"")
            lines.append(f"### 工具调用明细")
            lines.append(f"| # | 成功 | 失败 | 说话人 |")
            for t in self._turns:
                if t.tool_calls > 0:
                    lines.append(f"| {t.turn_index} | {t.tool_success} | {t.tool_fail} | {t.speaker_name or '-'} |")
        
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps([asdict(t) for t in self._turns], ensure_ascii=False, indent=2)

    def reset(self) -> None:
        '''重置所有数据（测试用）'''
        self._turns.clear()
        self._current = TurnMetrics()
        self._phase_starts.clear()

bench = Benchmark()
