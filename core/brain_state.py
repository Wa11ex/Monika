'''
Brain 状态机 — 流式 token 路由（层次化栈模型）

状态层次（栈结构），非常灵活：
  根状态：  THINK 或 TALK（互斥，只能二选一在栈底，至少一个）
  嵌套状态： TOOL  可被 THINK 或 TALK 包裹
            CODE  可被 TALK 包裹（think 内部不触发 CODE）
'''

from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Optional, Tuple

from core.tools.base import ToolResult
from utils.logger import get_logger

log = get_logger(__name__)

S_THINK = 0
S_TALK  = 1
S_TOOL  = 2
S_CODE  = 3
S_DONE  = 4

S_NAME = ["THINK", "TALK", "TOOL", "CODE", "DONE"]

_CS_DELIMITER = "</think>"
_TOOL_OPEN    = "<tool_call>"
_TOOL_CLOSE   = "</tool_call>"
_CODE_FENCE   = "```"

# 滑动窗口保留长度（确保不分割关键标签）
_KEEP_LEN = max(len(_CS_DELIMITER), len(_TOOL_OPEN), len(_TOOL_CLOSE)) * 2


class ToolFormatAdapter(ABC):
    '''Tool call 格式适配器基类
    每个模型有不同的 tool call 格式（Qwen XML、DeepSeek、Llama 等）
    子类实现 feed/flush/reset，由 BrainStateMachine 在 TOOL 状态下调用
    '''

    @abstractmethod
    def feed(self, token: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        '''处理一个 token，返回 (text, tool_call_or_None)'''
        ...

    @abstractmethod
    def flush(self) -> Tuple[str, Optional[Dict[str, Any]]]:
        '''流结束时的清理，返回未完成的残留文本'''
        ...

    @abstractmethod
    def reset(self) -> None:
        '''重置状态，准备新一轮解析'''
        ...


class QwenXMLAdapter(ToolFormatAdapter):
    '''Qwen3.5 tool call XML 格式适配器

    解析形：
        <tool_call>
        <function=read_file>
        <parameter=path>config.yaml</parameter>
        </function>
        </tool_call>
    '''

    _TOOL_CALL_OPEN  = "<tool_call>"
    _TOOL_CALL_CLOSE = "</tool_call>"
    _FUNCTION_RE     = re.compile(r"<function=([^>]+)>")
    _PARAM_OPEN_RE   = re.compile(r"<parameter=([^>]+)>")
    _PARAM_CLOSE     = "</parameter>"
    _FUNCTION_CLOSE  = "</function>"

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._buf = ""
        self._state = "idle"          # idle | in_tool_call | in_function | in_param
        self._func_name: str | None = None
        self._params: Dict[str, str] = {}
        self._current_param: str | None = None
        self._param_buf: str = ""
        self._text_buf: str = ""

    def feed(self, token: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        if not token:
            return "", None
        self._buf += token

        if self._state == "idle":
            idx = self._buf.find(self._TOOL_CALL_OPEN)
            if idx == -1:
                keep = min(len(self._buf), len(self._TOOL_CALL_OPEN))
                text = self._buf[:-keep] if len(self._buf) > keep else ""
                self._buf = self._buf[-keep:]
                self._text_buf += text
                return text, None
            text = self._buf[:idx]
            self._buf = self._buf[idx + len(self._TOOL_CALL_OPEN):]
            self._text_buf += text
            self._state = "in_tool_call"
            return text, None

        elif self._state == "in_tool_call":
            m = self._FUNCTION_RE.search(self._buf)
            if m:
                self._func_name = m.group(1).strip()
                self._buf = self._buf[m.end():]
                self._state = "in_function"
                return "", None
            if self._TOOL_CALL_CLOSE in self._buf:
                self.reset()
                return "", None
            return "", None

        elif self._state == "in_function":
            m = self._PARAM_OPEN_RE.search(self._buf)
            if m:
                self._current_param = m.group(1).strip()
                self._buf = self._buf[m.end():]
                self._param_buf = ""
                self._state = "in_param"
                return "", None
            idx = self._buf.find(self._FUNCTION_CLOSE)
            if idx != -1:
                self._buf = self._buf[idx + len(self._FUNCTION_CLOSE):]
                self._state = "after_function"
                return "", None
            return "", None

        elif self._state == "in_param":
            idx = self._buf.find(self._PARAM_CLOSE)
            if idx != -1:
                self._param_buf += self._buf[:idx]
                self._params[self._current_param] = self._param_buf.strip()
                self._buf = self._buf[idx + len(self._PARAM_CLOSE):]
                self._current_param = None
                self._param_buf = ""
                self._state = "in_function"
                return "", None
            else:
                keep = min(len(self._buf), len(self._PARAM_CLOSE))
                self._param_buf += self._buf[:-keep] if len(self._buf) > keep else ""
                self._buf = self._buf[-keep:]
                return "", None

        elif self._state == "after_function":
            idx = self._buf.find(self._TOOL_CALL_CLOSE)
            if idx != -1:
                result = {"name": self._func_name, "params": dict(self._params)}
                self.reset()
                return "", result
            return "", None

        return "", None

    def flush(self) -> Tuple[str, Optional[Dict[str, Any]]]:
        remaining = self._buf
        self.reset()
        if remaining:
            return remaining, None
        return "", None


def create_tool_adapter(fmt: str = "qwen_xml", template_str: str = "") -> ToolFormatAdapter:
    '''根据格式名称或模板内容创建 tool call 适配器
        fmt:          "qwen_xml" | "auto"（auto 时从模板探测）
        template_str: Jinja 模板内容（fmt="auto" 时用于探测）
        返回 ToolFormatAdapter 实例
    '''
    if fmt == "auto":
        fmt = _detect_tool_format(template_str)
    if fmt == "qwen_xml":
        return QwenXMLAdapter()
    # 未来扩展: deepseek, llama_func 等
    log.warning("[Brain.State] 未知 tool call 格式 '%s'，回退到 Qwen XML", fmt)
    return QwenXMLAdapter()


def _detect_tool_format(template_str: str) -> str:
    '''从 Jinja 模板内容探测 tool call 格式'''
    if "<tool_call>" in template_str:
        return "qwen_xml"
    if "tool▁call▁begin" in template_str:
        return "deepseek"
    if "<function=" in template_str and "<tool_call>" not in template_str:
        return "llama_func"
    return "qwen_xml"


class BrainStateMachine:
    '''流式 token 层次化状态机

    栈顶为当前活跃状态THINK 和 TALK 只存在于栈底（互斥），
    TOOL 和 CODE 作为嵌套状态推入栈中，完成后弹出恢复

    调用：
        sm = BrainStateMachine(enable_thinking=True, on_tool_call=my_tool_executor)
        async for talk_text in sm.process(token_stream):
            await tts.synthesize(talk_text)
    '''

    def __init__(
        self,
        enable_thinking: bool = True,
        on_tool_call: Optional[Callable[[str, dict], Awaitable[ToolResult]]] = None,
        tool_format: str = "qwen_xml",
    ):
        self._enable_thinking = enable_thinking
        self._on_tool_call = on_tool_call
        # 栈：栈底是根状态（THINK 或 TALK），栈顶是当前嵌套状态
        self._stack: list[int] = [S_THINK if enable_thinking else S_TALK]
        self._state_buf = ""
        self._talk_parts: list[str] = []
        self._think_parts: list[str] = []
        self._raw_buf: list[str] = []
        # tool call 格式适配器（可按模型切换）
        self._tool_adapter = create_tool_adapter(tool_format)

    # -- 栈操作 ------------------------------------------

    @property
    def _state(self) -> int:
        return self._stack[-1]

    def _push(self, state: int) -> None:
        self._stack.append(state)

    def _pop(self) -> int:
        '''弹出栈顶，返回弹出的状态栈底不可弹出'''
        if len(self._stack) > 1:
            return self._stack.pop()
        return self._stack[0]

    # -- 公开调用区 ----------------------------------------

    async def process(
        self,
        token_stream: AsyncGenerator[str, None],
    ) -> AsyncGenerator[str, None]:
        '''处理 token 流，yield TALK 文本片段'''
        log.debug("[Brain.State] -> %s (stack: %s)", S_NAME[self._state],
                  [S_NAME[s] for s in self._stack])

        async for token in token_stream:
            if not token:
                continue
            self._state_buf += token

            # -- 逐句调试输出：遇换行或累积超 200 字符就 flush --
            self._raw_buf.append(token)
            raw_len = sum(len(t) for t in self._raw_buf)
            if '\n' in token or raw_len > 200:
                self._flush_raw()

            if self._state == S_THINK:
                self._handle_think()

            elif self._state == S_TALK:
                yield_text = self._handle_talk()
                if yield_text:
                    yield yield_text

            elif self._state == S_TOOL:
                self._handle_tool()

            elif self._state == S_CODE:
                self._handle_code()

        # 收尾（flush 残留的非换行内容）
        self._flush_raw()
        
        # 注意：_state_buf 的追加由 flush_remaining() 统一处理，
        # 有一种可能性是流在 THINK 状态结束，导致 think_parts 有残留但未 flush，那就重复了
        if self._state == S_THINK and self._state_buf.strip():
            # 模型在 THINK 状态就结束了（可能提前输出 <|im_end|>，TODO）
            self._think_parts.append(self._state_buf)
            self._state_buf = ""
            log.warning("[Brain.State] [WARN] 流在 THINK 状态结束（可能提前 <|im_end|>），"
                        "think 长度=%d chars", sum(len(p) for p in self._think_parts))
        if self._think_parts:
            log.debug("[Brain.Think]\n%s", ''.join(self._think_parts)[:500])

    async def flush_remaining(self) -> Optional[str]:
        '''返回收尾时残留的 TALK 文本'''
        if self._state == S_TALK and self._state_buf.strip():
            self._talk_parts.append(self._state_buf)
            self._state_buf = ""
        if self._talk_parts:
            result = ''.join(self._talk_parts)
            self._talk_parts.clear()
            return result
        return None

    def get_pending_tool(self) -> Optional[tuple[str, dict]]:
        return getattr(self, '_pending_tool', None)

    def set_pending_tool(self, name: str, params: dict) -> None:
        self._pending_tool = (name, params)

    # -- 状态处理器 --------------------------------------

    def _handle_think(self) -> None:
        '''THINK: 检测 </think>（结束思考）或 <tool_call>（嵌套工具调用）'''
        buf = self._state_buf

        # 1. think内<tool_call>
        if _TOOL_OPEN in buf:
            idx = buf.find(_TOOL_OPEN)
            if idx > 0:
                self._think_parts.append(buf[:idx])
            self._state_buf = buf[idx:]
            self._push(S_TOOL)
            self._flush_raw()
            log.debug("[Brain.State] THINK -> TOOL (nested, stack: %s)",
                      [S_NAME[s] for s in self._stack])
            return

        # 2. </think>
        if _CS_DELIMITER in buf:
            idx = buf.find(_CS_DELIMITER)
            self._think_parts.append(buf[:idx])
            self._state_buf = buf[idx + len(_CS_DELIMITER):]
            # 切换根状态：THINK -> TALK
            self._stack[0] = S_TALK
            # 清理栈上残留的嵌套状态
            self._stack = self._stack[:1]
            self._flush_raw()
            log.debug("[Brain.State] THINK -> TALK")
            return

        # 3. DEBUG：滑动窗口优化：buf 太长时 flush 前半部分，保留尾部防标签截断
        # 例子：<tool_call>...</tool_call> 刚好被分割在两批 token 中，导致无法正确识别
        # 只保留最后一段（长度不超过 _KEEP_LEN）未 flush 的文本，前面部分 flush 到 think_parts
        if len(buf) > 80:
            keep = min(len(buf), _KEEP_LEN)
            self._think_parts.append(buf[:-keep])
            self._state_buf = buf[-keep:]

    def _handle_talk(self) -> Optional[str]:
        '''TALK: 累积文本；检测嵌套的 <tool_call> 或 ```'''
        buf = self._state_buf

        # 1. talk内<tool_call>
        if _TOOL_OPEN in buf:
            idx = buf.find(_TOOL_OPEN)
            if idx > 0:
                self._talk_parts.append(buf[:idx])
            result = ''.join(self._talk_parts) if self._talk_parts else None
            self._talk_parts.clear()
            self._state_buf = buf[idx:]
            self._push(S_TOOL)
            self._flush_raw()
            log.debug("[Brain.State] TALK -> TOOL (nested, stack: %s)",
                      [S_NAME[s] for s in self._stack])
            return result

        # 2. 检测 ```（代码块）
        if _CODE_FENCE in buf:
            idx = buf.find(_CODE_FENCE)
            if idx > 0:
                self._talk_parts.append(buf[:idx])
            result = ''.join(self._talk_parts) if self._talk_parts else None
            self._talk_parts.clear()
            self._state_buf = buf[idx:]
            self._push(S_CODE)
            self._flush_raw()
            log.debug("[Brain.State] TALK -> CODE")
            return result

        # 3. 自然断句：换行时立即 yield 到 TTS
        if '\n' in buf:
            self._talk_parts.append(buf)
            self._state_buf = ""
            result = ''.join(self._talk_parts)
            self._talk_parts.clear()
            return result

        return None

    def _handle_tool(self) -> None:
        '''TOOL: 累积 XML，使用 format adapter 解析 tool call'''
        if _TOOL_CLOSE not in self._state_buf:
            return

        end = self._state_buf.find(_TOOL_CLOSE) + len(_TOOL_CLOSE)
        tool_xml = self._state_buf[:end]
        self._state_buf = self._state_buf[end:]

        # 使用格式适配器解析（支持 Qwen XML 及未来其他格式）
        self._tool_adapter.reset()
        _text, result = self._tool_adapter.feed(tool_xml)
        if result is None:
            _text, result = self._tool_adapter.flush()

        if result is None:
            # 适配器无法解析，回退到正则解析
            name, params = _parse_tool_xml(tool_xml)
        else:
            name = result.get("name", "unknown")
            params = result.get("params", {})

        # -- 验证：工具名必须已注册，否则是模板示例/格式说明 --
        try:
            from core.tool_registry import get_tool_registry
            registry = get_tool_registry()
            has_registry = registry.count > 0
            valid_names = registry.list_names() if has_registry else []
        except Exception:
            has_registry = False
            valid_names = []

        if name == "unknown" or not params:
            log.debug("[Brain] 忽略无效 tool_call (name=%r, params=%s)",
                      name, params)
        elif has_registry and name not in valid_names:
            log.debug("[Brain] 忽略未注册 tool_call: %r (已注册: %s)",
                      name, valid_names)
        else:
            log.info("[Brain] Tool call: %s(%s)", name, params)
            self._pending_tool = (name, params)

        # 弹出 TOOL，回到上一状态（THINK 或 TALK）
        popped = self._pop()
        prev_name = S_NAME[self._state]
        log.debug("[Brain.State] TOOL -> %s (popped, stack: %s)",
                  prev_name, [S_NAME[s] for s in self._stack])

    def _handle_code(self) -> None:
        '''CODE: 累积直到找到闭合的 ```，然后弹出到 TALK'''
        second = self._state_buf.find(_CODE_FENCE, len(_CODE_FENCE))
        if second != -1:
            # 丢弃代码块内容（不进入 TTS）
            self._state_buf = self._state_buf[second + len(_CODE_FENCE):]
            self._pop()  # 弹出 CODE，回到 TALK
            self._flush_raw()
            log.debug("[Brain.State] CODE -> TALK")

    # -- 内部函数 --------------------------------------------

    def _flush_raw(self) -> None:
        if self._raw_buf:
            log.debug("[Brain.RAW] %s", ''.join(self._raw_buf))
            self._raw_buf.clear()

    def reset(self) -> None:
        '''重置状态，准备新一轮生成'''
        self._stack = [S_THINK if self._enable_thinking else S_TALK]
        self._state_buf = ""
        self._talk_parts.clear()
        self._think_parts.clear()
        self._raw_buf.clear()
        self._tool_adapter.reset()
        if hasattr(self, '_pending_tool'):
            del self._pending_tool

    @property
    def state(self) -> int:
        return self._state

    @property
    def state_name(self) -> str:
        return S_NAME[self._state]


# -- 工具 XML 解析 ---------------------------------------

def _parse_tool_xml(xml: str) -> tuple[str, dict]:
    '''解析 <tool_call><function=name><parameter=k>v</parameter></function></tool_call>'''
    fn = re.search(r"<function=([^>]+)>", xml)
    name = fn.group(1).strip() if fn else "unknown"
    params = {}
    for m in re.finditer(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", xml, re.DOTALL):
        params[m.group(1).strip()] = m.group(2).strip()
    return name, params
