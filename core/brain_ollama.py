'''
Ollama backend Brain (OpenAI-compatible API to Ollama). （目前已弃用）
All shared logic (history, memory, commit_response) is inherited from BrainBase.
TODO： 改用状态机
'''

import asyncio
import json
import aiohttp
from openai import AsyncOpenAI
from utils.config_loader import get_config
from utils.logger import get_logger
from core.brain_base import BrainBase, _ThinkFilter

log = get_logger(__name__)


class Brain(BrainBase):
    def __init__(self, model_name="Monika:latest", memory_manager=None):
        self._init_shared(memory_manager)
        self.client = AsyncOpenAI(
            base_url=get_config("brain.ollama.base_url", "http://localhost:11434/v1"),
            api_key="ollama",
            timeout=120
        )
        self.model_name = model_name

    async def get_capabilities(self):
        '''通过 Ollama /api/show 探测模型能力（异步，仅在初始化后调用一次）'''
        from core.interfaces import ModelCapabilities

        base_url = get_config("brain.ollama.base_url", "http://localhost:11434/v1")
        
        # /api/show 在 Ollama 根端口上
        ollama_root = base_url.rstrip("/")
        if ollama_root.endswith("/v1"):
            ollama_root = ollama_root[:-3]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{ollama_root}/api/show",
                    json={"name": self.model_name},
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    data = await resp.json()

            template = data.get("template", "")
            details = data.get("details", {})
            families = details.get("families") or []
            arch = details.get("family", "unknown")

            # 视觉支持：Ollama 把 clip 或 mllama 列为 family
            _VISION_FAMILIES = {"clip", "mllama", "llava"}
            supports_vision = bool(_VISION_FAMILIES & set(f.lower() for f in families))

            # 工具调用检查
            supports_native_tools = "{{.Tools}}" in template or "tool_call" in template

            # think链检查
            supports_thinking = "enable_thinking" in template


            max_ctx = 4096
            model_info = data.get("model_info", {})
            for key, val in model_info.items():
                if "context_length" in key:
                    try:
                        max_ctx = int(val)
                    except (ValueError, TypeError):
                        pass
                    break

            return ModelCapabilities(
                backend="ollama",
                model_name=self.model_name,
                architecture=arch,
                supports_vision=supports_vision,
                supports_native_tools=supports_native_tools,
                supports_thinking=supports_thinking,
                max_context_length=max_ctx,
            )
        except Exception as e:
            log.warning("[Brain(Ollama)] 能力探测失败（Ollama 可能未启动）: %s", e)
            return ModelCapabilities(backend="ollama", model_name=self.model_name)

    async def think_stream(self, text, speaker_info=None):
        messages, full_user_content = self._prepare_messages(text, speaker_info)
        log.info("[Brain(Ollama)] messages built (%d msgs): %s", len(messages), full_user_content[:80])

        # -- turn 级裁剪：与 GGUF 路径的 token 级裁剪对齐 --
        max_turns = get_config("brain.ollama.max_history_turns",
                               get_config("memory.session.max_turns", 14))
        
        # messages[0] 是 system，最后一条是当前 user，中间的是历史
        if len(messages) > max_turns * 2 + 1:
            system_msg = messages[0]
            current_user = messages[-1]
            history_turns = messages[1:-1]
            # 保留最新的 max_turns 对
            history_turns = history_turns[-(max_turns * 2):]
            messages = [system_msg] + history_turns + [current_user]
            log.debug("[Brain(Ollama)] 裁剪历史到 %d 对", max_turns)

        # -- TODO：思考模式 + 预算提示 --------------------------
        enable_thinking = bool(get_config("brain.ollama.enable_thinking", False))
        think_budget = int(get_config("brain.ollama.think_budget", 0))
        if enable_thinking and think_budget > 0 and messages and messages[0]['role'] == 'system':
            budget_hint = (f"\n\n【思考约束】请将内部推理控制在 {think_budget} 字符以内，"
                           "必须在预算内用</think>结束，不要超出"
                           "</think>之后的回复将直接发送给用户")
            messages[0] = {
                'role': 'system',
                'content': messages[0]['content'] + budget_hint,
            }

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                temperature=get_config("brain.ollama.temperature", 0.7)
            )
        except Exception as e:
            log.error("[Brain(Ollama)] 连接失败: %s", e)
            yield "(panic)[glitch]大脑连接好像出错了"
            return

        think_filter = _ThinkFilter( # 已弃用！！！
            start_in_think=enable_thinking,
        )
        think_buf = []

        async for chunk in response:
            if chunk.choices[0].delta.content:
                visible, think = think_filter.feed(chunk.choices[0].delta.content)
                if think:
                    think_buf.append(think)
                if visible:
                    yield visible

        leftover, _ = think_filter.flush()
        if leftover:
            yield leftover

        if think_buf and enable_thinking:
            log.debug("[Brain.Think]\n%s", ''.join(think_buf).strip())