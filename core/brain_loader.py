'''
GGUF 后端 Brain（通过 llama-cpp-python 直接运行本地 GGUF 模型）
共享逻辑（历史、记忆、commit_response 等）全部继承自 BrainBase
'''

import asyncio
import os
import time
import jinja2
import llama_cpp
from llama_cpp import Llama
from concurrent.futures import ThreadPoolExecutor
from utils.config_loader import get_config
from utils.helpers import resolve_path
from utils.logger import get_logger
from core.brain_base import BrainBase
from core.brain_state import BrainStateMachine
from utils.benchmark import bench

log = get_logger(__name__)



def _apply_chat_template(llm: Llama, messages: list, enable_thinking: bool, external_template_path: str | None = None, think_budget: int = 0, tools: list | None = None) -> str | None:
    try:
        # 外部 jinja 文本 
        template_str = ""
        if external_template_path and os.path.exists(external_template_path):
            with open(external_template_path, 'r', encoding='utf-8') as f:
                template_str = f.read()
            log.debug("[Brain] 成功加载外部 Jinja 模板: %s", external_template_path)
        else:
            # 退回
            template_str = llm.metadata.get('tokenizer.chat_template', '')
        
        if not template_str:
            return None

        eos_token = '<|im_end|>'
        bos_token = ''
        try:
            eos_id = int(llm.metadata.get('tokenizer.ggml.eos_token_id', -1))
            if eos_id >= 0:
                raw = llm.detokenize([eos_id])
                decoded = raw.decode('utf-8', errors='ignore').strip()
                if decoded:
                    eos_token = decoded
        except Exception:
            pass

        env = jinja2.Environment(
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        env.globals['raise_exception'] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
        template = env.from_string(template_str)
        return template.render(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            think_budget=think_budget,
            tools=tools or [],
            eos_token=eos_token,
            bos_token=bos_token,
        )
    except Exception as e:
        log.warning("[Brain] Jinja2 模板渲染失败: %s", e)
        return None


class GGufBrain(BrainBase):
    ''''''
    def __init__(self, model_path=None, memory_manager=None):
        self._init_shared(memory_manager)

        default_path = get_config("brain.gguf.model_path", r"assets\model\Monika-v3.gguf")
        self.model_path = model_path or resolve_path(default_path)

        self.enable_thinking = bool(get_config("brain.gguf.enable_thinking", False))
        self._tools_enabled = bool(get_config("tools.enabled", False))

        log.info("[Brain] 正在初始化 GGUF 核心: %s", self.model_path)
        try:
            _n_threads = get_config("brain.gguf.n_threads", 0)
            if _n_threads <= 0:
                import os as _os
                _n_threads = max(1, (_os.cpu_count() or 4) - 1)  # 留一个给系统
            _flash = bool(get_config("brain.gguf.flash_attn", False))
            _kv_q = int(get_config("brain.gguf.kv_quant", 0))  # 0=f16, 8=q8_0
            _llama_kwargs = dict(
                model_path=self.model_path,
                n_gpu_layers=get_config("brain.gguf.n_gpu_layers", -1),
                n_ctx=get_config("brain.gguf.n_ctx", 2048),
                n_batch=get_config("brain.gguf.n_batch", 512),
                n_threads=_n_threads,
                verbose=False,
                flash_attn=_flash,
            )
            if _kv_q == 8:
                _llama_kwargs["type_k"] = 8   # GGML_TYPE_Q8_0
                _llama_kwargs["type_v"] = 8
            self.llm = Llama(**_llama_kwargs)
            log.info("[Brain] GGUF 核心加载成功：GPU 加速已激活")
            # 启用 prompt 前缀缓存：ReAct 多轮 + 跨轮对话中相同前缀自动复用 KV
            # 这是 llama-cpp 官方高层缓存，内部自动管理，零 crash 风险
            if get_config("brain.gguf.enable_cache", True):
                try:
                    from llama_cpp import LlamaCache
                    self.llm.set_cache(LlamaCache(capacity_bytes=int(get_config("brain.gguf.cache_size", 2) * (1<<30))))
                    log.info("[Brain] KV 前缀缓存已启用 (auto-managed)")
                except Exception as _ce:
                    log.debug("[Brain] KV 缓存启用失败，跳过: %s", _ce)
        except Exception as e:
            log.error("[Brain] 初始化失败: %s", e)
            raise e

        self.executor = ThreadPoolExecutor(max_workers=1)

    async def get_capabilities(self):
        '''从 GGUF 文件元数据中探测模型能力'''
        from core.interfaces import ModelCapabilities

        _VISION_ARCHS = {"llava", "qwen2_vl", "obsidian", "moondream", "minicpmv", "mllama"}

        meta = self.llm.metadata
        arch = meta.get("general.architecture", "unknown")

        ctx_key = f"{arch}.context_length"
        try:
            max_ctx = int(meta.get(ctx_key, 4096))
        except (ValueError, TypeError):
            max_ctx = 4096

        supports_vision = arch.lower() in _VISION_ARCHS

        chat_template = meta.get("tokenizer.chat_template", "")
        supports_native_tools = (
            "tool_call" in chat_template
            or "{{.Tools}}" in chat_template
            or '"tools"' in chat_template
        )
        supports_thinking = "enable_thinking" in chat_template

        return ModelCapabilities(
            backend="gguf",
            model_name=os.path.basename(self.model_path),
            architecture=arch,
            supports_vision=supports_vision,
            supports_native_tools=supports_native_tools,
            supports_thinking=supports_thinking,
            max_context_length=max_ctx,
        )

    async def think_stream(self, text, speaker_info=None):
        '''GGUF 流式生成回复
        生成器 yield 字符串片段，直到生成结束或被中断
        '''
        bench.turn_start()
        think_enabled = self.enable_thinking

        # trait 更新（用真实时间差替代常数 1.0）
        _now = time.time()
        _dt = _now - getattr(self.trait_manager, '_last_tick', _now)
        if _dt > 0:
            self.trait_manager.tick(_dt)
        self.trait_manager.update("user_spoke", {"text": text})
        self.trait_manager.save()
        # 检查 trait 触发（主动行为），由外部消费
        for _t in self.trait_manager.check_triggers():
            log.debug("[Trait] 触发: %s (urgency=%.2f, source=%s)",
                      _t.action, _t.urgency, _t.trait_source)

        # -- RL: 本轮开始（下沉到 Brain 层，终端/GUI 共用） --
        _collector = getattr(self, 'collector', None)
        if _collector:
            try:
                _collector.on_turn_start(
                    user_text=text,
                    trait_snapshot=self.trait_manager.snapshot().__dict__,
                    speaker_name=speaker_info.get('name', '') if speaker_info else '',
                )
            except Exception:
                pass

        messages, full_user_content = self._prepare_messages(text, speaker_info)

        # -- RL 策略决策（闭包捕获，_generate 内使用） --
        _rl_decision = None
        if get_config("rl.policy.enabled", False):
            try:
                from rl.engine import PolicyEngine
                from rl.features import build_features
                if not hasattr(self, '_policy_engine'):
                    self._policy_engine = PolicyEngine()
                    self._policy_engine.try_load()
                if self._policy_engine.is_ready:
                    _snap = self.trait_manager.snapshot()
                    _feats = build_features(
                        trait_snapshot={"mood": _snap.mood, "loneliness": _snap.loneliness,
                                        "curiosity": _snap.curiosity, "affection": _snap.affection},
                        turn_count=getattr(self, '_turn_count', 0))
                    _rl_decision = self._policy_engine.decide(_feats)
                    log.debug("[RL] 决策: temp=%.2f max_tokens=%.0f",
                              _rl_decision["temperature"], _rl_decision["max_tokens"])
            except Exception as e:
                log.debug("[RL] 策略决策跳过: %s", e)

        # 思考模式：模型通过 Jinja 模板中的软限制然管理思考内容
        # think_budget > 0时模板已注入字符约束，此处不再追加冗余提示。= 0意味着不限制思考长度
        log.debug("[Brain(GGUF)] messages built (%d msgs): %s",
                 len(messages), full_user_content[:80])

        _think_budget = int(get_config("brain.gguf.think_budget", 0))  # 模板注入
        _pending_tools = getattr(self, '_pending_tools', None)
        if _pending_tools:
            log.debug("[Brain] 工具定义已注入: %d 个工具 (%s)",
                     len(_pending_tools),
                     ", ".join(t["function"]["name"] for t in _pending_tools))
        else:
            log.debug("[Brain] 无工具定义（tools.enabled=false 或未注册）")

        _current_gen = [0]

        def _generate():
            my_gen = _current_gen[0]
            n_ctx = get_config("brain.gguf.n_ctx", 2048)
            total_max_tokens = get_config("brain.gguf.max_tokens", 1024) if not think_enabled else 0 # 思考模式不限制 max_tokens，由剩余上下文空间决定

            gen_kwargs = dict(
                stream=True,
                max_tokens=total_max_tokens,
                temperature=get_config("brain.gguf.temperature", 0.85),
                top_p=get_config("brain.gguf.top_p", 0.95),
                top_k=int(get_config("brain.gguf.top_k", 20)),
                repeat_penalty=get_config("brain.gguf.repeat_penalty", 1.1),
                # 思考模式下不设 stop token：模型可能在 THINK 中途误输出 <|im_end|>
                # 导致生成被截断，</think> 来不及写就死了
                # TODO： 实际上llama cpp会自己加 EOS，很烦
                stop=None if think_enabled else ["<|im_end|>", "<|endoftext|>"],
            )

            # -- RL 残差覆盖：基线 + RL delta（仅覆盖温度与 max_tokens） --
            if _rl_decision is not None:
                from rl.engine import PolicyEngine as _PE
                _bl_temp = get_config("brain.gguf.temperature", 0.85)
                _bl_tokens = gen_kwargs["max_tokens"] or 512
                gen_kwargs["temperature"] = _PE.apply_residual(
                    _bl_temp, _rl_decision["temperature"], 0.5, 1.5)
                gen_kwargs["max_tokens"] = int(_PE.apply_residual(
                    _bl_tokens, _rl_decision["max_tokens"], 128, 1024))

            token_budget = n_ctx - (total_max_tokens or 0)

            trimmed = list(messages)
            if get_config("brain.gguf.external_template", None)!=None:
                template_path = resolve_path(get_config("brain.gguf.external_template"))
            else:
                template_path = None

            while True:
                prompt = _apply_chat_template(self.llm, trimmed, think_enabled,
                                              external_template_path=template_path,
                                              think_budget=_think_budget,
                                              tools=_pending_tools)
                if prompt is None:
                    log.error("[Brain] 模板渲染失败，无法生成")
                    loop.call_soon_threadsafe(
                        lambda: q.put_nowait(("ERROR", RuntimeError("模板渲染失败"))))
                    return
                token_count = len(self.llm.tokenize(
                    prompt.encode('utf-8'), add_bos=False))
                if token_count <= token_budget: # 关闭思考模式时，允许 prompt 占满上下文剩余空间
                    break
                cut_idx = None
                # 优先裁剪 tool 消息（最长且可丢弃），保护 user/assistant 对话结构
                for i in range(1, len(trimmed) - 1):
                    if trimmed[i]['role'] == 'tool':
                        cut_idx = i
                        break
                if cut_idx is None:
                    for i in range(1, len(trimmed) - 1):
                        if trimmed[i]['role'] in ('user', 'assistant'):
                            cut_idx = i
                            break
                if cut_idx is None:
                    log.warning("[Brain] 单条消息已超出上下文 (%d/%d)，截断到预算",
                                token_count, token_budget)
                    break
                removed = trimmed.pop(cut_idx)
                log.debug("[Brain] 上下文过长 (%d>%d)，裁剪最旧: [%s] %s",
                          token_count, token_budget, removed['role'],
                          str(removed['content'])[:30])

            # 如果 total_max_tokens 为 0（思考模式）：
            # 首次生成用 think_budget 限制 think 长度，超了触发续写
            # 续写时（messages 里已有 think）不限 max_tokens，让模型自然结束
            if total_max_tokens <= 0:
                _has_think_in_ctx = any(
                    '<think>' in m.get('content', '') for m in messages
                    if m.get('role') == 'assistant'
                )
                if _think_budget > 0 and not _has_think_in_ctx:
                    # 首次生成：think_budget 限制（token 近似 = 字符数 / 1.5）
                    final_max_tokens = max(int(_think_budget / 1.5), 64)
                else:
                    # 续写或无限 think：用剩余上下文空间
                    final_max_tokens = max(n_ctx - token_count, 1)
                gen_kwargs['max_tokens'] = final_max_tokens
                log.debug("[Brain] 思考模式 max_tokens = %d (think_budget=%d, 续写=%s)",
                          final_max_tokens, _think_budget, _has_think_in_ctx)

            from core.event import emit as _emit_token
            # 每次生成开始加分隔（两次输出之间换行）
            _emit_token("\n\n<<< GENERATION >>>\n")

            raw_stream = self.llm.create_completion(prompt=prompt, **gen_kwargs)

            loop.call_soon_threadsafe(
                lambda: q.put_nowait(("CONFIG", think_enabled)))

            chunk_count = 0
            total_text_len = 0
            try:
                for chunk in raw_stream:
                    if _current_gen[0] != my_gen:
                        log.debug("[Brain] 生成被取代 (gen %d->%d)，提前退出",
                                  my_gen, _current_gen[0])
                        return
                    chunk_count += 1
                    choices = chunk.get('choices', [])
                    if choices:
                        t = choices[0].get('text', '')
                        if t:
                            total_text_len += len(t)
                            _emit_token(t)
                    loop.call_soon_threadsafe(
                        lambda c=chunk: q.put_nowait(("DATA", c)))
            except Exception as e:
                log.debug("[Brain] 生成异常 (gen %d, chunks=%d): %s",
                          my_gen, chunk_count, e)
                loop.call_soon_threadsafe(
                    lambda e=e: q.put_nowait(("ERROR", e)))
            finally:
                log.debug("[Brain] 生成结束 (gen %d, chunks=%d, text_len=%d)",
                          my_gen, chunk_count, total_text_len)
                loop.call_soon_threadsafe(
                    lambda: q.put_nowait(("DONE", None)))

        loop = asyncio.get_event_loop()
        q = asyncio.Queue()
        self.executor.submit(_generate)

        msg_type, must_filter = await q.get()
        if msg_type != "CONFIG":
            log.error("[Brain] 生成异常，期望 CONFIG 收到 %s", msg_type)

        _last_talk = []

        _MAX_TOOL_RESULT_CHARS = 2000

        async def _tool_executor(name: str, params: dict):
            from core.tools.tool_registry import get_tool_registry
            from core.tools.base import ToolPermission, ToolResult
            registry = get_tool_registry()
            tool = registry.get(name)

            # 工具分类（权限）
            if tool and tool.permission in (ToolPermission.CONFIRM, ToolPermission.ALWAYS_ASK):
                if self._confirm_request:
                    self._confirm_info = f"{name}: {str(params)}"
                    self._confirm_result = False
                    self._confirm_request.clear()
                    log.info("[Brain] 等待用户确认: %s", self._confirm_info)
                    await self._confirm_request.wait()
                    if not self._confirm_result:
                        log.info("[Brain] 用户已拒绝: %s", name)
                        # 拒绝分支
                        self.trait_manager.update("tool_called", {"name": name})
                        bench.record("tool_calls", 1)
                        bench.record("tool_fail", 1)
                        self.trait_manager.update("tool_failure")
                        result = ToolResult(False, error="用户拒绝了此操作，请改用其他方式回复")
                        # 注入 assistant talk + tool result（同正常流程）
                        talk_text = ''.join(_last_talk) if _last_talk else ""
                        if talk_text:
                            messages.append({"role": "assistant", "content": talk_text})
                            self.history.append({"role": "assistant", "content": talk_text})
                            _last_talk.clear()
                        messages.append({"role": "tool", "content": "[用户拒绝了此操作，可能用户认为你的工具使用场景或方式不合理]"})
                        self.history.append({"role": "tool", "content": "[用户拒绝了此操作]"})
                        return result
                    log.info("[Brain] 用户已确认: %s", name)

            self.trait_manager.update("tool_called", {"name": name})
            result = await registry.execute(name, params)
            bench.record("tool_calls", 1)
            if result.success:
                bench.record("tool_success", 1)
                self.trait_manager.update("tool_success")
            else:
                bench.record("tool_fail", 1)
                self.trait_manager.update("tool_failure")
            status = "OK" if result.success else f"ERR: {result.error}"
            preview = (result.content or "")[:200].replace('\n', ' ')
            log.info("[Brain] Tool result [%s]: %s | %s", name, status, preview)
            talk_text = ''.join(_last_talk) if _last_talk else ""
            if talk_text:
                messages.append({"role": "assistant", "content": talk_text})
                self.history.append({"role": "assistant", "content": talk_text})
                _last_talk.clear()
            tool_content = result.content or result.error or ""
            if not result.success:
                tool_content = f"[工具调用失败] {tool_content}"
            if len(tool_content) > _MAX_TOOL_RESULT_CHARS:
                tool_content = tool_content[:_MAX_TOOL_RESULT_CHARS] + (
                    f"\n\n... [截断，原长度 {len(tool_content)} 字符]")
            messages.append({"role": "tool", "content": tool_content})
            self.history.append({"role": "tool", "content": tool_content})
            return result

        sm = BrainStateMachine(
            enable_thinking=must_filter,
            on_tool_call=_tool_executor,
        )

        async def _q_to_tokens():
            while True:
                mt, chunk = await q.get()
                if mt == "DONE" or mt == "ERROR":
                    return
                if mt != "DATA":
                    continue
                choices = chunk.get('choices', [])
                if choices:
                    t = choices[0].get('text', '')
                    if t:
                        yield t

        _need_wait = self._tts_ready is not None

        # -- Benchmark: 思考/生成阶段计时 ----------------------
        bench.phase_start("think")
        _first_talk = True

        async for talk_text in sm.process(_q_to_tokens()):
            if _first_talk:
                _first_talk = False
                bench.phase_end("think", "think_ms")
                bench.phase_start("talk")
            _last_talk.append(talk_text)
            yield talk_text

            if _need_wait and talk_text.strip():
                self._tts_ready.clear()
                await self._tts_ready.wait()

        # -- Benchmark: 关闭本次生成阶段计时 ------------------
        if _first_talk:
            bench.phase_end("think", "think_ms")
            bench.record("think_truncated", True)

        # -- ReAct 循环：模型输出 tool_call -> 执行 -> 结果注入 -> 重新生成 --
        _max_rounds = int(get_config("tools.max_tool_calls_per_turn", 5))
        _round = 0
        _last_tool_sig = None  # (name, frozenset(params.items())) 重复检测

        while _round < _max_rounds:
            pending = sm.get_pending_tool()
            if not pending:
                break

            name, params = pending
            sig = (name, frozenset(params.items()))

            # -- 重复工具调用检测：相同 tool和 params 视为连续调用，截断 --
            if sig == _last_tool_sig:
                log.warning("[Brain] [WARN] 重复工具调用 %s(%s)，截断循环",
                           name, {k: str(v)[:50] for k, v in params.items()})
                messages.append({
                    "role": "tool",
                    "content": "[系统] 已跳过重复调用（参数与上一轮完全相同）"
                })
                self.history.append({
                    "role": "tool",
                    "content": "[系统] 已跳过重复调用"
                })
                break
            _last_tool_sig = sig

            _round += 1
            log.info("[Brain] [Tool] round %d/%d: %s(%s)", _round, _max_rounds, name, params)
            await _tool_executor(name, params)

            _current_gen[0] += 1
            q = asyncio.Queue()
            self.executor.submit(_generate)
            # 排空到 CONFIG，丢弃中途生成结果（如果有）
            while True:
                _drain = await q.get()
                if _drain[0] == "CONFIG":
                    break

            sm.reset()
            _last_talk.clear()
            self._sentence_new_block = True  # 新 TALK 块，下一句开新段落
            async for talk_text in sm.process(_q_to_tokens()):
                _last_talk.append(talk_text)
                yield talk_text
                if _need_wait and talk_text.strip():
                    self._tts_ready.clear()
                    await self._tts_ready.wait()

        if _round >= _max_rounds:
            pending = sm.get_pending_tool()
            if pending:
                log.warning("[Brain] [WARN] 达到最大工具调用轮次 %d，丢弃待处理工具: %s",
                           _max_rounds, pending[0])

        # -- 安全兜底：工具调用后无 TALK 输出 ------------------
        if _round > 0 and not _last_talk:
            log.warning("[Brain] [WARN] 执行了 %d 轮工具调用但模型未生成任何回复文本", _round)

        remaining = await sm.flush_remaining()
        if remaining:
            yield remaining

        # -- THINK 截断续写：流在 THINK 状态结束时，保留 think 内容续写 TALK --
        _think_remnant = getattr(sm, 'think_remnant', '')
        if _think_remnant and not _last_talk:
            log.info("[Brain] THINK 截断检测，注入未闭合 think 续写 (%d chars)",
                     len(_think_remnant))
            # 把未闭合的 think 作为 assistant 消息注入（故意不闭合 </think>）
            messages.append({"role": "assistant", "content": f"<think>{_think_remnant}"})
            # 重新生成，模型看到未闭合的 <think> 会续写 </think> + TALK
            _current_gen[0] += 1
            q = asyncio.Queue()
            self.executor.submit(_generate)
            while True:
                _drain = await q.get()
                if _drain[0] == "CONFIG":
                    break
            sm.reset()
            _last_talk.clear()
            self._sentence_new_block = True
            async for talk_text in sm.process(_q_to_tokens()):
                _last_talk.append(talk_text)
                yield talk_text
                if _need_wait and talk_text.strip():
                    self._tts_ready.clear()
                    await self._tts_ready.wait()

        # -- Benchmark: 提交本轮数据 ----------------------------
        spkr = speaker_info.get("name", "") if speaker_info else ""
        bench.turn_end()
        bench.commit(speaker_name=spkr)

        # -- RL: 本轮结束（下沉到 Brain 层，终端/GUI 共用） --
        if _collector:
            try:
                _bt = bench._turns[-1] if bench._turns else None
                _all_talk = ''.join(_last_talk) if _last_talk else ""
                _collector.on_turn_end(
                    assistant_text=_all_talk,
                    interrupted=False,  # 打断状态由调用方传入，这里默认 False
                    tool_calls=int(_bt.tool_calls) if _bt else 0,
                    tool_success=int(_bt.tool_success) if _bt else 0,
                    tool_fail=int(_bt.tool_fail) if _bt else 0,
                    duration_ms=float(_bt.total_ms) if _bt else 0.0,
                )
            except Exception:
                pass
