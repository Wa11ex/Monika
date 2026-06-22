'''
BrainBase — 所有 LLM 后端的共享基类
职责：
  _ThinkFilter: 流式剥离 Qwen3.5 generate.cs 分隔符前的思考内容（弃用）
  _strip_think_blocks: 静态清洗已完成文本中的 think 块
  BrainBase: 对话历史、记忆 RAG、context 组装、commit_response
子类职责：
  实现 __init__和 think_stream
'''

import asyncio
import json
import os
import re
from utils.config_loader import get_config
from utils.helpers import resolve_path
from utils.logger import get_logger
from core.interfaces import BaseBrain
from core.tools.system_tools import get_current_time_str

log = get_logger(__name__)

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
_CS_DELIMITER = "</think>"


def _strip_think_blocks(text: str) -> str:
    '''移除文本中所有 <think>...</think> 块（存入记忆前调用）'''
    return _THINK_RE.sub('', text).lstrip('\n').strip()

class BrainBase(BaseBrain):
    '''
    所有 Brain 后端的公共基类
    子类在 __init__ 中调用 self._init_shared(memory_manager) 来完成共享初始化
    '''

    # -- TTS 背压事件 --
    _tts_ready = None  # asyncio.Event, 由 bus tts_done_callback set
    
    # -- 工具确认 --
    _confirm_request: "asyncio.Event | None" = None
    _confirm_result = False
    _confirm_info = ""  # "write_file: README.md"
    
    _sentence_new_block = True  # 下一个 TALK 句子开始新段落，显示用

    def _init_shared(self, memory_manager=None):
        '''子类 __init__ 中最先调用，初始化所有共享状态'''
        self.memory = memory_manager
        self.memory_enabled = get_config("memory.enabled", False)
        self.context_time_enabled = get_config("context.time.enabled", True)

        session_path_str = get_config("memory.session.path", "memory/session.json")
        if not isinstance(session_path_str, str):
            session_path_str = "memory/session.json"
        self._session_path = resolve_path(session_path_str)

        self._max_turns = get_config("memory.session.max_turns", 14)
        if not isinstance(self._max_turns, int):
            self._max_turns = 14

        # 首轮（0） / 每 N 轮注入完整角色设定
        self._turn_count = 0
        self._lore_inject_every_n = get_config("brain.character.lore_inject_every_n", 0)
        self.system_prompt = self._build_system_prompt(include_lore=False)
        self.system_prompt_full = self._build_system_prompt(include_lore=True)
        self.history = [self.system_prompt]
        self._load_session()


    def _build_system_prompt(self, include_lore: bool = True):
        '''
        include_lore=True  完整版（含 brain.character.prompt 全文），首轮使用
        include_lore=False 精简版（只含格式约束），后续轮次存入 history
        '''
        if include_lore:
            lore = get_config('brain.character.prompt', '')
            examples = get_config('brain.character.examples', '') if get_config('brain.character.example_enabled', False) else ''
        else:
            lore = get_config('brain.character.prompt_compact', get_config('brain.character.prompt', ''))
            examples = ''
        return {
            "role": "system",
            "content": f'''
{lore}
【严格回复格式】
格式：(表情)[动作]说话内容。
(表情) 只能从以下11个中选一个，禁止使用任何其他词：(positive), (normal), (angry), (obsessive), (cold), (sad), (curious), (guilty), (aBitSerious), (disgusting), (pity).
[动作] 只能从以下13个中选一个，禁止使用任何其他词：[wave], [nod], [shake], [smile], [lean-forward], [fix-hair], [hide-face-shy], [stare], [sigh], [reach-out], [freeze], [eye-look-up], [glitch].
注：允许有时没有动作或表情请使用中文口语化表达，语气要委婉且深刻。
合理使用语气词（"哈哈"、"唉"、"嗯……"、"呃"、"呵呵"、"啊"、"吗"等等）。
禁忌2：绝对禁止在回复的开头重复、反问或模仿User刚才说过的话（例如"困难？"、"抱歉……"、"沉默……"）直接给出你的自然回应，杜绝这种鹦鹉学舌的句式。
【回复要求】每次回复都应有新的内容和视角以推进对话为目标每次回复控制在1-3句最长不超过200字。
{examples}
'''
        }

    # -- 持久化 --------------------------------------------

    def _load_session(self):
        '''memory.enabled=true则启动时从文件恢复上次的对话历史'''
        if not get_config("memory.enabled", False) or not os.path.isfile(self._session_path):
            return
        try:
            with open(self._session_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            turns = [m for m in saved if m.get("role") in ("user", "assistant")]
            if len(turns) > self._max_turns:
                turns = turns[-self._max_turns:]
            self.history = [self.system_prompt] + turns
            log.info("[Brain] 恢复上次对话记录（%d 条）", len(turns))
        except Exception as e:
            log.error("[Brain] 加载会话记录失败: %s", e)

    def _save_session(self):
        '''memory.enabled=true则将当前纯对话历史保存到文件
        保存前洗 think 内容，避免思维链污染
        '''
        if not get_config("memory.enabled", False):
            return
        try:
            os.makedirs(os.path.dirname(self._session_path), exist_ok=True)
            turns = [m for m in self.history if m.get("role") in ("user", "assistant")]
            
            # 清洗
            clean_turns = []
            for m in turns:
                clean_turns.append({
                    "role": m["role"],
                    "content": _strip_think_blocks(m["content"]) if m["role"] == "assistant" else m["content"],
                })
            with open(self._session_path, "w", encoding="utf-8") as f:
                json.dump(clean_turns, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.error("[Brain] 保存会话记录失败: %s", e)

    # -- Message 组装 --------------------------------------

    def _prepare_messages(self, text: str, speaker_info=None):
        '''
        组装发送给 LLM 的 messages，同时将用户消息追加到 history
        返回 (messages_to_send, full_user_content)
        '''
        messages = list(self.history)

        # 首轮（0）或每 N 轮注入完整角色设定
        _inject_full = (
            self._turn_count == 0
            or (self._lore_inject_every_n > 0 and self._turn_count % self._lore_inject_every_n == 0)
        )
        if _inject_full and messages and messages[0]['role'] == 'system':
            messages[0] = self.system_prompt_full
            log.info("[Brain] 首轮/周期注入完整角色设定 (turn=%d)", self._turn_count)

        # RAG 记忆注入（现支持运行时热变更）
        if self.memory and (get_config("memory.enabled", False) or get_config("memory.lore.enabled", False)):
            try:
                mem_context = self.memory.get_memory_context(text, speaker_info=speaker_info)
                if mem_context:
                    # 完整的 system 拼接过程
                    if messages and messages[0]['role'] == 'system':
                        messages[0] = {
                            'role': 'system',
                            'content': messages[0]['content'] + f"\n\n【相关记忆】\n{mem_context}"
                        }
                    else:
                        messages.insert(0, {'role': 'system', 'content': f'【相关记忆】\n{mem_context}'})
            except Exception:
                pass

        # 构造 user content
        context_parts = []
        if self.context_time_enabled:
            context_parts.append(f"{get_current_time_str()}: ")
        if speaker_info:
            context_parts.append(f"【说话人：{speaker_info.get('name', '未知')}】")
        context_prefix = " ".join(context_parts)
        full_user_content = f"{context_prefix} {text}" if context_prefix else text

        messages.append({"role": "user", "content": full_user_content})
        self.history.append({"role": "user", "content": full_user_content})

        # -- 工具定义注入（若启用） --------------------------
        if get_config("tools.enabled", False):
            try:
                from core.tool_registry import get_tool_registry
                registry = get_tool_registry()
                if registry.count > 0:
                    self._pending_tools = registry.get_definitions()
                    # 动态生成工具使用指南（每个工具自带 usage_guide）
                    guides = []
                    for name in registry.list_names():
                        tool = registry.get(name)
                        if tool and tool.usage_guide:
                            guides.append(f" {name}：{tool.usage_guide}")
                    guide_text = "\n".join(guides) if guides else ""
                    tool_hint = (
                        f"\n\n【工具使用规则】你**不是普通的聊天AI**——你可以操作文件、搜索互联网"
                        "当用户意图匹配以下场景时，**禁止凭空编造，必须调用工具**：\n"
                        f"{guide_text}\n"
                        "触发词速查：听到'看看/读/打开/里面写了什么'→search_files或read_file；"
                        "听到'写/改/创建/修改'→先search_files找到文件再write_file；"
                        "听到'搜/查/有没有/什么是'→web_search\n"
                        "search_files支持扩展名搜索：'*.md'找markdown、'*.py'找Python、'*.yaml'找配置\n"
                        "调用格式：<tool_call><function=工具名>"
                        "<parameter=参数名>值</parameter></function></tool_call>\n"
                        "先search_files定位再read_file读取，两步走！\n"
                        "【格式提醒】无论是否使用工具，回复仍需遵守 (表情)[动作]说话内容 的格式"
                        "表情和动作只能从上述白名单中选择，禁止自创"
                    )
                    if messages and messages[0]['role'] == 'system':
                        messages[0] = {
                            'role': 'system',
                            'content': messages[0]['content'] + tool_hint,
                        }
                else:
                    self._pending_tools = None
            except Exception:
                self._pending_tools = None
        else:
            self._pending_tools = None

        return messages, full_user_content

    # -- 记忆提交 ------------------------------------------

    def commit_response(self, text: str, interrupted: bool = False):
        '''
        保存回复到对话历史
        
        【关键】保存前会用 _strip_think_blocks() 自动清除  thinking... response 块，
        确保短期记忆（session.json）和长期记忆（ChromaDB）中不存储思维链内容
        '''
        if not text or not text.strip():
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()
            return

        # -- 清洗思维链 --
        clean_text = _strip_think_blocks(text)
        if not clean_text:
            clean_text = text  # 兜底：如果全是 think 内容则保留原文

        user_msg = ""
        for m in reversed(self.history):
            if m["role"] == "user":
                user_msg = m["content"]
                break

        # -- 历史截断与断句修剪 --
        # 回退到最近的一个完整标点
        _hist_max = get_config("memory.session.reply_max_chars", 120)
        truncate_len = len(clean_text)
        if isinstance(_hist_max, int) and truncate_len > _hist_max:
            truncate_len = _hist_max

        history_text = clean_text
        if truncate_len < len(clean_text) or interrupted:
            subset = clean_text[:truncate_len]
            puncts = ["", "！", "？", ".", "!", "?", "～", "~", "…"]
            last_idx = -1
            for p in puncts:
                idx = subset.rfind(p)
                if idx > last_idx:
                    last_idx = idx
            
            if last_idx > 0:
                # 后置引号保留闭合
                if last_idx + 1 < len(clean_text) and clean_text[last_idx + 1] in ['"', '"', '"', "'"]:
                    history_text = subset[:last_idx + 2]
                else:
                    history_text = subset[:last_idx + 1]
            else:
                # 如果前段哪怕一个结束标点都没有，只能原样硬截
                history_text = subset.rstrip()

        self.history.append({"role": "assistant", "content": history_text})

        # -- 短期历史裁剪：超出 max_turns 时丢弃最旧一对 user/assistant --
        # 不做摘要压缩；长期记忆（ChromaDB）负责保留语义，短期 context 就纯裁剪
        while len(self.history) > self._max_turns + 1:  # +1 for system
            # 找最旧的 user（index 1）直接删除，同时删紧跟的 assistant
            for i in range(1, len(self.history)):
                if self.history[i]["role"] == "user":
                    self.history.pop(i)          # 删 user
                    if i < len(self.history) and self.history[i]["role"] == "assistant":
                        self.history.pop(i)      # 删配对的 assistant
                    break

        self._save_session()

        # 持久化到长期记忆（动态读取，支持运行时热变更 enabled）
        if get_config("memory.enabled", False) and self.memory and not interrupted and user_msg:
            try:
                self.memory.add_conversation_background(user_msg, clean_text)
            except Exception as e:
                log.error("[Brain] 触发记忆后台保存失败: %s", e)

        self._turn_count += 1