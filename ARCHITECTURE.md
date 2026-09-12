# Monika — Architecture & Design Notes

> 关键设计决策、数据流、模块交互的参考文档。

---

## 数据流全景

```
┌----------┐    ┌----------┐    ┌-----------┐    ┌--------┐    ┌----------┐
│ 麦克风/键盘 │ → │ Perception │ → │  Brain 后端  │ → │  Bus   │ → │ 扬声器    │
│ (Ears)    │    │ (事件统一)  │    │ GGUF/Ollama│    │ (编排)  │    │(AudioPlayer)│
└----------┘    └----------┘    └-----┬------┘    └---┬----┘    └----------┘
                                      │               │
                                 ┌----┴----┐     ┌----┴------------┐
                                 │ Memory  │     │ Live2D / VTS     │
                                 │(ChromaDB)│     │ (表情+口型同步)  │
                                 └---------┘     └-----------------┘
```

### 关键路径

1. **语音输入**：`Ears.listen_auto()` → VAD → ASR → 文本
2. **RAG 记忆**：`MemoryManager.get_memory_context()` → ChromaDB 检索 → system prompt
3. **LLM 推理**：`Brain.think_stream()` → Jinja 模板渲染 → llama-cpp / Ollama → 流式 token
4. **状态机分流**：Brain 状态机（THINK→TALK→TOOL/CODE）剥离标签、分流文本和工具调用
5. **TTS 合成**：`GPTSoVITSTTS.synthesize()` → HTTP 流式 → WAV
6. **播放+口型**：`AudioPlayer` 播放 → `analyze_wav_bytes` → Live2D Param*

### 启动顺序

```
gui.py / main.py
  ├- load_config("config.yaml")
  ├- launch_all()
  │   ├- 启动外部 apps（VTS / Voicemeeter）
  │   ├- 启动 Ollama serve（若配置）
  │   └- 启动 TTS Server → 返回 health-check Task（不阻塞）
  ├- [并行] Ears(SenseVoice) ← 约 31s
  ├- [并行] Brain(GGUF)       ← 约 3s
  ├- [并行] TTS health-wait   ← 约 30s
  └- 构造 Perception / PoStateMachine → 进入主循环
```

---

## 关键设计决策

### 1. 双 Brain 后端

| 决策 | 说明 |
|------|------|
| **为什么不合并** | GGUF 和 Ollama 的生成 API 完全不同：GGUF 用 llama-cpp 的 `create_completion` + 裸线程；Ollama 用 `AsyncOpenAI.chat.completions` + `async for` |
| **共享逻辑** | `BrainBase` 提供 history、memory RAG、system prompt、commit_response，两个子类只实现 `think_stream` |
| **导入策略** | `main.py` 用 `_create_brain()` 条件导入，避免同一命名空间的类名冲突 |

### 2. Jinja 模板外置

| 决策 | 说明 |
|------|------|
| **为什么外置** | llama-cpp 内嵌的 `tokenizer.chat_template` 可能不完整或有 bug；外置文件可独立调试和版本控制 |
| **加载优先级** | `external_template` 文件 > GGUF 内部 metadata |
| **think_budget 注入** | 模板中 `{% if think_budget > 0 %}` 在 `<think>` 内注入字符约束指令，模型主动遵守 |

### 3. 双层记忆

| 层级 | 存储 | 目的 |
|------|------|------|
| 短期 | `self.history` → `session.json` | 当前对话上下文，最大 N 轮纯裁剪 |
| 长期 | ChromaDB | 全量对话语义检索；Lore 角色记忆独立 collection |

**为什么不做摘要压缩**：当前采用纯裁剪 + 长期记忆语义召回。摘要压缩会让 AI "记住"压缩偏差，而语义召回能保持原文精度。

### 4. 状态机替代多层过滤器

| 决策 | 说明 |
|------|------|
| **之前** | `_ThinkFilter` + `ToolFormatAdapter` + `_strip_code_for_tts` 三层过滤器串行 |
| **问题** | Tool 前文本丢失、线程冲突、代码块进 TTS |
| **现在** | 单一状态机（THINK/TALK/TOOL/CODE），逐 token 处理，分割点在标识符之后 |
| **优势** | TALK→TOOL 时立即 yield 到 TTS、CODE 跳过、THINK 剥离，零泄漏 |

### 5. 关麦不中断思考

| 决策 | 说明 |
|------|------|
| **根因** | `_listen_task.cancel()` 的 CancelledError 沿 await 链级联到 `_process_input` → brain 生成中断 |
| **修复** | 处理中 `_is_processing=True` 时不 cancel `_listen_task`；等处理完自然退出 |

### 6. 陌生人敌意硬规则

| 决策 | 说明 |
|------|------|
| **触发条件** | speaker 名称包含 `"陌生人"` → 强制注入 "陌生人敌意" Lore |
| **为什么不用 `is_new`** | `is_new` 只在首次注册时为 True，后续同一陌生人不再触发 |
| **优先级** | 硬规则 Lore 以 `distance=0.0` 插在语义检索结果最前面 |

---

## 模块职责速查

| 模块 | 文件 | 职责 |
|------|------|------|
| **Brain** | `brain_base.py` | 对话历史、记忆注入、system prompt、commit_response |
| | `brain_loader.py` | GGUF 后端 + Jinja 模板渲染 |
| | `brain.py` | Ollama 后端 (OpenAI API) |
| **Ears** | `ears.py` | VAD、ASR (SenseVoiceSmall)、打断检测、声纹识别 |
| **Bus** | `bus.py` | 流式输入 → TTS 排队 → 播放调度 → 表情/动作触发 |
| **Memory** | `memory_manager.py` | ChromaDB、对话记忆检索、Lore 检索、用户画像 |
| **Perception** | `perception.py` | 统一输入事件 (Mic/Keyboard) |
| **Vision** | `vision/` | 摄像头/屏幕抓帧 + VLM 编码 → 文本注入 |
| **PO** | `event/po_manager.py` | 主动发言冷却窗口状态机 |
| **TTS** | `tts_client.py` | GPT-SoVITS HTTP 客户端 |
| **VTS** | `vts_client.py` | VTube Studio API 连接 |
| **Launcher** | `launcher.py` | 外部服务启动/健康检查/关闭 |
| **Config** | `config_loader.py` | YAML 配置读写（线程安全） |
| **Logger** | `logger.py` | 日志初始化 + Qt GUI 转发 |

---

## 配置键速查

| 键路径 | 默认值 | 用途 |
|--------|--------|------|
| `brain.backend` | `gguf` | LLM 后端选择 |
| `brain.gguf.enable_thinking` | `false` | GGUF 思维链开关 |
| `brain.gguf.think_budget` | `0` | 思维链最大字符数（0=不限） |
| `brain.gguf.external_template` | `""` | Jinja 模板路径 |
| `memory.lore.enabled` | `false` | 前世人设检索 |
| `memory.enabled` | `false` | 对话记忆检索 |
| `speaker_id.enabled` | `false` | 声纹识别 |
| `proactive_output.enabled` | `true` | 主动发言 |
| `ears.echo_cancellation.enable_deepfilter` | `false` | DeepFilterNet 回消 |
| `context.time.enabled` | `true` | 注入当前时间到 context |

---

### 7. 脑状态机设计（层次化栈模型）

```
栈底：THINK ←→ TALK（互斥根状态）
嵌套：TOOL 可被 THINK 或 TALK 包裹
      CODE 可被 TALK 包裹

Token 流 → while state != _S_DONE:
  _S_THINK:  检测 <tool_call> → push TOOL（嵌套，think 未退出）
              检测 </think> → 剥离 → 栈切换为 _S_TALK
  _S_TALK:   检测 <tool_call> → push TOOL → yield 已累积文本到 TTS
              检测 ``` → push CODE → yield 已累积文本
              否则 → 累积到 talk_parts（遇 \n 时 flush）
  _S_TOOL:   检测 </tool_call> → 解析 → 校验工具名 → pop 回上一状态
  _S_CODE:   检测 ``` → 丢弃 → pop 回 TALK
```

### 8. ReAct Agent 循环

```
think_stream():
  sm.process(token_stream)        # 首次生成
      ↓
  while round < max_rounds:       # ← 多步循环
      pending = sm.get_pending_tool()
      if not pending: break
      if 重复调用检测: break      # 相同 tool+params 连续调用 → 截断
      tool_result = ToolRegistry.execute(pending)
      结果截断到 2000 字符
      注入 tool_result 到 messages
      _generate() 重新生成
      sm.reset()
      sm.process(new_token_stream)
```

```
think_stream():
  sm.process(token_stream)        # 首次生成
      ↓
  while round < max_rounds:       # ← 多步循环（原 if → while）
      pending = sm.get_pending_tool()
      if not pending: break       # 模型决定停止工具调用
      tool_result = ToolRegistry.execute(pending)
      注入 tool_result 到 messages
      _generate() 重新生成
      sm.reset()
      sm.process(new_token_stream)  # 模型基于结果继续思考/说话/调用工具
```

**相比单步调用的改进**：
- `if pending:` → `while` 循环，支持连续多个工具调用
- 每轮执行后注入的 tool result 在 messages 中持续累积
- `tools.max_tool_calls_per_turn` (默认 5) 硬限制防止死循环
- 工具返回值超过 2000 字符自动截断，防止上下文溢出

---

### 9. 打断机制

```
用户开口 → ears VAD 检测 → bus.interrupt()
  ├- AudioPlayer.clear_and_stop()   # 停止播放
  ├- tts.cancel()                   # 取消合成
  ├- 清空 text_queue / audio_queue  # 丢弃待播句子
  └- 重置 Live2D 表情

process_stream_input 逐 chunk 检查 _interrupted → 立即退出
_watch_interrupt 在音频出声前不触发（避免思考期间误打断）
```

**已知限制**：工具执行期间不可打断（HTTP 请求无法安全取消）。
工具结果可能在打断后残留到下一轮 context。

---

## 目录结构

```
Monika/
├-- core/
│   ├-- brain_base.py        # LLM 基类 + 记忆注入 + 工具提示
│   ├-- brain_loader.py      # GGUF 后端 + 状态机 + Jinja 模板
│   ├-- brain_ollama.py      # Ollama 后端
│   ├-- bus.py               # 语音编排：分句→TTS→播放→Live2D
│   ├-- ears.py              # VAD + ASR (SenseVoiceSmall)
│   ├-- tool_registry.py     # 工具注册/执行单例
│   ├-- tool_format.py       # Tool call XML 解析器（Qwen/DeepSeek）
│   ├-- tools/               # 工具实现 (base, file_tools)
│   ├-- audio/               # 回声消除、口型同步、声纹识别
│   ├-- event/               # 事件总线 + traits + PO + token 流
│   ├-- vision/              # 视觉：抓帧 + VLM 编码 → 文本注入
│   └-- perception.py        # 输入感知 (Mic/Keyboard)
├-- memory/
│   └-- memory_manager.py    # ChromaDB + Lore 检索 + 用户画像
├-- ui/                      # PyQt6 GUI + Live2D 渲染
├-- utils/                   # 配置、日志、模型导入
└-- assets/                  # 模型权重、Jinja 模板、参考音频
```

---

**最后更新**: 2026-09-12
