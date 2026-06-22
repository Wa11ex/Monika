# Monika — Technical Deep Dive

> 面向技术面试官的深度文档。项目概览见 [README.md](README.md)。

---

## 架构总览

```
┌─────────────────────────────────────────────────────────┐
│                    GUI (PyQt6 + qasync)                  │
│   Live2D 渲染 │ 设置面板 │ 调试命令 │ 输入历史           │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────┐
│                  MonikaBus (事件总线)                     │
│   流式输入 → 分句 → TTS 排队 → AudioPlayer → Live2D      │
│   Tool Call 钩子 │ 打断检测 │ 表情/动作调度              │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────────┐
│              Brain 后端 (BaseBrain 接口)                  │
│   ┌─────────────────┐  ┌─────────────────┐              │
│   │ brain_loader.py │  │ brain_ollama.py │              │
│   │ (GGUF/llama-cpp)│  │ (OpenAI API)    │              │
│   └────────┬────────┘  └────────┬────────┘              │
│            │                    │                       │
│   ┌────────┴────────────────────┴────────┐              │
│   │        BrainStateMachine             │              │
│   │   THINK → TALK → TOOL → CODE         │              │
│   └────────┬─────────────────────────────┘              │
│            │                                            │
│   ┌────────┴────────┐  ┌──────────────┐                 │
│   │ ToolRegistry    │  │ MemoryManager │                 │
│   │ (工具注册/执行)  │  │ (ChromaDB)   │                 │
│   └─────────────────┘  └──────────────┘                 │
└─────────────────────────────────────────────────────────┘
```

---

## 设计模式

| 模式 | 应用位置 | 说明 |
|------|---------|------|
| **ReAct Loop** | `brain_loader.py` think_stream | Reason+Act 循环：模型输出 tool_call → 执行 → 结果注入 → 重新生成，while 直到模型停止或达上限 |
| **State Machine** | `brain_state.py` | Token 流 4 态路由，替代多层过滤器 |
| **Strategy** | `brain_loader.py` / `brain_ollama.py` | 双 LLM 后端可切换 |
| **Dependency Injection** | 各模块 `__init__` | 接受可选参数便于测试 |
| **Observer** | `bus.py` 钩子 | `on_tool_call` / `on_tool_result` 解耦工具执行与表情 |
| **Singleton** | `tool_registry.py` | 全局工具注册中心 |
| **Template Method** | `brain_base.py` | `_prepare_messages` → 子类实现 `think_stream` |

---

## Agent 架构（ReAct Loop）

Monika 实现了与 Claude Code、Cursor、Copilot 等商业 Agent 相同的 **ReAct (Reasoning + Acting)** 模式：

```
while round < max_tool_calls_per_turn:
    model output → BrainStateMachine 解析
        ├─ TALK 文本 → yield 到 TTS（用户听到）
        ├─ THINK 内容 → 剥离（内部推理）
        └─ TOOL 调用 → 检测到 </tool_call>
                            ↓
              ToolRegistry.execute(name, params)
                            ↓
              结果注入 messages (role: tool)
                            ↓
              重新 _generate() → 模型基于结果决定下一步
                            ↓
              if 模型输出更多 tool_call → 循环继续
              if 模型输出 TALK 文本     → 循环退出，回复用户
```

**安全边界**：
- `tools.max_tool_calls_per_turn` (默认 5) 硬上限防止无限循环
- 相同 tool+params 重复调用自动检测截断
- 工具结果超过 2000 字符自动截断，标注原长度
- 未注册工具名（模板示例）自动忽略，防止误触发
- `_current_gen` 计数器防止线程冲突（旧生成被新生成取代时自动静默退出）

### 打断机制

```
用户开口 → Ears VAD 检测 → Bus.interrupt()
  ├─ AudioPlayer.clear_and_stop()  # 立即停止播放
  ├─ TTS.cancel()                  # 取消正在合成的请求
  ├─ 清空 text_queue/audio_queue   # 丢弃排队中的句子
  └─ 重置 Live2D 表情

_watch_interrupt 任务：在音频出声前不触发（避免 LLM 思考期间误打断）
process_stream_input：逐 chunk 检查 _interrupted，立退
tool 执行期间不可打断（HTTP 请求安全限制），结果可能残留到下一轮
```

### 逐句显示同步

```
Bus._audio_player_worker 播放开始
  → sentence_callback(text)
    → main_window._on_sentence()
      → 第一句：chat_display.append()   （新建段落）
      → 后续句：chat_display.insertHtml()（同段拼接不换行）
      → verticalScrollBar.setValue(max) （自动滚到底）
```

**与商业 Agent 的对比**：

| 特性 | Monika | Claude Code | Cursor |
|------|--------|-------------|--------|
| ReAct 循环 | [x] while 多步 | [x] | [x] |
| 工具结果截断 | [x] 2000 chars | [x] | [x] |
| 用户确认 | [x] CONFIRM 权限 | [x] | [x] |
| 自主规划 | [ ] 待实现 | [x] | [x] |
| 沙箱隔离 | [x] 路径限制 | [x] 容器级 | [ ] |

---

## 关键技术决策

### 1. 为什么用状态机而不是多层过滤器

**之前**：`_ThinkFilter` → `ToolFormatAdapter` → `_strip_code_for_tts` 三层串行。

**问题**：
- Tool 前的 TALK 文本丢失（被当作 assistant 内容注入 messages）
- 线程冲突（Tool Loop 的 `_generate` 重入）
- `</think>` 泄漏进 TTS（re-generation 跳过了 THINK 状态）

**现在**：单一 `BrainStateMachine` 类，一次遍历 token 流，每个 token 属于且仅属于一个状态。分割点在结束标签之后（`</think>` 被 THINK 态吃掉，`<tool_call>` 被 TALK 态检测并切换）。

### 2. 为什么不微调模型

Qwen3.5 的 Jinja 模板**已内置**完整的 tool calling 支持（`<tool_call>` XML 格式）。模型在预训练阶段学过这个格式。我们只需：
1. 提供正确模板 → 模板负责"语言"
2. 状态机解析输出 → 适配器负责"翻译"
3. 用户换模型只需换模板 + 指定 `tools_format`

### 3. 记忆系统：双层 vs 单层

| 决策 | 说明 |
|------|------|
| 短期（context） | 全部对话历史（含 tool 调用结果），最大 N 轮裁剪 |
| 长期（ChromaDB） | 只存 TALK 文本（不含 think/tool），语义检索 |
| 为什么不存 think | 内部推理是瞬态的，存储会污染语义空间 |
| 为什么不存 tool | 工具调用是操作，不是知识 |

### 4. 安全模型

| 层级 | 机制 |
|------|------|
| 路径沙箱 | `_safe_path()` 限制在项目根目录 |
| 命令黑名单 | `rm -rf`、`format` 等（plan） |
| 权限分级 | auto（读）/ confirm（写、执行命令）/ always_ask（上网） |
| 文件备份 | WriteFile 前自动备份旧版本到 `assets/backups/`（plan） |

---

## 关键指标（预留）

| 指标 | 当前 | 目标 |
|------|------|------|
| 端到端延迟 (ASR→TTS) | ~3-8s | <5s |
| Tool call 成功率 | 待测 | >90% |
| 内存占用 | ~4-6GB | <8GB |
| 测试覆盖率 | 状态机已覆盖 | >70% |

---

## 目录结构（核心）

```
Monika/
├── core/
│   ├── brain_base.py        # LLM 基类（历史/记忆/工具注入）
│   ├── brain_loader.py      # GGUF 后端 + 状态机调用
│   ├── brain_state.py       # ★ 流式 token 状态机（可测试）
│   ├── bus.py               # 事件总线（TTS/播放/Live2D）
│   ├── tool_registry.py     # 工具注册单例
│   ├── tool_format.py       # Tool call XML 解析
│   ├── tools/               # 工具实现 (base, file_tools)
│   └── audio/               # 回消、口型同步、声纹
├── memory/memory_manager.py # ChromaDB + Lore
├── ui/                      # PyQt6 GUI
└── utils/                   # 配置、日志、模型导入
```

---

**最后更新**: 2026-06-16
