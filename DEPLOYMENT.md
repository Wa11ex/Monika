# Monika 部署指南

> 本文档包含完整的安装步骤、GPU 配置、模型权重准备、配置文件说明及常见问题。
> 项目特性介绍请参阅 [README.md](README.md)。

---

## 开始之前需要准备什么

### 硬件

- **GPU（强烈推荐）**：没有 GPU 也能跑，但 TTS 推理会非常慢。NVIDIA 显卡 + CUDA 是最省心的。6GB 显存经过测试是最低要求，8GB 以上比较舒服。
- **内存**：16GB 以上。
- **硬盘**：至少留 30GB 空间，光各种模型权重就很大。
- 目前经过测试，最低的可流畅运行配置是：RTX 3060 Laptop 6G + 11th i7 处理器（32GB 内存）

### 软件（必须提前装好）

| 软件 | 用途 | 下载 |
|------|------|------|
| Python 3.11 / 3.12 | 运行环境 | https://www.python.org/downloads/ |
| Git | 克隆模型仓库 | https://git-scm.com/download/win |

以下为可选：

| 软件 | 用途 | 下载 |
|------|------|------|
| Ollama | LLM 后端（之一） | https://ollama.com |
| Voicemeeter | 虚拟音频路由（可选） | https://vb-audio.com/Voicemeeter/Banana/ |
| VTube Studio | 外接 Live2D 桌面宠物 | Steam 免费 |

> Voicemeeter 和 VTube Studio **均为可选**。GUI 内置 Live2D 渲染，不需要 VTS。回声消除使用 Windows 原生 WASAPI Loopback，不需要虚拟音频驱动。

---

## GPU 加速配置

这是整个安装里最麻烦的一步，没法自动化。

**为什么麻烦？** PyTorch GPU 版本必须匹配你的 CUDA 驱动版本。

### 第一步：确认 N 卡 + 驱动

```bash
nvidia-smi
```

右上角的 "CUDA Version" 就是你的版本号。如果这个命令报错，说明没有 N 卡或驱动没装好。

### 第二步：装对应版本的 PyTorch

访问 https://pytorch.org/get-started/locally/，选好系统、CUDA 版本，复制安装命令。例如：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

`cu121` 对应 CUDA 12.1，`cu118` 对应 11.8。

### 第三步：验证

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

输出 `True` 即成功。

> 只用 CPU 的话跳过以上，直接 `pip install torch`。

---

## 安装步骤

### Windows

```bat
git clone https://github.com/Wa11ex/Monika.git
cd Monika
setup.bat
```

`setup.bat` 会做：
1. 检查 Python 和 Git
2. 克隆 SenseVoiceSmall（语音识别模型，尝试 ModelScope / hf-mirror / HuggingFace 三个源）
3. 引导下载 GPT-SoVITS v2pro 打包发行版（需手动下载解压）
4. 自动检测 CUDA 版本，选择最匹配的 llama-cpp-python 预编译 wheel
5. 安装 editdistance（预编译 wheel）
6. 安装剩余 pip 依赖

> **GPT-SoVITS 需要手动下载**：前往 https://github.com/RVC-Boss/GPT-SoVITS/releases
> 下载 `GPT-SoVITS-v2pro-20250604-*.7z`，解压到 `modules/GPT-SoVITS-v2pro-20250604/`。

### Linux

```bash
git clone https://github.com/Wa11ex/Monika.git
cd Monika
chmod +x setup.sh launcher.sh
./setup.sh
```

**Linux 下的主要区别：**
1. **TTS 引擎**：建议使用官方 GPT-SoVITS 源码和 Conda 部署，然后拷贝 `modules/tts_server/*.py` 进它的目录启动。自带发行版的 `runtime` 对 Linux 无效。
2. **虚拟麦克风**：使用 PulseAudio 的 `module-null-sink` 或 PipeWire 实现回环。
3. 启动项目使用 `./launcher.sh`

---

## 模型权重（大文件，需自行准备）

### GPT-SoVITS 模型权重

放在 `modules/GPT-SoVITS-v2pro-20250604/` 下：
- `GPT_weights_v2Pro/` — GPT 部分权重
- `SoVITS_weights_v2Pro/` — SoVITS 部分权重

这些是你自己训练或下载的角色声音模型，要匹配 v2pro 版本。

### 参考音频

放在 `assets/wav/ref/`：
- 一小段参考音频（wav 格式），TTS 用它来克隆说话风格
- 建议 10 秒以内，清晰无噪音
- 路径在 `config.yaml` 的 `tts.ref_audio_path` 配置

### LLM 模型

**Ollama 后端：**
```bash
ollama pull Monika:latest
```

**GGUF 后端：**
- 放在 `assets/model/` 下
- 路径在 `config.yaml` 的 `brain.gguf.model_path` 配置
- 可通过 GUI 设置 → 大脑 → 导入模型，支持自动转换 safetensors

SenseVoiceSmall 模型 `setup.bat` 会自动克隆，不用单独下载。

---

## 跑起来

### 方式一：图形界面（推荐）

```bash
python gui.py
```

- 可视化修改各项设置
- 麦克风设备下拉枚举
- Live2D 渲染：拖拽、缩放、位置自动记忆
- PTT 或麦克风常开切换
- VTS 同步：设置 → 系统 → VTS 区块 → "同步 VTS"

### 方式二：终端模式

```bash
python main.py
```

确保依赖服务已启动（或 `launcher.ollama.auto_start: true`）。对着麦克风说话，按 ESC 退出。

---

## 配置文件说明

所有配置可通过 GUI 设置界面修改（也可以直接编辑 `config.yaml`），自动保存在 `config.yaml` 和 `assets/user_settings.yaml`。

### 选 LLM 后端

**Ollama（推荐先试）：**
```yaml
brain:
  backend: "ollama"
  ollama:
    model_name: "Monika:latest"
    enable_thinking: true
    think_budget: 0       # 思维链字符预算，0=不限制
```

**本地 GGUF：**
```yaml
brain:
  backend: "gguf"
  gguf:
    model_path: "assets/model/Monika-qwen3.5.gguf"
    external_template: "assets/model/chat_template_Qwen3.5.jinja"
    n_gpu_layers: -1      # -1=全量GPU，显存不够改小
    enable_thinking: true
    think_budget: 512
```

### 音频设备

- **系统 tab → 音频路由**：选择 TTS 播放设备
- **降噪 & 回声消除 → EC 参考输出设备**：选择 WASAPI Loopback 监听设备

不确定设备名的话：
```bash
python utils/diagnose_audio.py list
```

### TTS 参考音频

```yaml
tts:
  ref_audio_path: "assets/wav/ref/2-2.wav"
  ref_text: "这里填参考音频对应的文本"
```

### 主动发言（PO）

```yaml
proactive_output:
  enabled: true
  cooling_windows:
    - [240, 360]    # 沉默 4~6 分钟后第一次主动发言
    - [360, 600]    # 再等 6~10 分钟第二次
                    # 之后不再主动，等你开口才重置
```

时间单位是秒。不想要就 `enabled: false`。

---

## 分发及打包说明

针对需要进行二次分发的开发者：

**强烈不建议**用 PyInstaller 打包成单文件 `.exe`（依赖地狱 + 启动极慢）。

官方建议的分发姿势：
1. **轻量级加密核心代码**：使用 PyArmor 或 Nuitka 将 `core/`、`ui/`、`utils/` 混淆或编译
2. **免安装绿色环境**：内置 Python Embedded 版，与加密项目文件打包散件发布
3. **快捷启动器**：用 C#/C++ 或转换 `launcher.bat` 的小型 `.exe` 做门面

---


## 常见问题

**Q：setup.bat 克隆 SenseVoiceSmall 很慢或失败？**

脚本按 ModelScope → hf-mirror → HuggingFace 顺序尝试。都不行的话，手动克隆放到 `modules/SenseVoiceSmall/`。

**Q：GPT-SoVITS 在哪下载？**

https://github.com/RVC-Boss/GPT-SoVITS/releases — 找 `GPT-SoVITS-v2pro-20250604-*.7z`。

**Q：`torch.cuda.is_available()` 返回 False？**

装的是 CPU 版 torch。重新装带 CUDA 的版本。

**Q：说话没反应？**

VAD 不够灵敏。在 `config.yaml` 把 `ears.vad_threshold` 改小（如 0.018 → 0.012）。

**Q：TTS 启动了没声音？**

打开设置 → 系统 → 音频路由，确认输出设备选对了。

**Q：VTube Studio 如何连接？**

打开设置 → 系统 → VTS 区块 → "同步 VTS"。第一次连接 VTS 会弹窗授权，点允许即可。VTS 里要先开 API 插件：Setting → VTube Studio API → Start API。

**Q：改了 `modules/tts_server/` 脚本怎么生效？**

重新运行 `setup.bat`（选 n 跳过重建虚拟环境），或手动：
```bat
copy modules\tts_server\tts_v3_server.py modules\GPT-SoVITS-v2pro-20250604\
```

---

**最后更新**: 2026-09
