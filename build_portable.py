'''
构建便携版 Monika：代码 + 独立 venv + 一键启动器。

用法：
    python build_portable.py

产出：release/Monika-portable/
    ├── src/                  # 项目代码（瘦身后）
    ├── .venv/                # 独立 Python 环境（预装依赖）
    ├── 启动Monika.bat        # 一键启动
    └── 首次使用说明.txt

注意：
    - 模型（LLM GGUF / SenseVoiceSmall / GPT-SoVITS）体积大，不随包分发，
      首次运行时引导用户自行放置（见"首次使用说明.txt"）。
    - torch / llama-cpp-python 的 CUDA 版需要单独装，本脚本默认装 CPU 版兜底，
      有 N 卡的用户可手动执行 GPU 版安装命令（脚本末尾会打印提示）。
'''

from __future__ import annotations

import os
import shutil
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RELEASE_DIR = os.path.join(PROJECT_ROOT, "release")
DIST_DIR = os.path.join(RELEASE_DIR, "Monika-portable")
SRC_DIR = os.path.join(DIST_DIR, "src")
VENV_DIR = os.path.join(DIST_DIR, ".venv")

# -- 资源目录：放便携包根（src 的上一级），不进 src -----------------
#    代码里 helpers._get_project_root() / ears.py 的"便携版"逻辑
#    约定：代码在 src/，资源（modules/assets）在 src 的上一级。
RESOURCE_DIRS = {"modules", "assets"}

# -- 不随包分发的目录/文件 -------------------------------------------------
EXCLUDE_DIRS = {
    ".git", ".vscode", "__pycache__", ".venv", "venv", "env",
    "test_output", "learning", "plans", "build", "release",
    "logs", "figs", "wheels",
    ".github", "previous core", "properties",
}

EXCLUDE_FILES = {
    # 用户私有配置/密钥，不随包分发
    "config.yaml", "vts_token.txt", ".webui_secret_key",
    "NOTES.txt", "check_function_calling.py",
    "test_brain.py", "test_report.md", "test_lipsync.py",
    "diagnose_audio.py",
}


def log(msg: str) -> None:
    print(f"[build] {msg}")


def py() -> str:
    '''返回系统 python 命令（优先 py -3.11，回退当前解释器）'''
    # 便携版约定：用当前运行本脚本的解释器版本建 venv
    return sys.executable


def venv_python() -> str:
    '''返回便携环境内的 python 路径'''
    if os.name == "nt":
        return os.path.join(VENV_DIR, "Scripts", "python.exe")
    return os.path.join(VENV_DIR, "bin", "python")


def stage_copy_code() -> None:
    '''把项目代码拷贝到 src/，跳过开发杂物和资源目录'''
    log("拷贝项目代码 -> src/ ...")
    if os.path.exists(SRC_DIR):
        shutil.rmtree(SRC_DIR)

    def _ignore(dirpath, names):
        ignored = set()
        for n in names:
            if n in EXCLUDE_DIRS or n in RESOURCE_DIRS:
                ignored.add(n)
            elif n in EXCLUDE_FILES:
                ignored.add(n)
            elif n.endswith(".pyc") or n.endswith(".pyo"):
                ignored.add(n)
        return ignored

    shutil.copytree(PROJECT_ROOT, SRC_DIR, ignore=_ignore)
    log("代码拷贝完成")


def stage_copy_resources() -> None:
    '''把资源目录（modules/assets）拷贝到便携包根（src 的上一级）'''
    for name in RESOURCE_DIRS:
        src = os.path.join(PROJECT_ROOT, name)
        dst = os.path.join(DIST_DIR, name)
        if not os.path.isdir(src):
            log(f"资源目录不存在，跳过: {name}")
            continue
        if os.path.exists(dst):
            shutil.rmtree(dst)
        log(f"拷贝资源目录 {name} -> {dst} ...")
        shutil.copytree(src, dst)
    log("资源拷贝完成")


def create_venv() -> None:
    '''在便携目录内创建独立 venv'''
    if os.path.exists(venv_python()):
        log("venv 已存在，跳过创建")
        return
    log("创建独立 venv ...")
    subprocess.check_call([py(), "-m", "venv", VENV_DIR])
    log("venv 创建完成")


def _has_nvidia_gpu() -> bool:
    '''检测是否有 N 卡（nvidia-smi 可用且能正常返回）'''
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except Exception:
        return False


def install_platform_deps() -> None:
    '''装平台特异依赖：torch + torchaudio + llama-cpp-python（按 CPU/CUDA 选版本）

    必须在 requirements.txt 之前装：sentence-transformers / chromadb 会间接
    依赖 torch，先装好对的版本，后面 pip 才会跳过。
    '''
    vpy = venv_python()
    if _has_nvidia_gpu():
        log("检测到 NVIDIA GPU，安装 CUDA 版 torch / llama-cpp-python ...")
        # torch + torchaudio 走 PyTorch 官方 index，自动保证版本匹配
        torch_ok = False
        for cu in ("cu124", "cu121"):
            try:
                subprocess.check_call([vpy, "-m", "pip", "install", "torch", "torchaudio",
                                       "--index-url", f"https://download.pytorch.org/whl/{cu}"])
                torch_ok = True
                break
            except subprocess.CalledProcessError:
                log(f"PyTorch {cu} 安装失败，尝试下一个 ...")
        if not torch_ok:
            log("PyTorch CUDA 版均失败，回退 CPU 版 torch ...")
            subprocess.check_call([vpy, "-m", "pip", "install", "torch", "torchaudio"])

        # llama-cpp-python 走 abetlen index
        try:
            subprocess.check_call([vpy, "-m", "pip", "install", "llama-cpp-python",
                                   "--extra-index-url",
                                   "https://abetlen.github.io/llama-cpp-python/whl/cu124"])
        except subprocess.CalledProcessError:
            log("llama-cpp-python CUDA 版失败，回退 CPU 版 ...")
            subprocess.check_call([vpy, "-m", "pip", "install", "llama-cpp-python"])
    else:
        log("未检测到 NVIDIA GPU，安装 CPU 版 torch / llama-cpp-python ...")
        subprocess.check_call([vpy, "-m", "pip", "install", "torch", "torchaudio", "llama-cpp-python"])


def install_deps() -> None:
    '''安装依赖（走清华镜像加速）'''
    vpy = venv_python()
    log("升级 pip ...")
    subprocess.check_call([vpy, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel",
                           "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])

    # 平台特异依赖（torch / llama-cpp-python），必须在 requirements 之前
    install_platform_deps()

    # editdistance 从 wheels/ 离线装（C++ 源码在 Windows MSVC 下编译失败）
    wheels_dir = os.path.join(PROJECT_ROOT, "wheels")
    if os.path.isdir(wheels_dir):
        log("离线安装 editdistance（bundled wheel）...")
        subprocess.check_call([vpy, "-m", "pip", "install", "editdistance",
                               "--no-index", "--find-links", wheels_dir])

    log("安装 requirements.txt ...")
    req = os.path.join(PROJECT_ROOT, "requirements.txt")
    subprocess.check_call([vpy, "-m", "pip", "install", "-r", req,
                           "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"])


def write_launcher() -> None:
    '''生成一键启动器'''
    bat = os.path.join(DIST_DIR, "启动Monika.bat")
    vpy = ".venv\\Scripts\\python.exe"
    content = f'''@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo   Monika - 便携版启动器
echo ============================================
echo.
echo [*] 启动 GUI（首次启动会自动初始化模型）
echo.
"{vpy}" src\\gui.py
if errorlevel 1 (
    echo.
    echo [错误] 启动失败，请检查：
    echo   1. 是否已放置 LLM 模型到 assets\\model\\
    echo   2. 是否已按"首次使用说明.txt"完成配置
    echo.
    pause
)
'''
    with open(bat, "w", encoding="utf-8") as f:
        f.write(content)
    log(f"生成启动器: {bat}")


def write_readme() -> None:
    '''生成首次使用说明'''
    txt = os.path.join(DIST_DIR, "首次使用说明.txt")
    content = '''============================================================
  Monika 便携版 - 首次使用说明
============================================================

本包自带 Python 运行环境和全部依赖，无需安装 Python。

一、需要你自行放置的文件（模型体积大，不随包分发）

1. LLM 模型（二选一）
   GGUF 后端：
     - 放置 .gguf 到  assets\\model\\ 目录
     - 编辑  src\\config example.yaml  中 brain.gguf.model_path
   Ollama 后端（推荐，免放模型）：
     - 先安装并启动 Ollama：https://ollama.com
     - 拉取模型：ollama pull <模型名>

2. 语音识别模型 SenseVoiceSmall（约 1GB）
   - 下载后放到  modules\\SenseVoiceSmall\\

3. TTS 引擎 GPT-SoVITS（可选，需要语音回复时）
   - 下载 GPT-SoVITS 便携版，解压到  modules\\GPT-SoVITS-v2pro-20250604\\

二、配置

1. 复制  src\\config example.yaml  为  src\\config.yaml
2. 按需修改配置（模型路径、设备、TTS 参考音频等）

三、启动

双击  启动Monika.bat

============================================================
'''
    with open(txt, "w", encoding="utf-8") as f:
        f.write(content)
    log(f"生成说明: {txt}")


def init_default_config() -> None:
    '''首次运行：若 src/config.yaml 不存在，从 config example.yaml 复制一份默认配置'''
    src_cfg = os.path.join(SRC_DIR, "config example.yaml")
    dst_cfg = os.path.join(SRC_DIR, "config.yaml")
    if os.path.isfile(src_cfg) and not os.path.isfile(dst_cfg):
        shutil.copy(src_cfg, dst_cfg)
        log("已生成默认 config.yaml（从 config example.yaml 复制）")
    elif not os.path.isfile(src_cfg):
        log("警告：找不到 config example.yaml，未生成 config.yaml")


def main() -> None:
    log("=== 构建便携版 Monika ===")
    os.makedirs(DIST_DIR, exist_ok=True)

    stage_copy_code()
    stage_copy_resources()
    init_default_config()
    create_venv()
    install_deps()
    write_launcher()
    write_readme()

    log("=== 构建完成 ===")
    log(f"输出目录: {DIST_DIR}")
    log("")
    log("平台特异依赖（torch / llama-cpp-python）已按显卡自动选择 CPU/CUDA 版。")
    log("双击 release\\Monika-portable\\启动Monika.bat 即可运行。")


if __name__ == "__main__":
    main()
