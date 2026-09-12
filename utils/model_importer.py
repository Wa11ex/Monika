'''
模型导入器

支持GGUF文件或 HF safetensors 文件夹两种导入路径：
都会转换为 GGUF 文件（F16）并注册

progress_cb 签名：(pct: int, msg: str) -> None
  pct  : 0-100 进度百分比
  msg  : 当前状态说明文字
'''

import os
import sys
import asyncio
from typing import Callable

from utils.model_registry import add_model

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_DIR    = os.path.join(_PROJECT_ROOT, "assets", "model")
_CONVERTER    = os.path.join(_PROJECT_ROOT, "utils", "convert_hf_to_gguf.py")



def detect_format(path: str) -> str:
    '''识别路径对应的模型格式
    返回值：'gguf' | 'hf_safetensors' | 'unknown'
    '''
    path = path.strip()
    if os.path.isfile(path) and path.lower().endswith(".gguf"):
        return "gguf"
    if os.path.isdir(path):
        files = os.listdir(path)
        has_config = "config.json" in files
        has_tensors = any(f.endswith(".safetensors") for f in files)
        if has_config and has_tensors:
            return "hf_safetensors"
    return "unknown"


# -- GGUF 直接导入 ---------------------------------------------

async def import_gguf(src_path: str, output_name: str,
                      progress_cb: Callable[[int, str], None]) -> str:
    '''将 GGUF 文件复制到 assets/model/ （不在的话）并注册
    返回目标文件的绝对路径
    '''
    os.makedirs(_MODEL_DIR, exist_ok=True)
    src_path = os.path.normpath(os.path.abspath(src_path))
    dst_path = os.path.normpath(os.path.join(_MODEL_DIR, f"{output_name}.gguf"))

    if src_path == dst_path:
        progress_cb(100, "源文件已在目标目录，直接注册")
        add_model(output_name, dst_path, fmt="gguf")
        return dst_path

    try:
        total = os.path.getsize(src_path)
    except OSError:
        total = 0

    progress_cb(0, f"正在复制 -> {os.path.basename(dst_path)}")

    loop = asyncio.get_event_loop()

    def _copy_chunked():
        copied = 0
        with open(src_path, "rb") as fsrc, open(dst_path, "wb") as fdst:
            while True:
                chunk = fsrc.read(4 * 1024 * 1024)  # 4 MB
                if not chunk:
                    break
                fdst.write(chunk)
                copied += len(chunk)
                pct = int(copied * 100 / total) if total else 50
                progress_cb(pct, f"已复制 {copied // (1024*1024)} MB / {total // (1024*1024)} MB")

    await loop.run_in_executor(None, _copy_chunked)

    add_model(output_name, dst_path, fmt="gguf")
    progress_cb(100, "✓ 导入完成")
    return dst_path


# -- HF safetensors 文件夹转换导入 -----------------------------------

async def import_hf(src_path: str, output_name: str,
                    progress_cb: Callable[[int, str], None]) -> str:
    '''将 HF safetensors 文件夹转换为 F16 GGUF 并注册
    用的是 llama.cpp 仓库中的 convert_hf_to_gguf.py 转换器，我直接偷来了
    返回生成的 GGUF 文件的绝对路径
    '''
    os.makedirs(_MODEL_DIR, exist_ok=True)
    dst_path = os.path.normpath(os.path.join(_MODEL_DIR, f"{output_name}.gguf"))

    if not os.path.exists(_CONVERTER):
        raise FileNotFoundError(
            f"转换器脚本未找到: {_CONVERTER}\n"
            "请将 llama.cpp 仓库中的 convert_hf_to_gguf.py 放入项目 utils/ 目录"
        )

    progress_cb(0, "正在检查依赖...")
    missing = []
    for pkg in ("transformers", "sentencepiece", "numpy"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        raise ImportError(
            f"缺少 Python 依赖: {', '.join(missing)}\n"
            f"请运行: pip install {' '.join(missing)}"
        )

    progress_cb(5, f"开始转换 {os.path.basename(src_path)} -> F16 GGUF...")

    cmd = [
        sys.executable, _CONVERTER,
        src_path,
        "--outtype", "f16",
        "--outfile", dst_path,
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    last_pct = 5
    async for raw_line in proc.stdout:
        line = raw_line.decode("utf-8", errors="replace").rstrip()
        if not line:
            continue
        
        # 尝试解析进度百分比，如果有的话
        if "%" in line:
            try:
                pct_str = line.split("%")[0].split()[-1].strip(".(")
                pct = max(5, min(95, int(float(pct_str))))
                last_pct = pct
            except (ValueError, IndexError):
                pass
        progress_cb(last_pct, line[:160])

    await proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"转换失败，退出码 {proc.returncode}\n"
            "请查看上方日志了解详情"
        )

    # 读架构 TODO
    arch = "unknown"
    try:
        import json
        with open(os.path.join(src_path, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        arch = cfg.get("model_type") or (
            cfg.get("architectures", ["unknown"])[0]
            if cfg.get("architectures") else "unknown"
        )
    except Exception:
        pass

    add_model(output_name, dst_path, fmt="gguf", arch=arch)
    progress_cb(100, "✓ 转换并导入完成")
    return dst_path
