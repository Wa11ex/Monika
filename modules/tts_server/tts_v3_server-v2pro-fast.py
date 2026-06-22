import os
import sys
import io
import logging
import numpy as np
import torch
import uvicorn
import asyncio
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator

# 配置区
DEFAULT_GPT_PATH = r"GPT_weights_v2Pro\v2PROOOO-e15.ckpt"
DEFAULT_SOVITS_PATH = r"SoVITS_weights_v2Pro\v2PROOOO_e8_s192.pth"

os.environ["version"] = "v2Pro"
os.environ["is_half"] = "True"
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir, "GPT_SoVITS"))

# 屏蔽无关日志
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

# 注意：不在模块顶层 import TTS_infer_pack / inference_webui，
# 避免 Gradio、BERT 模型等在 import 阶段就被加载（~3-5s 纯浪费）
# 所有模型加载推迟到 startup_event，由 uvicorn 的 lifespan 机制触发

INFERENCE_SEMAPHORE = asyncio.Semaphore(1)

class TTSRequest(BaseModel):
    text: str
    text_lang: str = "zh"
    ref_audio_path: str
    ref_text: str
    ref_lang: str = "all_zh"
    top_k: int = 30
    top_p: float = 0.85
    temperature: float = 1.0
    speed: float = 1.0
    tts_sr: bool = False


app = FastAPI(title="Monika TTS Server")

# TTS pipeline 见 startup_event 
_tts_pipeline = None
_is_initialized = False


def _sync_inference(inputs: dict) -> bytes:
    '''在线程池中同步执行推理，返回完整 WAV bytes'''
    audio_chunks = []
    sr = 24000
    for sampling_rate, audio_data in _tts_pipeline.run(inputs):
        sr = sampling_rate
        audio_chunks.append(audio_data)

    if not audio_chunks:
        raise RuntimeError("生成失败，没有音频数据")

    final_audio = np.concatenate(audio_chunks)
    mem_file = io.BytesIO()
    sf.write(mem_file, final_audio, sr, format="wav")
    mem_file.seek(0)
    return mem_file.read()


async def _stream_audio(audio_data: bytes) -> AsyncGenerator[bytes, None]:
    chunk_size = 8192
    for i in range(0, len(audio_data), chunk_size):
        yield audio_data[i:i + chunk_size]
        await asyncio.sleep(0)


@app.post("/tts_stream")
async def tts_stream_endpoint(req: TTSRequest):
    if not _is_initialized:
        raise HTTPException(status_code=503, detail="TTS 模型尚未就绪")

    async with INFERENCE_SEMAPHORE:
        clean_text = req.text.replace("\n", " ").strip()
        print(f">>> [Infer] 锁定推理: {clean_text[:20]}...")

        inputs = {
            "text": clean_text,
            "text_lang": req.text_lang,
            "ref_audio_path": req.ref_audio_path,
            "prompt_text": req.ref_text,
            "prompt_lang": req.ref_lang,
            "top_k": req.top_k,
            "top_p": req.top_p,
            "temperature": req.temperature,
            "text_split_method": "cut0",  # 不切，旧版：how_to_cut="不切"
            "speed_factor": req.speed,
            "sample_steps": 16,
            "batch_size": 1,
            "fragment_interval": 0.1,
            "seed": -1,
            "parallel_infer": True,
        }

        loop = asyncio.get_event_loop()
        try:
            audio_data = await loop.run_in_executor(None, _sync_inference, inputs)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"推理失败: {e}")

        print(f">>> [Infer] 完成 (size: {len(audio_data)})")
        return StreamingResponse(_stream_audio(audio_data), media_type="audio/wav")


@app.on_event("startup")
async def startup_event():
    '''
    在 uvicorn lifespan 中加载模型
    直接使用 TTS_infer_pack.TTS，跳过 inference_webui（含 Gradio、BERT 提前加载等开销）
    '''
    global _tts_pipeline, _is_initialized
    print(">>> [Startup] 正在初始化 TTS 推理管线 (TTS_infer_pack.TTS)...")

    def _load_pipeline():
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
        device = "cuda" if torch.cuda.is_available() else "cpu"
        is_half = device == "cuda"
        cfg = TTS_Config({
            "custom": {
                "device": device,
                "is_half": is_half,
                "version": "v2Pro",
                "t2s_weights_path": DEFAULT_GPT_PATH,
                "vits_weights_path": DEFAULT_SOVITS_PATH,
                "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
                "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
            }
        })
        return TTS(cfg)

    loop = asyncio.get_event_loop()
    try:
        _tts_pipeline = await loop.run_in_executor(None, _load_pipeline)
        _is_initialized = True
        print(">>> [Startup] TTS 管线加载完成")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f">>> [Startup] TTS 管线加载失败: {e}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": _is_initialized,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, workers=1)
