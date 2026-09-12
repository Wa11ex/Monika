import os
import sys
import logging
import numpy as np
import torch
import uvicorn
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator

# 配置区
DEFAULT_GPT_PATH = r"GPT_weights_v2\Monika-v2-e5.ckpt"
DEFAULT_SOVITS_PATH = r"SoVITS_weights_v2\Monika-v2_e4_s96.pth"

os.environ["version"] = "v2"
os.environ["is_half"] = "True"
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append(os.path.join(now_dir, "GPT_SoVITS"))

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

print(">>> [Init] 正在加载核心推理模块 (inference_webui)...")

try:
    import GPT_SoVITS.inference_webui as core
except Exception as e:
    print("!!! 导入失败，请确保文件在 GPT-SoVITS 根目录下运行")
    raise e

if hasattr(core, "init_bigvgan"):
    print(">>> [Init] 初始化 BigVGAN...")
    core.init_bigvgan()

INFERENCE_SEMAPHORE = asyncio.Semaphore(1)

class TTSRequest(BaseModel):
    text: str
    text_lang: str = "zh"
    ref_audio_path: str
    ref_text: str
    ref_lang: str = "all_zh"
    top_k: int = 20
    top_p: float = 1
    temperature: float = 0.9
    speed: float = 1.0
    gpt_path: str = None
    sovits_path: str = None
    tts_sr: bool = False


app = FastAPI(title="Monika V3-LoRA Server")

class MonikaEngine:
    def __init__(self):
        self.current_gpt = None
        self.current_sovits = None
        self.lang_map = { "all_zh": "中文", "en": "英文", "ja": "日文", "zh": "中英混合" }
        self.is_initialized = False

    def load_models(self, gpt_path, sovits_path):
        if gpt_path and gpt_path != self.current_gpt:
            print(f">>> [Model] 切换 GPT: {gpt_path}")
            core.change_gpt_weights(gpt_path)
            self.current_gpt = gpt_path
        
        if sovits_path and sovits_path != self.current_sovits:
            print(f">>> [Model] 切换 SoVITS : {sovits_path}")
            try:
                generator = core.change_sovits_weights(sovits_path)
                while True: next(generator)
            except StopIteration: pass
            except Exception as e: print(f"!!! 模型加载警告: {e}")
            self.current_sovits = sovits_path
        self.is_initialized = True

    async def infer_with_lock(self, req: TTSRequest) -> bytes:
        '''
        fix：
        1. 移除换行符，防止底层 webui 强制切分导致中英混合句子断裂
        2. 保持 text_lang="auto" 以支持多语种混合
        '''
        async with INFERENCE_SEMAPHORE:
            if req.gpt_path and req.sovits_path:
                if req.gpt_path != self.current_gpt or req.sovits_path != self.current_sovits:
                    print(f">>> [Infer] 检测到模型切换请求")
                    self.load_models(req.gpt_path, req.sovits_path)
            elif not self.is_initialized:
                raise HTTPException(status_code=400, detail="模型未加载且未提供模型路径")

            p_lang = self.lang_map.get(req.ref_lang, "中文")

            t_lang = self.lang_map.get(req.text_lang, "中英混合") 

            clean_text = req.text.replace("\n", " ").strip()
            
            print(f">>> [Infer] 锁定推理: {clean_text[:20]}...")

            loop = asyncio.get_event_loop()
            
            req.text = clean_text 
            
            audio_data = await loop.run_in_executor(
                None, 
                self._sync_inference,
                req, p_lang, t_lang
            )
            
            print(f">>> [Infer] 完成 (size: {len(audio_data)})")
            return audio_data

    def _sync_inference(self, req: TTSRequest, p_lang: str, t_lang: str) -> bytes:
        '''
        同步推理函数，在线程池中执行
        返回完整的 WAV 文件（带文件头）
        '''
        generator = core.get_tts_wav(
            ref_wav_path=req.ref_audio_path,
            prompt_text=req.ref_text,
            prompt_language=p_lang,
            text=req.text,
            text_language=t_lang,
            how_to_cut="不切",
            top_k=req.top_k,
            top_p=req.top_p,
            temperature=req.temperature,
            ref_free=False,
            speed=req.speed,
            if_freeze=False,
            inp_refs=None,
            sample_steps=16, 
            if_sr=req.tts_sr,
            pause_second=0.1
        )

        audio_chunks = []
        sr = 24000
        try:
            for sampling_rate, audio_data in generator:
                sr = sampling_rate
                audio_chunks.append(audio_data)
        except Exception as e:
            print(f"!!! 推理出错: {e}")
            import traceback
            traceback.print_exc()
            raise

        if not audio_chunks:
            raise Exception("生成失败，没有音频数据")

        final_audio = np.concatenate(audio_chunks)

        max_val = np.abs(final_audio).max()
        if max_val > 32767:
            final_audio = final_audio / max_val * 32767
        final_audio = final_audio.astype(np.int16)

        import io
        import soundfile as sf
        
        mem_file = io.BytesIO()
        sf.write(mem_file, final_audio, sr, format='wav')
        mem_file.seek(0)
        
        return mem_file.read()

    async def stream_audio(self, audio_data: bytes) -> AsyncGenerator[bytes, None]:

        chunk_size = 8192 
        for i in range(0, len(audio_data), chunk_size):
            yield audio_data[i:i + chunk_size]
            await asyncio.sleep(0) 

engine = MonikaEngine()

@app.post("/tts_stream")

async def tts_stream_endpoint(req: TTSRequest):
    #流
    audio_data = await engine.infer_with_lock(req)
    
    return StreamingResponse(
        engine.stream_audio(audio_data), 
        media_type="audio/wav"
    )
    
@app.on_event("startup")
async def startup_event():
    print(">>> [Startup] 正在预加载 Monika TTS 模型...")
    try:
        engine.load_models(DEFAULT_GPT_PATH, DEFAULT_SOVITS_PATH)
        print(">>> [Startup] 模型加载完成")
    except Exception as e:
        print(f">>> [Startup] 模型预加载失败: {e}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": engine.is_initialized,
        "current_gpt": os.path.basename(engine.current_gpt) if engine.current_gpt else None,
        "current_sovits": os.path.basename(engine.current_sovits) if engine.current_sovits else None
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000, workers=1)