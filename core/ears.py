import sounddevice as sd
import numpy as np
import logging
import keyboard
import threading
import time
from collections import deque
from typing import Optional
import random
import os
from utils.config_loader import get_config
from utils.logger import get_logger
from core.audio.audio_utils import find_device, get_device_channels, print_all_devices
from core.interfaces import BaseEars

# 强制 ModelScope/funasr 在本地目录寻找模型，不联网查版本
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 便携版: 如果代码在 src/ 子目录，项目根再往上一级
if os.path.basename(_ROOT) == "src":
    _ROOT = os.path.dirname(_ROOT)
_PROJECT_ROOT = _ROOT
os.environ.setdefault("MODELSCOPE_CACHE_HOME", os.path.join(_PROJECT_ROOT, "modules"))

# PO 超时标记
SILENCE_TIMEOUT = "__silence_timeout__"

logging.getLogger("modelscope").setLevel(logging.ERROR)
log = get_logger(__name__)

class Ears(BaseEars):
    def __init__(self, model_dir=None, device=None, speaker_identifier=None):
        '''初始化听觉系统，所有配置从 config.yaml 读取'''
        log.info("[Ears] 初始化听觉系统配置（ASR 模型延迟加载）...")

        # -- base 配置 --
        self.model_dir = model_dir or get_config("ears.model_name", r"modules\SenseVoiceSmall")
        self.device = device or get_config("ears.device", "cuda")
        self.samplerate = get_config("ears.samplerate", 24000)
        self.block_size = get_config("ears.block_size", 4000)
        self.vad_threshold = get_config("ears.vad_threshold", 0.007)
        self.interrupt_vad_threshold = get_config("ears.interrupt_vad_threshold", 0.015)
        self.silence_limit = get_config("ears.silence_limit", 0.5)
        self.pre_buffer_len = get_config("ears.pre_buffer_len", 4)
        
        # -- emotion 检测配置 --
        emotion_config = get_config("ears.emotion", {})
        self.emotion_enabled = emotion_config.get("enabled", True) if isinstance(emotion_config, dict) else True
        self.emotion_add_to_context = emotion_config.get("add_to_context", True) if isinstance(emotion_config, dict) else True
        
        # -- 一些记忆和后台流 --
        self.speaker_id = speaker_identifier
        self.last_speaker = None
        self.last_emotion = None
        self.monitoring_stream = None
        self.loopback_stream = None
        
        # -- 外部停止信号（供 GUI 麦克风切换和关麦使用） --
        self._stop_listen = threading.Event()

        # ASR 模型延迟加载：首次 generate 时才加载，避免启动时阻塞 15s
        self.model = None
        self._model_loading = False
        
        # ec设置
        ec_config = get_config("ears.echo_cancellation", {})
        self.ec_enabled = ec_config.get("enable_deepfilter", True) if isinstance(ec_config, dict) else True
        self.echo_canceller = None
        
        # Loopback 参考设备
        self.loopback_keyword = ec_config.get("loopback_keyword", "Voicemeeter Out B2")
        self.loopback_id = find_device(self.loopback_keyword)
        self.mic_id = get_config("ears.mic_id", sd.default.device[0])
        
        if self.ec_enabled:
            from core.audio.ec import EchoCanceller
            self.echo_canceller = EchoCanceller(samplerate=self.samplerate)
            log.info("[Ears] EC 已启用 | DeepFilterNet")
        
        log.info("[Ears] 配置已封装 (sr=%d, emotion=%s, Loopback:%s)",
                 self.samplerate, self.emotion_enabled, self.loopback_id)

    def _ensure_model(self) -> None:
        '''延迟加载 ASR 模型：首次调用时才加载（~15s），启动时不阻塞'''
        if self.model is not None:
            return
        if self._model_loading:
            return  # 防止重复加载
        self._model_loading = True
        log.info("[Ears] 首次使用 ASR，正在加载 SenseVoiceSmall（~15s）...")
        from funasr import AutoModel
        _abs_model_dir = (
            self.model_dir if os.path.isabs(self.model_dir)
            else os.path.join(_PROJECT_ROOT, self.model_dir)
        )
        self.model = AutoModel(
            model=_abs_model_dir,
            trust_remote_code=True,
            device=self.device,
            disable_update=True,
            verbose=False,
        )
        self._model_loading = False
        log.info("[Ears] ASR 模型加载完成")

    def _identify_speaker(self, recording: np.ndarray) -> None:
        '''内部用声纹识别，更新 self.last_speaker'''
        self.last_speaker = None
        if self.speaker_id is None:
            return
        try:
            result = self.speaker_id.identify(recording)
            self.last_speaker = result
            if result:
                if result["is_new"]:
                    log.info("[SpeakerID] 新朋友来了！已自动命名为 [%s]", result['name'])
                else:
                    log.info("[SpeakerID] 识别到 [%s]  置信度: %.2f", result['name'], result['confidence'])
        except Exception as e:
            log.warning("[SpeakerID] 识别异常（已跳过）: %s", e)

    def identify_speaker(self, audio_data) -> Optional[dict]:
        '''公开接口：识别说话人身份（实现 BaseEars.identify_speaker）
        Args:
            audio_data: float32 mono ndarray
            
        Returns:
            说话人dict {"name": ..., "is_new": bool, "confidence": float} 或 None
        '''
        self._identify_speaker(audio_data)
        return self.last_speaker

    def _process_raw_audio(self, raw_data):
        '''对原始音频进行 ec （DeepFilterNet） 处理'''
        if self.ec_enabled and self.echo_canceller and self.echo_canceller.available:
            return self.echo_canceller.process(raw_data)
        return raw_data
    
    def _get_loopback_stream(self):
        '''
        返回 (stream, samplerate)
        如果 loopback 设备不可用，返回 (None, self.samplerate)
        '''
        from core.audio.loopback import make_loopback_stream
        loopback_device_id = get_config("ears.echo_cancellation.loopback_device_id", None)
        return make_loopback_stream(self.loopback_keyword, self.samplerate, loopback_device_id)

    def listen_auto(self, listen_timeout=None):
        '''
        带回溯的自动录音

        参数:
            listen_timeout: 多长时间视为slient
        
        流程：
        1. 麦克风读取原始音频块
        2. 立即通过 EC 处理获得干净音频（永久在线）
        3. 基于干净音频的 RMS 做 VAD 决策，判断录音时间
        4. 若3通过，记录干净音频块
        5. 结束后直接进行 ASR/声纹识别
        '''
        print("\r>>> [Ears] 聆听中... (Esc退出)", end="")
        
        frames = []        
        pre_buffer = deque(maxlen=self.pre_buffer_len)
        is_speaking = False
        silence_start_time = None
        listen_start_time = time.time()

        mic_stream = sd.InputStream(device=self.mic_id, channels=1, samplerate=self.samplerate, dtype='float32')
        loopback_stream, loopback_sr = self._get_loopback_stream()
        loopback_block = int(self.block_size * loopback_sr / self.samplerate)

        try:
            mic_stream.start()
            if loopback_stream:
                loopback_stream.start()
            print("\r>>> [Ears] 聆听中... (Esc退出)", end="")

            while True:
                if keyboard.is_pressed('esc') or self._stop_listen.is_set():
                    print("\n>>> [Ears] 停止")
                    return "exit"

                raw_data, _ = mic_stream.read(self.block_size)
                loop_data = np.zeros(loopback_block)
                if loopback_stream:
                    loop_data, _ = loopback_stream.read(loopback_block)
                
                
                mic_rms = np.sqrt(np.mean(raw_data**2))
                loop_rms = np.sqrt(np.mean(loop_data**2))
                minus_rms = max(mic_rms - loop_rms, 0.0) # 方法；相减

                if is_speaking:
                    frames.append(raw_data)  # 存原始音频
                    if minus_rms < self.vad_threshold:
                        if silence_start_time is None:
                            silence_start_time = time.time()
                        elif time.time() - silence_start_time > self.silence_limit:
                            print(f"\n>>> [Ears] 录音结束 ({len(frames)*self.block_size/self.samplerate:.1f}s)")
                            break
                    else:
                        silence_start_time = None

                    if self._stop_listen.is_set(): # 录音中收到停止信号： 丢弃本次录音，立即退出
                        print("\n>>> [Ears] 录音被外部停止，丢弃")
                        return "exit"
                else:
                    # PO 触发
                    if listen_timeout is not None and time.time() - listen_start_time >= listen_timeout:
                        return SILENCE_TIMEOUT

                    if minus_rms > self.vad_threshold:
                        print("\n>>> [Ears] 捕捉到声音!")
                        is_speaking = True
                        silence_start_time = None
                        frames.extend(pre_buffer)  # +预缓冲
                        frames.append(raw_data)
                    else:
                        pre_buffer.append(raw_data)  # 存原始音频

        finally:
            mic_stream.stop(); mic_stream.close()
            if loopback_stream:
                try:
                    loopback_stream.stop()
                    loopback_stream.close() # 关闭
                except Exception:
                    log.debug("[Ears] loopback 流关闭异常（可能已关闭）", exc_info=True)

        if len(frames) <= self.pre_buffer_len + 1:
            return None

        recording = np.squeeze(np.concatenate(frames, axis=0))

        if len(recording) / self.samplerate < 0.8:
            return None

        self._identify_speaker(recording) # 识别说话人

        # ASR
        recording_for_asr = recording
        if self.ec_enabled and self.echo_canceller and self.echo_canceller.available:
            recording_for_asr = self.echo_canceller.process(recording)

        text = self.transcribe(recording_for_asr)
        
        # 情感检测
        if text and self.emotion_enabled and self.emotion_add_to_context and self.last_emotion:
            text = f"[情感：{self.last_emotion}] {text}"
        
        return text
    
    def ptt_listen(self):
        '''
        PTT 录音
        
        使用 self._stop_listen 事件来控制何时停止录音
        1. 在 pressed 时调用此方法
        2. 在 released 时设置 self._stop_listen
        
        返回识别文本 或 None
        '''
        print(">>> [Ears] PTT 录音开始...")        
        self._stop_listen.clear()
        
        frames = []
        mic_stream = sd.InputStream(device=self.mic_id, channels=1, samplerate=self.samplerate, dtype='float32')
        loopback_stream, loopback_sr = self._get_loopback_stream()
        loopback_block = int(self.block_size * loopback_sr / self.samplerate)
        
        try:
            mic_stream.start()
            if loopback_stream:
                loopback_stream.start()
            
            while not self._stop_listen.is_set():
                raw_data, _ = mic_stream.read(self.block_size)
                loop_data = np.zeros(loopback_block)
                if loopback_stream:
                    try:
                        loop_data, _ = loopback_stream.read(loopback_block)
                    except Exception:
                        pass
                
                frames.append(raw_data)
        
        finally:
            mic_stream.stop(); mic_stream.close()
            if loopback_stream:
                try:
                    loopback_stream.stop()
                    loopback_stream.close()
                except Exception:
                    log.debug("[Ears] loopback 流关闭异常（可能已关闭）", exc_info=True)

        log.info("[Ears] PTT 录音结束 (%.1fs)", len(frames)*self.block_size/self.samplerate)
        
        if len(frames) < 2:
            return None
        
        recording = np.squeeze(np.concatenate(frames, axis=0))
        
        if len(recording) / self.samplerate < 0.3:
            log.debug("[Ears] 录音过短，已忽略")
            return None
        
        self._identify_speaker(recording)
        
        # ASR
        recording_for_asr = recording
        if self.ec_enabled and self.echo_canceller and self.echo_canceller.available:
            recording_for_asr = self.echo_canceller.process(recording)
        
        text = self.transcribe(recording_for_asr)
        
        # 情感检测
        if text and self.emotion_enabled and self.emotion_add_to_context and self.last_emotion:
            text = f"[情感：{self.last_emotion}] {text}"
        
        return text
    
    def start_interrupt_monitoring(self):
        '''
        后台监听打断
        '''
        self.interrupt_detected = False
        
        loopback_stream, loopback_sr = self._get_loopback_stream()
        loopback_block = int(self.block_size * loopback_sr / self.samplerate)
        
        if loopback_stream:
            loopback_stream.start()

        def audio_callback(indata, frames, time_info, status):
            # loopback 音频
            loop_data = np.zeros(loopback_block)
            if loopback_stream and loopback_stream.active:
                try:
                    loop_data, _ = loopback_stream.read(loopback_block) # 非阻塞读
                except Exception:
                    pass

            mic_rms = np.sqrt(np.mean(indata**2))
            loop_rms = np.sqrt(np.mean(loop_data**2))
            minus_rms = max(mic_rms - loop_rms, 0.0)

            if minus_rms > self.interrupt_vad_threshold:
                self.interrupt_detected = True

        self.monitoring_stream = sd.InputStream(
            device=self.mic_id,
            channels=1,
            samplerate=self.samplerate,
            dtype='float32',
            callback=audio_callback,
            blocksize=self.block_size
        )
        self.monitoring_stream.start()
        
        # 缓存
        self._int_loopback_stream = loopback_stream
        
        log.debug("[Ears] 后台打断监听已开启")

    
    def stop_interrupt_monitoring(self):
        '''停止后台监听'''
        if self.monitoring_stream:
            self.monitoring_stream.stop()
            self.monitoring_stream.close()
            self.monitoring_stream = None
            
        if hasattr(self, '_int_loopback_stream') and self._int_loopback_stream:
            try:
                self._int_loopback_stream.stop()
                self._int_loopback_stream.close()
            except Exception:
                pass
            self._int_loopback_stream = None
            
        log.debug("[Ears] 后台打断监听已关闭")

    
    def check_interrupt(self):
        '''检查是否检测到打断'''
        return self.interrupt_detected

    def transcribe(self, audio_data):
        if len(audio_data) < 100: 
            return None
        audio_data = np.squeeze(audio_data).astype(np.float32)
        try:
            import re
            # 重采样至原生 samplerate
            asr_sr = 16000
            if self.samplerate != asr_sr:
                import torch
                import torchaudio.functional as F
                t = torch.from_numpy(audio_data).unsqueeze(0)
                t = F.resample(t, self.samplerate, asr_sr)
                audio_data = t.squeeze(0).numpy()
            self._ensure_model()
            res = self.model.generate(
                input=audio_data, cache={}, language="zh",
                use_itn=True, batch_size_s=60,
            )
            
            # 提取文本
            text = res[0]["text"]
            clean_text = re.sub(r'<\|.*?\|>', '', text).strip()
            
            # 提取情感
            self.last_emotion = None
            if "emotion" in res[0]:
                try:
                    emotion_str = res[0]["emotion"].strip()
                    if emotion_str and emotion_str not in ["unknown", "<unk>", ""]:
                        self.last_emotion = emotion_str
                        log.debug("[Ears] 检测到情感: %s", emotion_str)
                except Exception:
                    pass
            
            if not clean_text or re.match(r'^[\s\W_]+$', clean_text):

                silence_rate = float(get_config("ears.silence_as_text_rate", 0.5))
                if random.random() >= silence_rate: # 随机丢弃无效噪音或触发 PO
                    log.debug("[Ears] 忽略无效噪音")
                    return None
                else:
                    clean_text = "[沉默]"
            return clean_text
        except Exception as e:
            import traceback
            log.error("[Ears] 识别出错: %s", e)
            traceback.print_exc()
            return None


