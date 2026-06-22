'''
Loopback 音频：从扬声器抓取声音，用来做回声消除的参考信号

两种后端：
  SdLoopbackStream        —— 接 Voicemeeter 之类的虚拟线
  SoundcardLoopbackStream —— Windows 原生 WASAPI，不需要装虚拟音频软件

用法统一：.start()  .read(n)  .stop()  .close()  .active
'''

import threading
import numpy as np
import sounddevice as sd
from math import gcd
from utils.logger import get_logger

log = get_logger(__name__)



# 后端 A：sounddevice（Voicemeeter兼容）

class SdLoopbackStream:
    '''套了一层 sounddevice InputStream，给 Voicemeeter B2 这类虚拟线用'''

    def __init__(self, device_id: int, samplerate: int, channels: int):
        self._stream = sd.InputStream(
            device=device_id, channels=channels,
            samplerate=samplerate, dtype='float32'
        )
        self.samplerate = samplerate

    @property
    def active(self) -> bool:
        return self._stream.active

    def start(self):
        self._stream.start()

    def read(self, n: int):
        '''返回 (data, overflow)，data shape = (n, channels)'''
        return self._stream.read(n)

    def stop(self):
        try:
            self._stream.stop()
        except Exception:
            pass

    def close(self):
        try:
            self._stream.close()
        except Exception:
            pass



# 后端 B：soundcard WASAPI（windows 原生） TODO： linux


class SoundcardLoopbackStream:
    '''Windows 原生 WASAPI loopback，后台线程读、环形缓冲、按需取

    .read(n) 非阻塞，缓冲不够就补零，不会卡住调用方
    '''

    _NATIVE_SR  = 48000   # WASAPI 固定 48k
    _CHUNK_SAMP = 960     # 每次读 20ms * 48kHz

    def __init__(self, samplerate: int = 16000, speaker_id: str | None = None):
        import soundcard as sc

        if speaker_id:
            self._recorder_cm = sc.get_microphone(speaker_id, include_loopback=True).recorder(
                samplerate=self._NATIVE_SR
            )
        else:
            spk = sc.default_speaker()
            self._recorder_cm = sc.get_microphone(spk.id, include_loopback=True).recorder(
                samplerate=self._NATIVE_SR
            )

        self.samplerate = samplerate

        # 重采样比例
        _g = gcd(self._NATIVE_SR, samplerate)
        self._up = samplerate // _g
        self._dn = self._NATIVE_SR // _g
        self._need_resample = (self._up != self._dn)

        # loop缓冲
        self._buf  = np.empty(0, dtype=np.float32)
        self._lock = threading.Lock()

        self._running  = False
        self._thread   = None
        self._recorder = None

    @property
    def active(self) -> bool:
        return self._running

    def start(self):
        self._running  = True
        self._recorder = self._recorder_cm.__enter__()
        self._thread   = threading.Thread(target=self._read_loop, daemon=True, name="sc-loopback")
        self._thread.start()
        log.debug("[Loopback] SoundcardLoopbackStream 已启动 (目标 %dHz)", self.samplerate)

    def _read_loop(self):
        from scipy.signal import resample_poly
        while self._running:
            try:
                data = self._recorder.record(numframes=self._CHUNK_SAMP)
                
                # 多声道转单声道
                mono = data.mean(axis=1) if data.ndim > 1 else data.ravel()
                if self._need_resample:
                    mono = resample_poly(mono, self._up, self._dn).astype(np.float32)
                else:
                    mono = mono.astype(np.float32)
                with self._lock:
                    self._buf = np.concatenate([self._buf, mono])
            except Exception as e:
                if self._running:
                    log.warning("[Loopback] 读取异常: %s", e)

    def read(self, n: int):
        '''
        读取音频样本，原理是：
        取 n 个样本，不够就补零，返回 (data, False)， data shape = (n, 1)
        '''
        with self._lock:
            if len(self._buf) >= n:
                out = self._buf[:n].copy()
                self._buf = self._buf[n:]
            else:
                out = np.zeros(n, dtype=np.float32)
        return out.reshape(-1, 1), False

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            self._recorder_cm.__exit__(None, None, None)
        except Exception:
            pass

    def close(self):
        pass



# 工厂


def make_loopback_stream(loopback_keyword: str, samplerate: int, loopback_device_id: int | None = None):
    '''按配置挑一个可用的 loopback 后端，返回 (stream, samplerate) 或 (None, samplerate)

    优先级：指定设备 ID -> Voicemeeter 虚拟线 -> WASAPI 默认扬声器
    '''
    from core.audio.audio_utils import find_device, get_device_channels

    # 用户选择id
    if loopback_device_id is not None:
        try:
            import soundcard as _sc
            dev_info = sd.query_devices(int(loopback_device_id))
            dev_name = dev_info['name'].strip().lower()
            matched_spk = None
            for spk in _sc.all_speakers():
                spk_name_l = spk.name.strip().lower()
                if dev_name in spk_name_l or spk_name_l in dev_name:
                    matched_spk = spk
                    break
            if matched_spk is None:
                log.warning("[Loopback] 未能按名称匹配 soundcard 扬声器 '%s'，使用默认扬声器",
                            dev_info['name'])
                matched_spk = _sc.default_speaker()
            stream = SoundcardLoopbackStream(samplerate=samplerate, speaker_id=matched_spk.id)
            log.debug("[Loopback] WASAPI loopback: [%d] %s -> %s @ %dHz",
                     loopback_device_id, dev_info['name'], matched_spk.name, samplerate)
            return stream, samplerate
        except ImportError:
            log.warning("[Loopback] soundcard 未安装，跳过设备指定 loopback")
        except Exception as e:
            log.warning("[Loopback] 指定设备 %d WASAPI loopback 失败: %s", loopback_device_id, e)

    # 否则走 voicemeeter 等虚拟线，匹配关键字
    if loopback_keyword:
        device_id = find_device(loopback_keyword)
        if device_id is not None:
            max_ch = get_device_channels(device_id)
            dev_name = sd.query_devices(device_id)['name']
            for sr in [samplerate, 48000, 44100, 24000]:
                for ch in [max_ch, 2, 1]:
                    try:
                        stream = SdLoopbackStream(device_id, sr, ch)
                        log.info("[Loopback] 虚拟线: [%d] %s @ %dHz (%dch)",
                                 device_id, dev_name, sr, ch)
                        return stream, sr
                    except Exception:
                        pass
            log.warning("[Loopback] 关键字 '%s' 设备存在但无法打开，回退到 WASAPI loopback",
                        loopback_keyword)
        else:
            log.info("[Loopback] 未找到 '%s'，尝试 WASAPI loopback", loopback_keyword)

    # 最差：WASAPI loopback
    try:
        import soundcard as _sc
        spk_name = _sc.default_speaker().name
        stream = SoundcardLoopbackStream(samplerate=samplerate)
        log.debug("[Loopback] WASAPI loopback (默认): %s @ %dHz", spk_name, samplerate)
        return stream, samplerate
    except ImportError:
        log.warning("[Loopback] soundcard 未安装，loopback 不可用 (pip install soundcard)")
    except Exception as e:
        log.warning("[Loopback] WASAPI loopback 初始化失败: %s", e)

    return None, samplerate
