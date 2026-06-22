'''
回音消除模块 - 基于 DeepFilterNet 用来做背景消噪，实际上是做不了ec的，
在加载时加了 torchaudio.backend.common 兼容性 shim，修复 torchaudio 2.x 的 API 变更
'''

import sys
import types
import dataclasses
import numpy as np

# -- torchaudio 2.x 兼容 shim ----------------------------------
# DeepFilterNet 的 df/io.py 依赖 torchaudio.backend.common.AudioMetaData
# 该类在 torchaudio 2.x 中被移除，这里注入一个最小兼容实现
def _patch_torchaudio():
    if 'torchaudio.backend' in sys.modules:
        return

    @dataclasses.dataclass
    class AudioMetaData:
        sample_rate: int = 0
        num_frames: int = 0
        num_channels: int = 0
        bits_per_sample: int = 0
        encoding: str = ""

    backend_mod = types.ModuleType('torchaudio.backend')
    common_mod  = types.ModuleType('torchaudio.backend.common')
    common_mod.AudioMetaData = AudioMetaData
    backend_mod.common = common_mod

    sys.modules['torchaudio.backend']        = backend_mod
    sys.modules['torchaudio.backend.common'] = common_mod

_patch_torchaudio()

# DeepFilterNet 延迟导入：只在 EchoCanceller 被实例化时才 import，
# 这样 config 关闭 DFN 时完全不加载 torch/df，不占 VRAM 也不拖慢启动
_df_available = None   # None = 尚未尝试加载
_df_import_error = ""


def _ensure_df_loaded():
    '''首次调用时尝试导入 DeepFilterNet，之后复用结果'''
    global _df_available, _df_import_error
    if _df_available is not None:
        return _df_available
    try:
        import torch           # noqa: F401  — 副作用：初始化 CUDA context
        from df.enhance import enhance, init_df  # noqa: F401
        _df_available = True
    except Exception as e:
        _df_available = False
        _df_import_error = str(e)
    return _df_available


class EchoCanceller:
    '''
    DeepFilterNet 消噪
    '''

    DF_SR = 48000   # DeepFilterNet 原生采样率！

    def __init__(self, samplerate: int = 24000):
        self.samplerate = samplerate
        self._model = None
        self._df_state = None

        if not _ensure_df_loaded():
            print(f"!!! [EC] DeepFilterNet 不可用: {_df_import_error}")
            print("!!! [EC] 回音消除已禁用")
        else:
            try:
                import torch                        # noqa: F401
                from df.enhance import init_df
                # DeepFilterNet v0.5.x 的 DFState（Rust 扩展）与 CUDA 紧耦合，
                # 保持默认 GPU 行为，没法改到 CPU 别试了
                # 模型本身约 50-100 MB VRAM，代价可接受
                self._model, self._df_state, _ = init_df()
                print(f">>> [EC] DeepFilterNet 加载成功 (原生 sr={self.DF_SR}Hz)")
            except Exception as e:
                print(f"!!! [EC] 加载 DeepFilterNet 失败: {e}")

    @property
    def available(self) -> bool:
        return self._model is not None

    def process(self, audio: np.ndarray) -> np.ndarray:
        '''
        对单帧/整段音频做回音消除

        Args:
            audio: float32 数组，shape (N,) 或 (N,1)，范围 [-1, 1]，采样率 = self.samplerate

        Returns:
            同形状、同采样率的 float32 数组
        '''
        if not self.available:
            return audio

        audio = np.squeeze(audio.astype(np.float32))   # -> (N,)

        try:
            import torchaudio.functional as F

            t = torch.from_numpy(audio).unsqueeze(0)   # (1, N)，报错没有问题，import在前面，不然卡死了

            # 重采样到原生 48kHz
            if self.samplerate != self.DF_SR:
                t = F.resample(t, self.samplerate, self.DF_SR)

            # DeepFilterNet 处理，报错没有问题，import在前面，不然卡死了
            enhanced = enhance(self._model, self._df_state, t)  # (1, N')

            # 还回采样率
            if self.samplerate != self.DF_SR:
                enhanced = F.resample(enhanced, self.DF_SR, self.samplerate)

            result = enhanced.squeeze(0).numpy()

            # TODO ；这里的逻辑是补零或截断，确保与输入长度相同，算是兜底，目前有效
            if len(result) < len(audio):
                result = np.pad(result, (0, len(audio) - len(result)))
            else:
                result = result[:len(audio)]

            return result

        except Exception as e:
            print(f"!!! [EC] 处理出错: {e}")
            return audio
