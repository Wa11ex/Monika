'''
口型同步 MFCC 分析模块

算法：uLipSync MFCC 管道
  1. 预加重 -> Hamming 窗 -> FFT -> Mel 滤波器组 -> log -> DCT -> MFCC
  2. Z-score 归一化
  3. 与自动标定的参考向量做余弦相似度
  4. pow(SHARP) + softmax -> AIUEO 分布

主要入口：
  analyze_wav_bytes(wav_bytes) -> (times, open_y, scores)
'''

from __future__ import annotations

import io
import numpy as np
import librosa
import soundfile as sf
from scipy.fft import dct as _scipy_dct

# -- 微调常量 -------------------------------------------------
HOP_MS         = 10      # 帧步长 ms
FRAME_MS       = 30      # 分析窗口 ms
N_MFCC         = 13      # MFCC 系数数（不含 C0）
N_MELS         = 24      # Mel 滤波器组数
SMOOTH_K       = 5       # 平滑帧数
SHARP          = 20.0    # 余弦锐化指数
GAMMA          = 0.55    # RMS 伽玛校正
OPEN_THRESHOLD = 0.08    # 闭嘴阈值
CALIB_PCTL     = 15      # 自动标定百分位数

# Mel 滤波器组缓存（避免每帧重复计算）
_mel_fb_cache: dict = {}


# -- 内部工具 -------------------------------------------------

def _compute_mfcc(y: np.ndarray, sr: int, hop_len: int, frame_len: int) -> np.ndarray:
    '''批量 MFCC，返回 (N_MFCC, n_frames)'''
    y_pe = np.append(y[0], y[1:] - 0.97 * y[:-1])
    mfcc_full = librosa.feature.mfcc(
        y=y_pe, sr=sr,
        n_mfcc=N_MFCC + 1,
        n_fft=frame_len,
        hop_length=hop_len,
        n_mels=N_MELS,
        window="hamming",
    )
    return mfcc_full[1:, :]   # (N_MFCC, n_frames)


def _auto_references(mfcc_z: np.ndarray, rms_norm: np.ndarray) -> np.ndarray:
    '''提取 A/I/U/E/O 参考向量，返回 (5, N_MFCC)'''
    voiced = rms_norm > 0.15
    if voiced.sum() < 10:
        voiced = rms_norm > rms_norm.mean() * 0.3

    m   = mfcc_z[:, :voiced.shape[0]]
    m_v = m[:, voiced[:m.shape[1]]]      # (N_MFCC, voiced_frames)

    c1 = m_v[0]; c2 = m_v[1]
    hi_c1  = np.percentile(c1, 100 - CALIB_PCTL)
    lo_c1  = np.percentile(c1, CALIB_PCTL)
    hi_c2  = np.percentile(c2, 100 - CALIB_PCTL)
    lo_c2  = np.percentile(c2, CALIB_PCTL)
    mid_c1 = np.median(c1)
    mid_c2 = np.median(c2)

    def _ref(c1_tgt, c2_tgt):
        mask = (np.abs(m_v[0] - c1_tgt) < 0.5) & (np.abs(m_v[1] - c2_tgt) < 0.5)
        if mask.sum() < 5:
            dist = (m_v[0] - c1_tgt) ** 2 + (m_v[1] - c2_tgt) ** 2
            idx  = np.argsort(dist)[:max(5, int(mask.sum()) + 3)]
            mask = np.zeros(m_v.shape[1], bool)
            mask[idx] = True
        return m_v[:, mask].mean(axis=1)

    return np.stack([
        _ref(hi_c1,  mid_c2),   # A
        _ref(lo_c1,  hi_c2),    # I
        _ref(lo_c1,  lo_c2),    # U
        _ref(mid_c1, hi_c2),    # E
        _ref(mid_c1, lo_c2),    # O
    ], axis=0)   # (5, N_MFCC)



# -- 公开调用区 -------------------------------------------------

def analyze_lipsync_raw(
    y: np.ndarray, sr: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    输入 PCM 数据，返回逐帧唇形参数
    Return:
        times  : (n,) 帧时间戳（秒）
        open_y : (n,) RMS-based MouthOpenY [0, 1]
        scores : (n, 5) AIUEO softmax 得分
        refs   : (5, N_MFCC) 自动标定参考向量
    '''
    hop_len   = max(1, int(sr * HOP_MS   / 1000))
    frame_len = max(1, int(sr * FRAME_MS / 1000))

    rms        = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    rms_norm   = rms / (rms.max() + 1e-8)
    open_y_raw = np.clip(rms_norm ** GAMMA, 0.0, 1.0)

    mfcc   = _compute_mfcc(y, sr, hop_len, frame_len)
    mean   = mfcc.mean(axis=1, keepdims=True)
    std    = mfcc.std(axis=1,  keepdims=True) + 1e-8
    mfcc_z = (mfcc - mean) / std

    n          = min(mfcc_z.shape[1], len(rms_norm))
    mfcc_z     = mfcc_z[:, :n]
    rms_norm   = rms_norm[:n]
    open_y_raw = open_y_raw[:n]

    refs = _auto_references(mfcc_z, rms_norm)

    frames_n = mfcc_z.T / (np.linalg.norm(mfcc_z.T, axis=1, keepdims=True) + 1e-8)
    refs_n   = refs     / (np.linalg.norm(refs,       axis=1, keepdims=True) + 1e-8)
    sim      = np.clip(frames_n @ refs_n.T, 0, 1)
    sharp    = sim ** SHARP
    scores   = sharp / (sharp.sum(axis=1, keepdims=True) + 1e-30)

    def smooth(x: np.ndarray, k: int = SMOOTH_K) -> np.ndarray:
        return np.convolve(x, np.ones(k) / k, mode="same")

    open_y = smooth(open_y_raw)
    for v in range(5):
        scores[:, v] = smooth(scores[:, v])

    scores[rms_norm < 0.08] = 0.0
    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=hop_len)
    return times, open_y, scores, refs


def analyze_wav_bytes(
    wav_bytes: bytes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    输入 TTS 返回的 WAV 字节，返回逐帧唇形参数
    Return: 
        times  : (n,) 帧时间戳（秒）
        open_y : (n,) MouthOpenY [0, 1]       — RMS 驱动开合度
        scores : (n, 5) AIUEO 得分（和为 1）
    '''
    y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)

    times, open_y, scores, _ = analyze_lipsync_raw(y, sr)

    return (
        times,
        open_y.astype(np.float32),
        scores.astype(np.float32),
    )


# -- 多参数嘴型驱动（基于 VBridger 实验数据） ---------------------------------
#
# 实验流程：每个元音发音 3 次，记录 VBridger 输出（= VTS 输入），
# 再通过 Mk3_VB.vtube.json 中的线性映射反算出 VTS 输出（Param*）
#
# 线性映射公式：out = OutLow + (in - InLow) / (InHigh - InLow) * (OutHigh - OutLow)
# 参数映射表来自 assets/live2d/…/Mk3_VB.vtube.json 的 ParameterSettings


_MOUTH_PARAM_NAMES: list[str] = [
    "ParamMouthOpenY",       # [0, 1]
    "ParamMouthForm",        # [-1, 1]  正=扁唇 I/E，负=圆唇 U/O
    "ParamMouthFunnel",      # [0, 1]
    "ParamMouthPressLipOpen",# [-1, 1]
    "ParamMouthShrug",       # [0, 1]
    "ParamMouthPuckerWiden", # [-1, 1]  负=撅嘴 U/O，正=嘴角展开
    "ParamJawOpen",          # [0, 1]
]

_VOWEL_KEYPOSE = np.array([
    #  OpenY   Form  Funnel  PressLip  Shrug  PuckerW  JawOpen
    [ 0.825,  0.910,  0.010,  0.594,  0.033,  0.037,  0.933],  # A
    [ 0.284,  1.000,  0.017,  0.258,  0.010,  0.000,  0.212],  # I
    [ 0.071,  0.100,  0.127, -0.177,  0.000, -0.950,  0.000],  # U
    [ 0.800,  1.000,  0.020,  0.232,  0.014,  1.000,  0.500],  # E
    [ 0.300, -0.487,  0.120,  1.000,  0.000, -1.000,  0.800],  # O
], dtype=np.float32)

# 归零
MOUTH_PARAMS_RESET: dict = {n: 0.0 for n in _MOUTH_PARAM_NAMES}


def scores_to_params_multi(
    scores: np.ndarray,  # (5,) AIUEO 置信度，和为 1
    open_y: float,       # RMS 口腔开合度 [0, 1]，用作静音门限
) -> dict:
    '''
    AIUEO 置信度 + RMS 开合度 -> Param* 参数字典，直接写入 Live2D coreModel

    方法：对 5 个元音关键姿态做加权平均，
    再乘以 open_y 使静音时闭嘴归零
    '''
    weighted = float(open_y) * (scores @ _VOWEL_KEYPOSE)  # (7,)
    return {
        name: float(np.clip(v, -1.0, 1.0))
        for name, v in zip(_MOUTH_PARAM_NAMES, weighted)
    }
