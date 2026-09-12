"""
权重导出 — PyTorch PolicyModel → flat binary（C++ PolicyModel::load 格式）

二进制布局（与 rl/rl_engine/include/rl_engine/policy.h::load() 对齐）：
  fc1.weight (15*64) | fc1.bias (64)
  ln1.gamma (64)     | ln1.bias (64)
  fc2.weight (64*32) | fc2.bias (32)
  ln2.gamma (32)     | ln2.bias (32)
  head_memory.weight (32*1) | head_memory.bias (1)
  head_use_tool.weight (32*1) | head_use_tool.bias (1)
  head_temperature.weight (32*1) | head_temperature.bias (1)
  head_max_tokens.weight (32*1) | head_max_tokens.bias (1)
  head_tool_choice.weight (32*6) | head_tool_choice.bias (6)
  head_mode.weight (32*3) | head_mode.bias (3)

所有权重以 float32 连续排列，无 header。
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch

from rl.models import PolicyModel


def _linear_to_flat(layer: torch.nn.Linear) -> np.ndarray:
    """Linear 层 → (weight, bias) flat arrays

    PyTorch weight shape: (out, in)，C++ 期望 (in, out) → 需要转置
    """
    w = layer.weight.detach().cpu().numpy().T.astype(np.float32)  # (in, out)
    b = layer.bias.detach().cpu().numpy().astype(np.float32)      # (out,)
    return w.ravel(), b.ravel()


def _layernorm_to_flat(layer: torch.nn.LayerNorm) -> tuple:
    """LayerNorm → (gamma, beta) flat arrays"""
    g = layer.weight.detach().cpu().numpy().astype(np.float32)  # gamma
    b = layer.bias.detach().cpu().numpy().astype(np.float32)    # beta
    return g.ravel(), b.ravel()


def _sequential_linear_to_flat(seq: torch.nn.Sequential) -> tuple:
    """从 Sequential(Linear, Sigmoid) 提取 Linear 层"""
    for m in seq:
        if isinstance(m, torch.nn.Linear):
            return _linear_to_flat(m)
    raise ValueError("Sequential 里没找到 Linear 层")


def export_policy_weights(policy: PolicyModel, out_path: str) -> None:
    """导出 PolicyModel 权重为 C++ 可读的 flat binary

    Args:
        policy: 训练好的 PyTorch PolicyModel
        out_path: 输出文件路径（如 rl/weights/policy.bin）
    """
    # 按顺序收集所有权重（与 policy.h::load() 的读取顺序一致）
    chunks = []

    # backbone: fc1, ln1, fc2, ln2
    w, b = _linear_to_flat(policy.backbone.fc1)
    chunks += [w, b]
    g, be = _layernorm_to_flat(policy.backbone.ln1)
    chunks += [g, be]
    w, b = _linear_to_flat(policy.backbone.fc2)
    chunks += [w, b]
    g, be = _layernorm_to_flat(policy.backbone.ln2)
    chunks += [g, be]

    # 6 个输出头
    # memory, use_tool, temperature, max_tokens 都是 Sequential(Linear, Sigmoid)
    for head in [policy.head_memory, policy.head_use_tool,
                 policy.head_temperature, policy.head_max_tokens]:
        w, b = _sequential_linear_to_flat(head)
        chunks += [w, b]

    # tool_choice, mode 是纯 Linear
    w, b = _linear_to_flat(policy.head_tool_choice)
    chunks += [w, b]
    w, b = _linear_to_flat(policy.head_mode)
    chunks += [w, b]

    # 拼接 + 写入
    total = np.concatenate(chunks)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(total.tobytes())

    # 验证：打印每层尺寸
    print(f"[Export] {out} ({len(total) * 4} bytes, {len(total)} floats)")
    _verify_layout(policy)


def _verify_layout(policy: PolicyModel):
    """打印每层权重尺寸，方便对照 C++ load() 顺序"""
    layers = [
        ("fc1.weight", policy.backbone.fc1.weight, (15, 64)),
        ("fc1.bias", policy.backbone.fc1.bias, (64,)),
        ("ln1.gamma", policy.backbone.ln1.weight, (64,)),
        ("ln1.beta", policy.backbone.ln1.bias, (64,)),
        ("fc2.weight", policy.backbone.fc2.weight, (64, 32)),
        ("fc2.bias", policy.backbone.fc2.bias, (32,)),
        ("ln2.gamma", policy.backbone.ln2.weight, (32,)),
        ("ln2.beta", policy.backbone.ln2.bias, (32,)),
    ]
    for name, param, expected in layers:
        actual = tuple(param.shape)
        print(f"  {name}: pytorch={actual} cpp_expected={expected}")
        # PyTorch Linear weight 是 (out, in)，C++ 是 (in, out)
        if "weight" in name and len(expected) == 2:
            assert actual == expected[::-1], f"{name} shape mismatch!"
