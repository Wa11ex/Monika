"""
PyTorch 模型定义 — RewardModel + PolicyModel（训练用）

训练时用 PyTorch，推理时导出权重给 C++ 引擎。
两个模型共享 backbone 结构，方便 RM 的 hidden state 传给 Policy。

架构（与 C++ policy.h 对齐）：
  backbone: Linear(15→64) → LayerNorm → GELU → Linear(64→32) → LayerNorm → GELU
  RM head:  Linear(32→1) → Sigmoid → satisfaction [0,1]
  Policy heads: memory(32→1), use_tool(32→1), temperature(32→1),
                max_tokens(32→1), tool_choice(32→6), mode(32→3)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SharedBackbone(nn.Module):
    """共享脊骨 — RM 和 Policy 共用"""

    def __init__(self, d_in: int = 15, d_hidden: int = 64, d_out: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.ln1 = nn.LayerNorm(d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.ln2 = nn.LayerNorm(d_out)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 15) → (batch, 32)"""
        x = self.act(self.ln1(self.fc1(x)))
        x = self.act(self.ln2(self.fc2(x)))
        return x


class RewardModel(nn.Module):
    """用户满意度预测器 — KTO 训练

    输入: (batch, 15) 特征
    输出: (batch, 1) satisfaction ∈ [0, 1]
    """

    def __init__(self, d_in: int = 15):
        super().__init__()
        self.backbone = SharedBackbone(d_in)
        self.head = nn.Sequential(
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return torch.sigmoid(self.head(h))


class PolicyModel(nn.Module):
    """决策路由器 — 策略梯度训练

    输入: (batch, 15) 特征
    输出: dict of decisions (与 C++ PolicyOutput 对齐)
    """

    def __init__(self, d_in: int = 15):
        super().__init__()
        self.backbone = SharedBackbone(d_in)

        # 6 个输出头（与 C++ policy.h 完全对齐）
        self.head_memory = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.head_use_tool = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.head_temperature = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.head_max_tokens = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        self.head_tool_choice = nn.Linear(32, 6)   # logits（softmax 在 loss 里算）
        self.head_mode = nn.Linear(32, 3)

    def forward(self, x: torch.Tensor) -> dict:
        h = self.backbone(x)
        return {
            "memory_count":   self.head_memory(h) * 4 + 1,        # [1, 5]
            "use_tool_prob":  self.head_use_tool(h),               # [0, 1]
            "temperature":    self.head_temperature(h) * 1.0 + 0.5, # [0.5, 1.5]
            "max_tokens":     self.head_max_tokens(h) * 896 + 128,  # [128, 1024]
            "tool_logits":    self.head_tool_choice(h),             # (B, 6)
            "mode_logits":    self.head_mode(h),                    # (B, 3)
        }


def kto_loss(
    rm: RewardModel,
    good_features: torch.Tensor,
    bad_features: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """KTO 损失 — 只需单条标签，不需要偏好对

    good_features: (B_good, 15) 用户满意的轮次
    bad_features:  (B_bad, 15)  用户不满意的轮次
    beta: 温度系数，控制偏离参考点的力度

    数学：
      好样本: 最大化 sigmoid(beta * (score - ref))
      坏样本: 最大化 sigmoid(beta * (ref - score))
      ref = 0.5（中性参考点）
    """
    ref = 0.5

    if good_features.shape[0] > 0:
        good_scores = rm(good_features).squeeze(-1)
        loss_good = -torch.log(
            torch.sigmoid(beta * (good_scores - ref)) + 1e-8
        ).mean()
    else:
        loss_good = torch.tensor(0.0, device=rm.backbone.fc1.weight.device)

    if bad_features.shape[0] > 0:
        bad_scores = rm(bad_features).squeeze(-1)
        loss_bad = -torch.log(
            torch.sigmoid(beta * (ref - bad_scores)) + 1e-8
        ).mean()
    else:
        loss_bad = torch.tensor(0.0, device=rm.backbone.fc1.weight.device)

    return loss_good + loss_bad


def policy_gradient_loss(
    policy: PolicyModel,
    rm: RewardModel,
    features: torch.Tensor,
    baseline_temp: float = 1.0,
    baseline_tokens: int = 512,
) -> torch.Tensor:
    """简单策略梯度 — 最大化 E[reward * log_prob(action)]

    features: (B, 15) 所有样本（不分好坏）
    rm: 已训练的 RewardModel，提供 reward 信号
    baseline_temp/tokens: 基线参数，用于计算残差

    简化版 REINFORCE：
      1. Policy 输出 temperature/max_tokens
      2. RM 对 (features) 打分作为 reward
      3. loss = -mean(reward * sigmoid(temperature - baseline))
         鼓励高 reward 时输出更接近 baseline 的温度（不乱调）
    """
    decisions = policy(features)
    with torch.no_grad():
        rewards = rm(features).squeeze(-1)  # (B,)

    # 温度残差：Policy 输出与基线的差
    temp_residual = decisions["temperature"] - baseline_temp
    # 策略梯度：高 reward 时鼓励 |残差| 小（保守），低 reward 时鼓励探索
    # 用 tanh 限制范围，避免梯度爆炸
    advantage = rewards - 0.5  # 中心化
    loss_temp = -(advantage * torch.tanh(temp_residual)).mean()

    # max_tokens 同理
    token_residual = (decisions["max_tokens"] - baseline_tokens) / 512.0
    loss_tokens = -(advantage * torch.tanh(token_residual)).mean()

    # L2 正则化，防止 Policy 偏离基线太远
    l2_reg = sum(
        p.pow(2).sum() for p in policy.parameters() if p.requires_grad
    ) * 0.001

    return loss_temp + loss_tokens + l2_reg
