"""
RL 训练入口 — KTO 训练 RewardModel + 策略梯度训练 PolicyModel

用法：
    python -m rl.train                    # 训练 + 导出
    python -m rl.train --eval-only        # 只评估不训练
    python -m rl.train --min-samples 50   # 最少 50 样本才训练
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from rl.dataset import RLDataset
from rl.models import RewardModel, PolicyModel, kto_loss, policy_gradient_loss


def train_rm(
    rm: RewardModel,
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 16,
) -> dict:
    """KTO 训练 RewardModel

    features: (N, 15) 所有样本
    labels:   (N,) 1=good, 0=bad, -1=neutral(丢弃)
    """
    good_mask = labels == 1
    bad_mask = labels == 0
    n_good = int(good_mask.sum())
    n_bad = int(bad_mask.sum())

    if n_good < 2 or n_bad < 2:
        print(f"[RM] 数据不足: good={n_good} bad={n_bad}，至少各需 2 条")
        return {"rm_loss": float("inf"), "n_good": n_good, "n_bad": n_bad}

    good_feats = torch.tensor(features[good_mask], dtype=torch.float32)
    bad_feats = torch.tensor(features[bad_mask], dtype=torch.float32)

    optimizer = torch.optim.AdamW(rm.parameters(), lr=lr, weight_decay=0.01)

    print(f"[RM] 开始 KTO 训练: good={n_good} bad={n_bad} epochs={epochs}")
    for epoch in range(epochs):
        # 随机打乱
        perm_g = torch.randperm(n_good)
        perm_b = torch.randperm(n_bad)
        g = good_feats[perm_g]
        b = bad_feats[perm_b]

        # 对齐 batch（循环采样少数类）
        n_batches = max(n_good, n_bad) // batch_size + 1
        epoch_loss = 0.0
        for i in range(n_batches):
            g_batch = g[i * batch_size:(i + 1) * batch_size]
            b_batch = b[i * batch_size:(i + 1) * batch_size]
            if len(g_batch) == 0 or len(b_batch) == 0:
                continue
            # 少数类循环补齐
            if len(g_batch) < len(b_batch):
                g_batch = g_batch.repeat(len(b_batch) // len(g_batch) + 1, 1)[:len(b_batch)]
            elif len(b_batch) < len(g_batch):
                b_batch = b_batch.repeat(len(g_batch) // len(b_batch) + 1, 1)[:len(g_batch)]

            loss = kto_loss(rm, g_batch, b_batch, beta=0.1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{epochs}  loss={epoch_loss/n_batches:.4f}")

    # 评估：好样本均分 vs 坏样本均分
    with torch.no_grad():
        good_scores = rm(good_feats).mean().item()
        bad_scores = rm(bad_feats).mean().item()
    print(f"[RM] 训练完成: good_mean={good_scores:.3f} bad_mean={bad_scores:.3f} "
          f"gap={good_scores - bad_scores:.3f}")

    return {"rm_loss": epoch_loss / n_batches, "good_mean": good_scores,
            "bad_mean": bad_scores, "n_good": n_good, "n_bad": n_bad}


def train_policy(
    policy: PolicyModel,
    rm: RewardModel,
    features: np.ndarray,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 16,
) -> dict:
    """策略梯度训练 PolicyModel"""
    feats = torch.tensor(features, dtype=torch.float32)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.01)

    print(f"[Policy] 开始策略梯度训练: samples={len(feats)} epochs={epochs}")
    for epoch in range(epochs):
        perm = torch.randperm(len(feats))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(feats), batch_size):
            batch = feats[perm[i:i + batch_size]]
            if len(batch) < 2:
                continue
            loss = policy_gradient_loss(policy, rm, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{epochs}  loss={epoch_loss/max(n_batches,1):.4f}")

    print("[Policy] 训练完成")
    return {"policy_loss": epoch_loss / max(n_batches, 1)}


def main():
    parser = argparse.ArgumentParser(description="Monika RL 训练")
    parser.add_argument("--log-dir", default="rl/logs", help="日志目录")
    parser.add_argument("--min-samples", type=int, default=10, help="最少样本数")
    parser.add_argument("--rm-epochs", type=int, default=50, help="RM 训练轮数")
    parser.add_argument("--policy-epochs", type=int, default=30, help="Policy 训练轮数")
    parser.add_argument("--eval-only", action="store_true", help="只评估不训练")
    parser.add_argument("--export", action="store_true", help="训练后导出 C++ 权重")
    args = parser.parse_args()

    # 加载数据
    ds = RLDataset(args.log_dir)
    stats = ds.stats()
    print(f"[Data] {stats}")

    if stats["total"] < args.min_samples:
        print(f"[Data] 样本不足 ({stats['total']} < {args.min_samples})，跳过训练")
        return

    features, rewards, labels = ds.load()

    if args.eval_only:
        print("[Eval] --eval-only 模式，跳过训练")
        return

    # 训练 RM
    rm = RewardModel()
    rm_metrics = train_rm(rm, features, labels, epochs=args.rm_epochs)
    if rm_metrics.get("rm_loss") == float("inf"):
        print("[RM] 训练失败，数据不足")
        return

    # 训练 Policy
    policy = PolicyModel()
    policy_metrics = train_policy(policy, rm, features, epochs=args.policy_epochs)

    # 保存 PyTorch checkpoint
    ckpt_dir = Path("rl/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"rm": rm.state_dict(), "policy": policy.state_dict()},
               ckpt_dir / "latest.pt")
    print(f"[Save] checkpoint → {ckpt_dir / 'latest.pt'}")

    # 导出 C++ 权重
    if args.export:
        from rl.export import export_policy_weights
        out_path = Path("rl/weights/policy.bin")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        export_policy_weights(policy, str(out_path))
        print(f"[Export] C++ 权重 → {out_path}")


if __name__ == "__main__":
    main()
