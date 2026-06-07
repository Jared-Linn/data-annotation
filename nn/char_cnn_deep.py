#!/usr/bin/env python3
"""
CharCNN 深度改进版 — 残差连接 + 深层卷积
========================================

架构对比：
  Original:   Embed → 1层conv(k=3,5,7) → 池化 → FC
  Deep v3:    Embed → 2×残差块(k=3) → 2×残差块(k=5) → 2×残差块(k=7) → 双池化 → FC
  Deep v4:    Embed → 4×残差块(k=3, 膨胀卷积) → 全局特征 → FC

残差连接原理：
  传统深层网络的问题：梯度消失（反向传播时梯度越传越小）
  残差连接: 输出 = F(x) + x
  即使 F(x) 学不到东西，梯度也能通过 x 直接传回 → 允许更深的网络

用法：
  python -m nn.char_cnn_deep --subset 50000 --epochs 15
"""

import json
import re
import time
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── 字符表（与原版一致） ──
_CHARS = sorted(set(
    'abcdefghijklmnopqrstuvwxyz0123456789'
    '的一是不了人在我有他这那中心大小上到说会走时自家为以看好起学过如生动作发后出没开面'
    '心理情绪压力焦虑抑郁恐惧强迫悲伤愤怒痛苦绝望伤害死亡自杀攻击暴力报复学业考试工作'
    '家庭关系婚姻恋爱男女朋友父母孩子教育成绩毕业考研就业睡梦哭吃喝玩钱想知道看见听见'
))
C2I = {c: i + 1 for i, c in enumerate(_CHARS)}
VOCAB_SIZE = len(C2I) + 1
MAX_LEN = 200

DATA_PATH = Path('data/人工标注/pseudo_labeled_all.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ═══════════════════════════════════════════════════════════
# 1. 模型定义
# ═══════════════════════════════════════════════════════════

class ResidualConvBlock(nn.Module):
    """
    残差卷积块

    结构: Conv → BN → ReLU → Conv → BN → (+输入) → ReLU

    当输入输出通道不同时，用 1×1 卷积对齐维度
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1):
        super().__init__()
        padding = kernel_size // 2 * dilation  # 保持序列长度

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 维度匹配：当 in ≠ out 或 stride≠1 时，用 1×1 conv 调整 shortcut
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        # 主路径
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 残差连接: F(x) + x
        out += self.shortcut(x)
        return self.relu(out)


class CharCNN_Original(nn.Module):
    """原版 CharCNN（对比基线）"""
    def __init__(self, vocab_size, n_classes, embed_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, 64, k, padding=k // 2),
                nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveMaxPool1d(1),
            ) for k in [3, 5, 7]
        ])
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(64 * 3, n_classes))

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        x = torch.cat([conv(x).squeeze(-1) for conv in self.convs], dim=1)
        return self.fc(x)


class CharCNN_Deep_v3(nn.Module):
    """
    CharCNN v3 — 深度残差版

    架构:
      Embed(128)
      → 2×ResBlock(k=3, 128→128) → MaxPool
      → 2×ResBlock(k=5, 128→256) → MaxPool
      → 2×ResBlock(k=7, 256→512) → MaxPool
      → 全局平均池化 + 全局最大池化 (双通道)
      → Dropout → FC

    参数量约: 2.5× 原版，但残差连接保证可训练
    """
    def __init__(self, vocab_size, n_classes, embed_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 三个尺度分支，每个分支2层残差块
        self.branch3 = nn.Sequential(
            ResidualConvBlock(embed_dim, 128, 3),
            ResidualConvBlock(128, 128, 3),
            nn.AdaptiveMaxPool1d(1),
        )
        self.branch5 = nn.Sequential(
            ResidualConvBlock(embed_dim, 128, 5),
            ResidualConvBlock(128, 256, 5),
            nn.AdaptiveMaxPool1d(1),
        )
        self.branch7 = nn.Sequential(
            ResidualConvBlock(embed_dim, 128, 7),
            ResidualConvBlock(128, 256, 7),
            nn.AdaptiveMaxPool1d(1),
        )

        # 双池化：平均 + 最大
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # 分支输出: branch3=128, branch5=256, branch7=256 + avg_pool=128
        feat_dim = 128 + 256 + 256 + 128  # = 768

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 256),
            nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)  # (B, E, L)

        # 三分支（最大池化）
        f3 = self.branch3(x).squeeze(-1)  # (B, 128)
        f5 = self.branch5(x).squeeze(-1)  # (B, 256)
        f7 = self.branch7(x).squeeze(-1)  # (B, 256)

        # 平均池化（互补特征）
        p = self.avg_pool(x).squeeze(-1)  # (B, 128)

        features = torch.cat([f3, f5, f7, p], dim=1)  # (B, 768)
        return self.fc(features)


class CharCNN_Deep_v4(nn.Module):
    """
    CharCNN v4 — 串联深层 + 膨胀卷积

    架构:
      Embed(128)
      → ResBlock(k=3, d=1) ×3   (感受野: 7)
      → ResBlock(k=3, d=2) ×3   (感受野: 13)
      → ResBlock(k=3, d=3) ×3   (感受野: 19)
      → 全局平均+最大池化
      → FC

    膨胀卷积(Dilated Conv)：
      在卷积核之间插入空洞，扩大感受野而不增加参数量
      dilation=1: 感受野3  dilation=2: 感受野5  dilation=3: 感受野7
      适合捕捉长距离字符依赖
    """
    def __init__(self, vocab_size, n_classes, embed_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 3个stage，每个stage膨胀系数递增
        self.stage1 = nn.Sequential(
            ResidualConvBlock(embed_dim, 128, 3, dilation=1),
            ResidualConvBlock(128, 128, 3, dilation=1),
            nn.AdaptiveMaxPool1d(1),
        )
        self.stage2 = nn.Sequential(
            ResidualConvBlock(embed_dim, 128, 3, dilation=2),
            ResidualConvBlock(128, 256, 3, dilation=2),
            nn.AdaptiveMaxPool1d(1),
        )
        self.stage3 = nn.Sequential(
            ResidualConvBlock(embed_dim, 128, 3, dilation=3),
            ResidualConvBlock(128, 256, 3, dilation=3),
            nn.AdaptiveMaxPool1d(1),
        )

        # 平均池化（全局特征）
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        feat_dim = 128 + 256 + 256 + 128  # = 768
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 256),
            nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        f1 = self.stage1(x).squeeze(-1)
        f2 = self.stage2(x).squeeze(-1)
        f3 = self.stage3(x).squeeze(-1)
        p = self.avg_pool(x).squeeze(-1)
        return self.fc(torch.cat([f1, f2, f3, p], dim=1))


# ═══════════════════════════════════════════════════════════
# 2. 数据工具
# ═══════════════════════════════════════════════════════════

def load_data(subset=None):
    with open(DATA_PATH, encoding='utf-8') as f:
        data = json.load(f)
    if subset:
        data = data[:subset]
    texts, labels = [], []
    for item in data:
        parts = [item.get('question_title', ''), item.get('question_content', '')]
        for a in item.get('answers', []):
            for d in a.get('dialogs', []):
                parts.append(d.get('content', ''))
        texts.append(' '.join(parts))
        labels.append(item['labels']['label'][0])  # S级
    return texts, labels


def to_char_seq(texts, max_len=MAX_LEN):
    X = np.zeros((len(texts), max_len), dtype=np.int32)
    for i, t in enumerate(texts):
        t = re.sub(r'\s+', '', t)[:max_len]
        for j, c in enumerate(t):
            X[i, j] = C2I.get(c, 0)
    return X


# ═══════════════════════════════════════════════════════════
# 3. 训练
# ═══════════════════════════════════════════════════════════

def train_and_eval(model, X_tr, y_tr, X_val, y_val, epochs, tag=''):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    X_tr_t = torch.tensor(X_tr)
    y_tr_t = torch.tensor(y_tr)
    X_val_t = torch.tensor(X_val).to(device)
    y_val_t = torch.tensor(y_val).to(device)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), 128, shuffle=True)
    best_acc = 0
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            criterion(model(bx), by).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_val_t).argmax(dim=1)
            acc = (preds == y_val_t).float().mean().item()
        if acc > best_acc:
            best_acc = acc
        if (ep + 1) % 5 == 0:
            print(f"    [{tag:>12}] ep{ep+1:2d}/{epochs} val_acc={acc:.4f}")

    print(f"  >> {tag:>12} 最佳: {best_acc:.4f} ({time.time()-t0:.1f}s)")
    return best_acc


def count_params(model):
    """统计参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════
# 4. Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=int, default=50000)
    parser.add_argument('--epochs', type=int, default=15)
    args = parser.parse_args()

    print("=" * 60)
    print("CharCNN 深度改进 — 模型对比")
    print(f"设备: {device}")
    print("=" * 60)

    # ── 加载 ──
    print("\n▶ 加载数据")
    texts, labels = load_data(subset=args.subset)
    s_map = {'1': 0, '2': 1, '3': 2}
    y = np.array([s_map[l] for l in labels])
    print(f"  共 {len(texts)} 条")

    # ── 特征 ──
    X = to_char_seq(texts)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=3000, random_state=42, stratify=y
    )
    print(f"  训练: {len(X_tr)} / 验证: {len(X_val)}")

    # ── 模型定义 ──
    models = [
        ('Original', CharCNN_Original(VOCAB_SIZE, 3)),
        ('Deep v3', CharCNN_Deep_v3(VOCAB_SIZE, 3)),
        ('Deep v4', CharCNN_Deep_v4(VOCAB_SIZE, 3)),
    ]

    results = []
    print(f"\n{'='*60}")
    print("▶ 模型训练对比")
    print(f"{'='*60}\n")

    for name, model in models:
        n_params = count_params(model)
        print(f"  {name}: {n_params:,} 参数")
        acc = train_and_eval(model, X_tr, y_tr, X_val, y_val,
                             epochs=args.epochs, tag=name)
        results.append((name, n_params, acc))

    # ── 结果对比 ──
    print(f"\n{'='*60}")
    print("📊 模型对比总结 (S1/S2/S3)")
    print(f"{'='*60}")
    print(f"  {'模型':<16} {'参数量':<12} {'验证准确率':<12} {'提升':<10}")
    print(f"  {'-'*50}")
    base_acc = results[0][2]
    for name, n_params, acc in results:
        gain = acc - base_acc
        print(f"  {name:<16} {n_params:<12,} {acc:<12.4f} {gain:>+8.4f}")

    # ── 参数量 vs 性能分析 ──
    print(f"\n  {'='*50}")
    print(f"  分析")
    print(f"  {'='*50}")
    for name, n_params, acc in results:
        ratio = n_params / results[0][1]
        gain = (acc / results[0][2] - 1) * 100
        print(f"  {name}: {ratio:.1f}x参数 → {gain:+.2f}%准确率")

    # ── 保存最佳模型 ──
    best_idx = max(range(len(results)), key=lambda i: results[i][2])
    best_name, _, best_acc = results[best_idx]
    print(f"\n  ✅ 最佳: {best_name} (acc={best_acc:.4f})")

    model_dir = Path('nn/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    save_path = model_dir / 'char_cnn_deep_best.pt'
    torch.save(models[best_idx][1].state_dict(), save_path)
    print(f"  模型已保存: {save_path}")


if __name__ == '__main__':
    main()
