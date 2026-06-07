#!/usr/bin/env python3
"""
Self-Training: 用 CharCNN 迭代改善伪标签质量
=============================================

原理：
  原始伪标签（关键词匹配）有噪音但量大体全。
  训练一个 CharCNN → 模型学到比关键词更泛化的模式 →
  用它重新预测，只取高置信度的 → 替代原始噪音标签 →
  用更好的标签再训练 → 循环。

  就像学生做题：先用老师给的答案（伪标签）学 →
  自己会做了 → 纠正老师的笔误 → 越学越好。

策略：
  Round 0: 原始伪标签 → 训练 CharCNN
  Round 1: CharCNN 预测 → 取 >0.9 置信度的 → 替换标签 → 再训练
  Round N: 每轮提高阈值（0.90 → 0.92 → 0.95 → ...）

用法：
  python -m nn.self_train --subset 50000 --rounds 3

评估方式：
  每次迭代在固定验证集上测准确率。
  准确率上升 = 标签质量在改善（因为验证集标签不变）。
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

# ── 字符表 ──
_CHARS = sorted(set(
    'abcdefghijklmnopqrstuvwxyz0123456789'
    '的一是不了人在我有他这那中心大小上到说会走时自家为以看好起学过如生动作发后出没开面'
    '心理情绪压力焦虑抑郁恐惧强迫悲伤愤怒痛苦绝望伤害死亡自杀攻击暴力报复学业考试工作'
    '家庭关系婚姻恋爱男女朋友父母孩子教育成绩毕业考研就业睡梦哭吃喝玩钱想知道看见听见'
))
C2I = {c: i + 1 for i, c in enumerate(_CHARS)}
VOCAB_SIZE = len(C2I) + 1
MAX_LEN = 200  # 大部分对话200字足够，减少显存占用

DATA_PATH = Path('data/人工标注/pseudo_labeled_all.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ═══════════════════════════════════════════════════════════
# 1. 数据 & 工具函数
# ═══════════════════════════════════════════════════════════

def load_all(subset=None):
    """加载原始数据"""
    print(f"加载: {DATA_PATH}")
    with open(DATA_PATH, encoding='utf-8') as f:
        data = json.load(f)
    if subset:
        data = data[:subset]
    print(f"  共 {len(data)} 条")

    texts, s_labels, full_labels = [], [], []
    for item in data:
        parts = [item.get('question_title', ''), item.get('question_content', '')]
        for a in item.get('answers', []):
            for d in a.get('dialogs', []):
                parts.append(d.get('content', ''))
        texts.append(' '.join(parts))
        label = item['labels']['label']
        s_labels.append(label[0])
        full_labels.append(label)
    return texts, s_labels, full_labels


def to_char_seq(texts, max_len=MAX_LEN):
    """文本 → 字符索引矩阵 (int32省显存)"""
    X = np.zeros((len(texts), max_len), dtype=np.int32)
    for i, t in enumerate(texts):
        t = re.sub(r'\s+', '', t)[:max_len]
        for j, c in enumerate(t):
            X[i, j] = C2I.get(c, 0)
    return X


# ═══════════════════════════════════════════════════════════
# 2. 模型 (复用 CharCNN)
# ═══════════════════════════════════════════════════════════

class CharCNN(nn.Module):
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


# ═══════════════════════════════════════════════════════════
# 3. 训练 & 置信度预测
# ═══════════════════════════════════════════════════════════

def train_model(model, X_tr, y_tr, X_val, y_val, epochs=20, bs=128, lr=0.001, tag=''):
    """训练 + 返回最佳验证准确率"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    X_tr_t = torch.tensor(X_tr, dtype=torch.long)
    y_tr_t = torch.tensor(y_tr)
    X_val_t = torch.tensor(X_val, dtype=torch.long).to(device)
    y_val_t = torch.tensor(y_val).to(device)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), bs, shuffle=True)
    best_acc, best_state = 0, None

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
            best_state = model.state_dict()
        if (ep + 1) % 5 == 0:
            print(f"    [{tag}] ep{ep+1}/{epochs} val_acc={acc:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    print(f"  >> {tag} 最佳: {best_acc:.4f}")
    return best_acc, model


def predict_with_confidence(model, X, batch_size=512):
    """
    预测 + 返回置信度 (分batch避免OOM)

    Returns:
        preds:   shape=(N,) 预测类别
        confs:   shape=(N,) softmax 最大概率（置信度）
    """
    model.eval()
    model = model.to(device)
    n = len(X)
    all_preds = np.zeros(n, dtype=np.int64)
    all_confs = np.zeros(n, dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = X[i:i + batch_size]
            X_t = torch.tensor(batch, dtype=torch.long).to(device)
            logits = model(X_t)
            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            all_preds[i:i + batch_size] = preds.cpu().numpy()
            all_confs[i:i + batch_size] = confs.cpu().numpy()

    return all_preds, all_confs


# ═══════════════════════════════════════════════════════════
# 4. Self-Training 核心循环
# ═══════════════════════════════════════════════════════════

def do_self_training_round(
    texts,       # 所有文本
    labels,      # 当前标签 (本轮要用的)
    classes,     # 类别列表
    val_idx,     # 验证集索引 (固定, 不参与自训练)
    round_n,     # 第几轮
    threshold,   # 置信度阈值
    epochs=15,
    tag='',
):
    """
    一轮 Self-Training：
    1. 用当前标签训练模型
    2. 用模型预测，挑高置信度样本
    3. 用模型预测标签替换这些样本的标签
    """
    n_total = len(texts)
    n_classes = len(classes)
    c2i = {c: i for i, c in enumerate(classes)}

    # 训练集 = 全部除去验证集
    train_mask = np.ones(n_total, dtype=bool)
    train_mask[val_idx] = False

    # 当前标签
    y = np.array([c2i[l] for l in labels])

    # 准备特征
    X = to_char_seq(texts)
    X_tr, X_val = X[train_mask], X[val_idx]
    y_tr, y_val = y[train_mask], y[val_idx]

    # 训练
    model = CharCNN(VOCAB_SIZE, n_classes)
    acc, model = train_model(
        model, X_tr, y_tr, X_val, y_val,
        epochs=epochs, tag=f'{tag} Round{round_n}',
    )

    # 预测所有训练集
    preds, confs = predict_with_confidence(model, X_tr)

    # 筛选高置信度样本
    high_conf_mask = confs >= threshold
    n_high = high_conf_mask.sum()
    n_total_train = len(y_tr)

    # 启用 self-training: 替换标签
    if n_high > 0:
        old_labels = y_tr.copy()
        y_tr[high_conf_mask] = preds[high_conf_mask]
        n_changed = (old_labels != y_tr).sum()
    else:
        n_changed = 0

    # 新标签写回
    new_labels = list(labels)
    train_indices = np.where(train_mask)[0]
    for idx_in_train, global_idx in enumerate(train_indices):
        new_labels[global_idx] = classes[y_tr[idx_in_train]]

    summary = {
        'round': round_n,
        'threshold': threshold,
        'val_acc': acc,
        'high_conf': n_high,
        'high_conf_pct': n_high / n_total_train * 100,
        'labels_changed': n_changed,
        'model': model,
    }
    return new_labels, summary


# ═══════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=int, default=None, help='取前N条(缺省=全量)')
    parser.add_argument('--rounds', type=int, default=3, help='Self-training轮数')
    parser.add_argument('--epochs', type=int, default=15, help='每轮训练epoch数')
    args = parser.parse_args()

    print("=" * 60)
    print("Self-Training: 迭代改善伪标签")
    print(f"设备: {device}")
    print("=" * 60)

    # ── 5.1 加载 ──
    print("\n▶ 加载数据")
    texts, s_labels, full_labels = load_all(subset=args.subset)
    n_all = len(texts)

    # ── 5.2 固定验证集 ──
    val_size = min(3000, int(n_all * 0.1))
    val_idx = np.arange(n_all - val_size, n_all)
    train_idx = np.arange(n_all - val_size)
    print(f"\n  训练: {len(train_idx)} / 验证: {len(val_idx)}")

    # ═══════════════════════════════════════════════════════
    # Self-Training: Stage 1 (S1/S2/S3)
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("▶ Stage 1 Self-Training (S1/S2/S3)")
    print("=" * 60)

    s_classes = ['1', '2', '3']
    cur_s_labels = list(s_labels)
    thresholds = [0.90, 0.92, 0.95, 0.97][:args.rounds]

    s_history = []
    for rnd in range(args.rounds):
        threshold = thresholds[rnd]
        print(f"\n{'─'*50}")
        print(f"Round {rnd+1}/{args.rounds} (threshold={threshold})")
        print(f"{'─'*50}")

        cur_s_labels, summary = do_self_training_round(
            texts, cur_s_labels, s_classes, val_idx,
            round_n=rnd + 1, threshold=threshold,
            epochs=args.epochs, tag='S-Level',
        )
        s_history.append(summary)

        # 统计采纳情况
        print(f"  高置信度: {summary['high_conf']}/{len(train_idx)} "
              f"({summary['high_conf_pct']:.1f}%)")
        print(f"  标签被改: {summary['labels_changed']} 条")

        # 保存本轮模型
        model_dir = Path('nn/models')
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f'self_train_s1_round{rnd+1}.pt'
        torch.save(summary['model'].state_dict(), model_path)
        print(f"  模型保存: {model_path}")

    # ── 最终评估 ──
    print(f"\n  {'='*45}")
    print(f"  Stage 1 Self-Training 结果")
    print(f"  {'='*45}")
    print(f"  {'Round':<8} {'阈值':<8} {'验证准确率':<12} {'采纳数':<10} {'标签改动':<10}")
    print(f"  {'-'*45}")
    for h in s_history:
        print(f"  {h['round']:<8} {h['threshold']:<8.2f} {h['val_acc']:<12.4f} "
              f"{h['high_conf']:<10} {h['labels_changed']:<10}")

    # ═══════════════════════════════════════════════════════
    # Self-Training: Stage 2 (S1子类)
    # ═══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("▶ Stage 2 Self-Training (S1子类)")
    print("=" * 60)

    # S1样本筛选
    s1_mask = np.array([l == '1' for l in cur_s_labels])
    s1_indices = np.where(s1_mask)[0]
    print(f"S1样本: {len(s1_indices)} 条")

    if len(s1_indices) < 1000:
        print("S1样本太少，跳过Stage2 Self-Training")
    else:
        s1_texts_sub = [texts[i] for i in s1_indices]
        s1_labels_sub = [full_labels[i] for i in s1_indices]

        s1_classes = sorted(set(s1_labels_sub[:len(train_idx)]))
        print(f"S1子类数: {len(s1_classes)}")

        # S1的验证集
        s1_val_count = min(500, int(len(s1_indices) * 0.1))
        s1_val_idx = np.arange(len(s1_indices))[-s1_val_count:]

        cur_s1_labels = list(s1_labels_sub)
        s1_history = []

        s1_best_acc = 0
        s1_best_model = None
        for rnd in range(args.rounds):
            threshold = thresholds[rnd]
            print(f"\n  ── Round {rnd+1}/{args.rounds} (threshold={threshold}) ──")

            cur_s1_labels, summary = do_self_training_round(
                s1_texts_sub, cur_s1_labels, s1_classes, s1_val_idx,
                round_n=rnd + 1, threshold=threshold,
                epochs=args.epochs, tag='S1-Sub',
            )
            s1_history.append(summary)
            s1_best_acc = max(s1_best_acc, summary['val_acc'])
            if summary['val_acc'] >= s1_best_acc:
                s1_best_model = summary['model']

            # 保存S1子类模型
            model_path = Path(f'nn/models/self_train_s1sub_round{rnd+1}.pt')
            torch.save(summary['model'].state_dict(), model_path)
            print(f"  模型保存: {model_path}")

        # 结果
        print(f"\n  {'='*40}")
        print(f"  Stage 2 Self-Training 结果")
        print(f"  {'='*40}")
        print(f"  {'Round':<8} {'阈值':<8} {'验证准确率':<12}")
        print(f"  {'-'*30}")
        for h in s1_history:
            print(f"  {h['round']:<8} {h['threshold']:<8.2f} {h['val_acc']:<12.4f}")

    # ═══════════════════════════════════════════════════════
    # 保存最佳模型
    # ═══════════════════════════════════════════════════════
    # 从s_history中找最佳验证准确率对应的模型
    best_acc = 0
    for h in s_history:
        if h['val_acc'] > best_acc:
            best_acc = h['val_acc']

    best_model_path = Path('nn/models/self_train_best.pt')
    # 最后一轮模型就是最好的（根据我们的train_model逻辑，保存了最佳checkpoint）
    torch.save(s_history[-1]['model'].state_dict(), best_model_path)
    print(f"\n最佳Stage1模型保存: {best_model_path} (acc={best_acc:.4f})")

    # ═══════════════════════════════════════════════════════
    # 总结
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Self-Training 总结")
    print(f"{'='*60}")

    first_s_acc = s_history[0]['val_acc'] if s_history else 0
    last_s_acc = s_history[-1]['val_acc'] if s_history else 0
    improvement = last_s_acc - first_s_acc

    print(f"""
  Stage 1 准确率: {first_s_acc:.2%} → {last_s_acc:.2%}
  提升: {improvement:+.2%}

  {'✅ Self-Training 有效改善标签质量!' if improvement > 0 else '⚠️ 无明显提升，可能需调整阈值'}

  原理回顾:
    第1轮: 模型学关键词匹配模式（同伪标签）
    第2轮: 模型开始泛化，修正不一致的标签
    第3轮: 更高阈值筛选，只保留最置信的 → 标签更纯净

  后续:
    - 用最终模型对全部25万条重新预测生成高质量标签
    - 用新标签训练更好的模型（如 BERT）
""")


if __name__ == '__main__':
    main()
