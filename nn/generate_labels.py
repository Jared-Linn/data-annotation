#!/usr/bin/env python3
"""
用 Deep v3 模型生成全量高质量标签
=================================

流程：
  1. 加载 Deep v3 模型（best）
  2. 对全部 25 万条对话做 S 层级预测
  3. 筛选高置信度（>0.9）的预测 → 替代原始伪标签
  4. 保存为新标签文件，供 BERT 训练使用

输出: data/人工标注/pseudo_labeled_refined.json
"""

import json
import re
import time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

# ── 字符表（必须与训练时一致） ──
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
MODEL_PATH = Path('nn/models/char_cnn_deep_best.pt')
OUTPUT_PATH = Path('data/人工标注/pseudo_labeled_refined.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Deep v3 模型（必须与 char_cnn_deep.py 一致） ──
class ResidualConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, dilation=1, stride=1):
        super().__init__()
        p = k // 2 * dilation
        self.conv1 = nn.Conv1d(in_c, out_c, k, padding=p, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, k, padding=p, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.shortcut = nn.Sequential()
        if in_c != out_c or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, 1, stride=stride), nn.BatchNorm1d(out_c))
        self.relu = nn.ReLU()

    def forward(self, x):
        o = self.relu(self.bn1(self.conv1(x)))
        o = self.bn2(self.conv2(o))
        return self.relu(o + self.shortcut(x))


class CharCNN_Deep_v3(nn.Module):
    def __init__(self, vocab_size, n_classes=3, embed_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
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
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 256),
            nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        f3 = self.branch3(x).squeeze(-1)
        f5 = self.branch5(x).squeeze(-1)
        f7 = self.branch7(x).squeeze(-1)
        p = self.avg_pool(x).squeeze(-1)
        return self.fc(torch.cat([f3, f5, f7, p], dim=1))


def main():
    print("=" * 60)
    print("Deep v3 全量标签生成")
    print(f"设备: {device}")
    print("=" * 60)

    # ── 1. 加载模型 ──
    print(f"\n▶ 加载模型: {MODEL_PATH}")
    model = CharCNN_Deep_v3(VOCAB_SIZE, n_classes=3).to(device)
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ── 2. 加载数据 ──
    print(f"\n▶ 加载数据: {DATA_PATH}")
    with open(DATA_PATH, encoding='utf-8') as f:
        data = json.load(f)
    print(f"  共 {len(data)} 条")

    # ── 3. 准备特征 ──
    print(f"\n▶ 构建特征...")
    t0 = time.time()
    texts = []
    for item in data:
        parts = [item.get('question_title', ''), item.get('question_content', '')]
        for a in item.get('answers', []):
            for d in a.get('dialogs', []):
                parts.append(d.get('content', ''))
        texts.append(' '.join(parts))

    # 转字符序列
    X = np.zeros((len(texts), MAX_LEN), dtype=np.int32)
    for i, t in enumerate(texts):
        t = re.sub(r'\s+', '', t)[:MAX_LEN]
        for j, c in enumerate(t):
            X[i, j] = C2I.get(c, 0)
    print(f"  特征矩阵: {X.shape} ({time.time()-t0:.1f}s)")

    # ── 4. 预测（分 batch 防 OOM） ──
    print(f"\n▶ 预测 S 层级...")
    t0 = time.time()
    all_probs = []
    all_preds = []
    batch_size = 512

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.tensor(X[i:i+batch_size]).to(device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            batch_probs, batch_preds = probs.max(dim=1)
            all_probs.extend(batch_probs.cpu().numpy())
            all_preds.extend(batch_preds.cpu().numpy())
            if (i // batch_size) % 50 == 0:
                print(f"  {i}/{len(X)} ({i/len(X)*100:.0f}%)")

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    s_map = {0: '1', 1: '2', 2: '3'}
    s_preds = [s_map[p] for p in all_preds]
    print(f"  预测完成 ({time.time()-t0:.1f}s)")
    print(f"  平均置信度: {all_probs.mean():.4f}")
    print(f"  高置信度(>0.9): {(all_probs>0.9).sum()}/{len(all_probs)}")

    # ── 5. 生成新标签 ──
    print(f"\n▶ 生成新标签...")
    n_changed = 0
    for i, item in enumerate(data):
        old_label = item['labels']['label']
        s_level = s_preds[i]
        confidence = all_probs[i]

        # 保留原子类标签（保留原始伪标签的子类部分，替换S层级）
        # 格式: '1.3' → 新S层级 + 原子类后缀
        old_sub = old_label.split('.')[-1]  # '3' from '1.3'

        if confidence > 0.9:
            # 高置信度：用模型预测的S层级 + 保留子类
            new_label = f"{s_level}.{old_sub}"
        else:
            # 低置信度：保留原始标签
            new_label = old_label

        item['labels']['label'] = new_label
        # 记录模型置信度
        item['_s_confidence'] = float(confidence)
        item['_s_pred'] = s_level
        n_changed += 1 if new_label != old_label else 0

    print(f"  标签改动: {n_changed} 条")

    # ── 6. 统计 & 保存 ──
    stats = Counter(it['labels']['label'] for it in data)
    print(f"\n  标签分布:")
    for label in sorted(stats.keys()):
        pct = stats[label] / len(data) * 100
        print(f"    {label}: {stats[label]:>6}条 ({pct:5.1f}%)")

    print(f"\n▶ 保存: {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # 同时生成一份只包含新标签的轻量级训练文件（不带_pseudo等字段，BERT用）
    slim_path = Path('data/人工标注/bert_training_data.json')
    slim_data = []
    for item in data:
        slim_data.append({
            'text': texts[data.index(item)],
            'label': item['labels']['label'],
        })
    # 更高效的方式
    slim_data = []
    for i, item in enumerate(data):
        slim_data.append({
            'text': texts[i],
            'label': item['labels']['label'],
        })
    with open(slim_path, 'w', encoding='utf-8') as f:
        json.dump(slim_data, f, ensure_ascii=False)
    print(f"  轻量版: {slim_path} ({len(slim_data)}条)")

    print(f"\n完成! 生成 {len(data)} 条带标签数据")
    print(f"标签文件: {OUTPUT_PATH}")
    print(f"BERT训练: {slim_path}")


if __name__ == '__main__':
    main()
