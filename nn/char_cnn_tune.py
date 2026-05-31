#!/usr/bin/env python3
"""CharCNN 优化: 两阶段 + 超参调优 + 全量训练"""
import json, re, time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA = Path('data')
OUT = Path('data/人工标注')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 扩展字符表（覆盖心理咨询常见词）
CHARS = set('abcdefghijklmnopqrstuvwxyz0123456789')
COMMON_CHARS = '的一是不了人在我有他这那中大小上到说会走时自家为以看好起学过如生动作发后出没开工面头部和企业级能于方式去向好问通路经点什么定对两三天来从就还用女'
PSYCH_CHARS = '心理情绪压力焦虑抑郁恐惧强迫悲伤愤怒痛苦绝望伤害死亡自杀攻击暴力报复学业考试工作家庭关系婚姻恋爱男女朋友父母孩子教育成绩毕业考研就业睡梦哭吃喝玩花钱想知觉见听说话读写看懂爱恨情仇'
ALL_CHARS = sorted(set(COMMON_CHARS + PSYCH_CHARS))
CHAR_TO_IDX = {c: i+1 for i, c in enumerate(ALL_CHARS)}
VOCAB_SIZE = len(CHAR_TO_IDX) + 1

# 加载数据
with open(OUT / 'No-01_待标注_3000_已标注.json', encoding='utf-8') as f:
    seed = json.load(f)

classes = sorted(set(item['labels']['label'] for item in seed))
class_to_idx = {c: i for i, c in enumerate(classes)}

def bld_text(item):
    p = [item.get('question_title',''), item.get('question_content','')]
    for a in item.get('answers',[]):
        for d in a.get('dialogs',[]): p.append(d.get('content',''))
    return ' '.join(p)

texts = [bld_text(it) for it in seed]
labels = [class_to_idx[it['labels']['label']] for it in seed]

def to_char_seq(text, max_len):
    text = re.sub(r'\s+', '', text)[:max_len]
    seq = [CHAR_TO_IDX.get(c, 0) for c in text]
    return (seq + [0]*max_len)[:max_len]

# === CharCNN v2: 更深的网络 ===
class CharCNNv2(nn.Module):
    def __init__(self, vocab_size, n_classes, embed_dim=128, max_len=400):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 多尺度卷积
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, 128, ks, padding=ks//2) for ks in [2, 3, 4, 5]
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(128) for _ in range(4)])
        # 全局池化 + 全连接
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4, 256),
            nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )
    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        convs = []
        for conv, bn in zip(self.convs, self.bns):
            h = torch.relu(bn(conv(x)))
            convs.append(self.pool(h).squeeze(-1))
        return self.fc(torch.cat(convs, dim=1))

def train_model(model, X_tr, y_tr, X_te, y_te, epochs=30, lr=0.001, bs=64):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X_tr_t = torch.tensor(X_tr, dtype=torch.long)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_te_t = torch.tensor(X_te, dtype=torch.long).to(device)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), bs, shuffle=True)
    best = 0
    best_state = None
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            criterion(model(bx), by).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_te_t).argmax(dim=1).cpu().numpy()
            acc = accuracy_score(y_te, preds)
            if acc > best:
                best = acc
                best_state = model.state_dict()
        if (ep+1) % 5 == 0:
            print(f"  Epoch {ep+1}/{epochs} acc={acc:.4f} ({time.time()-t0:.1f}s)")

    # 恢复最佳模型
    if best_state:
        model.load_state_dict(best_state)
    return best

print("=" * 60)
print("CharCNN v2 优化")
print("=" * 60)

# ===== 实验1: 两阶段 =====
print("\n【实验1】两阶段 CharCNN")
s_levels = [l[0] for l in labels]

# Stage1: S1/S2/S3
s1_mask = [l == '1' for l in s_levels]
s2_mask = [l == '2' for l in s_levels]
s3_mask = [l == '3' for l in s_levels]

for name, mask, max_len, epochs in [
    ('S1/S2/S3 层级', None, 300, 20),
    ('S1子类(17类)', s1_mask, 400, 30),
    ('S2子类(9类)', s2_mask, 400, 20),
    ('S3子类(5类)', s3_mask, 400, 20),
]:
    if mask is None:
        sub_texts = texts
        sub_labels = s_levels
        n_cls = 3
    else:
        sub_texts = [t for t, m in zip(texts, mask) if m]
        sub_labels = [l for l, m in zip(labels, mask) if m]
        # 重编码
        sub_cls = sorted(set(sub_labels))
        sub_c2i = {c: i for i, c in enumerate(sub_cls)}
        sub_labels = [sub_c2i[l] for l in sub_labels]
        n_cls = len(sub_cls)

    # 处理成字符序列
    X = np.array([to_char_seq(t, max_len) for t in sub_texts])
    y_arr = np.array(sub_labels)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_arr, test_size=0.2, random_state=42)

    model = CharCNNv2(VOCAB_SIZE, n_cls, max_len=max_len)
    acc = train_model(model, X_tr, y_tr, X_te, y_te, epochs=epochs)
    print(f"  >> {name}: {acc:.4f} ({len(sub_texts)}条, {n_cls}类)\n")

# ===== 实验2: 超参对比 =====
print("\n【实验2】超参对比 (S1+S2+S3, 3类)")
X_3 = np.array([to_char_seq(t, 300) for t in texts])
y_3 = np.array([class_to_idx[l] for l in labels])
X_tr, X_te, y_tr, y_te = train_test_split(X_3, y_3, test_size=0.2, random_state=42)

for embed_dim, lr, label in [(64, 0.001, 'base'), (128, 0.001, 'large'), (64, 0.0005, 'slow_lr'), (128, 0.0005, 'large_slow')]:
    model = CharCNNv2(VOCAB_SIZE, len(classes), embed_dim=embed_dim, max_len=300)
    acc = train_model(model, X_tr, y_tr, X_te, y_te, epochs=20, lr=lr)
    print(f"  {label}: embed={embed_dim} lr={lr} acc={acc:.4f}\n")

print(f"\n设备: {device} | 字符表: {len(ALL_CHARS)}字")
