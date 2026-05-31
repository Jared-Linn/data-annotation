#!/usr/bin/env python3
"""轻量神经网络: 字符级CNN + MLP 对比实验"""
import json, re, time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DATA = Path('data')
OUT = Path('data/人工标注')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 字符表（涵盖中英文常见字）
CHARS = 'abcdefghijklmnopqrstuvwxyz0123456789的一是不了人在我有他这那中大小上到说会走时自家为以看好起学过如生动作发后出没开工面头部和企业级能于方式去向好问通路经点什么定对两三天来从就还用女'
CHARS += '心理情绪压力焦虑抑郁恐惧强迫悲伤愤怒痛苦绝望伤害死亡自杀攻击暴力报复学业考试工作家庭关系婚姻恋爱男女朋友父母孩子教育成绩毕业考研就业'
CHAR_TO_IDX = {c: i+1 for i, c in enumerate(set(CHARS))}
VOCAB_SIZE = len(CHAR_TO_IDX) + 1
MAX_LEN = 300

def char_seq(text):
    text = re.sub(r'\s+', '', text)[:MAX_LEN]
    seq = [CHAR_TO_IDX.get(c, 0) for c in text]
    if len(seq) < MAX_LEN:
        seq += [0] * (MAX_LEN - len(seq))
    return seq[:MAX_LEN]

def bld_text(item):
    p = [item.get('question_title',''), item.get('question_content','')]
    for a in item.get('answers',[]):
        for d in a.get('dialogs',[]): p.append(d.get('content',''))
    return ' '.join(p)

with open(OUT / 'No-01_待标注_3000_已标注.json', encoding='utf-8') as f:
    seed = json.load(f)

classes = sorted(set(item['labels']['label'] for item in seed))
class_to_idx = {c: i for i, c in enumerate(classes)}
n_classes = len(classes)
texts = [bld_text(it) for it in seed]
labels = [class_to_idx[it['labels']['label']] for it in seed]
y = np.array(labels)

print(f"数据: {len(texts)}条, {n_classes}类")

# MLP模型
class MLP(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )
    def forward(self, x): return self.net(x)

# 字符CNN
class CharCNN(nn.Module):
    def __init__(self, vocab_size, n_classes, embed_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(embed_dim, 64, ks), nn.ReLU(), nn.AdaptiveMaxPool1d(1))
            for ks in [3, 5, 7]
        ])
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(64*3, n_classes))
    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        x = [conv(x).squeeze(-1) for conv in self.convs]
        return self.fc(torch.cat(x, dim=1))

def train_model(model, X_tr, y_tr, X_te, y_te, epochs=30, bs=64):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    X_tr_t = torch.from_numpy(X_tr) if isinstance(X_tr, np.ndarray) else X_tr
    y_tr_t = torch.from_numpy(y_tr) if isinstance(y_tr, np.ndarray) else y_tr
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), bs, shuffle=True)
    best = 0
    for ep in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            criterion(model(bx), by).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            X_te_t = torch.from_numpy(X_te).to(device) if isinstance(X_te, np.ndarray) else X_te.to(device)
            preds = model(X_te_t).argmax(dim=1).cpu().numpy()
            a = accuracy_score(y_te, preds)
            if a > best: best = a
        if (ep+1)%10==0: print(f"  Epoch {ep+1}/{epochs} acc={a:.4f}")
    return best

# 实验1: LR + TF-IDF 基线
print(f"\n实验1: LR + TF-IDF")
v = TfidfVectorizer(ngram_range=(1,1), max_features=5000, sublinear_tf=True)
X = v.fit_transform(texts).toarray().astype(np.float32)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
lr = LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced', random_state=42)
lr.fit(X_tr, y_tr)
print(f"  LR + TF-IDF: {accuracy_score(y_te, lr.predict(X_te)):.4f}")

# 实验2: MLP + TF-IDF
print(f"\n实验2: MLP + TF-IDF")
t0 = time.time()
mlp_acc = train_model(MLP(X.shape[1], n_classes), X_tr, y_tr, X_te, y_te)
print(f"  MLP + TF-IDF: {mlp_acc:.4f} ({time.time()-t0:.0f}s)")

# 实验3: CharCNN
print(f"\n实验3: 字符级 CNN")
X_char = np.array([char_seq(t) for t in texts])
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_char, y, test_size=0.2, random_state=42, stratify=y)
t0 = time.time()
cnn_acc = train_model(CharCNN(VOCAB_SIZE, n_classes), X_tr_c, y_tr_c, X_te_c, y_te_c, epochs=20)
print(f"  CharCNN: {cnn_acc:.4f} ({time.time()-t0:.0f}s)")

print(f"\n{'='*50}")
print("对比总结")
print(f"{'='*50}")
print(f"  LR + TF-IDF:     ~0.45")
print(f"  MLP + TF-IDF:    {mlp_acc:.4f}")
print(f"  CharCNN:         {cnn_acc:.4f}")
