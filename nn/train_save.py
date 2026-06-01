#!/usr/bin/env python3
"""训练 CharCNN Stage1 模型并保存到 nn/models/"""
import json, re, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

OUT = Path('data/人工标注')
MODEL_DIR = Path('nn/models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# 字符表（覆盖心理咨询常用字）
CHARS = sorted(set('abcdefghijklmnopqrstuvwxyz0123456789' +
    '的一是不了人在我有他这那中心大小上到说会走时自家为以看好起学过如生动作发后出没开面'
    '心理情绪压力焦虑抑郁恐惧强迫悲伤愤怒痛苦绝望伤害死亡自杀攻击暴力报复学业考试工作'
    '家庭关系婚姻恋爱男女朋友父母孩子教育成绩毕业考研就业睡梦哭吃喝玩钱想知道看见听见'))
C2I = {c:i+1 for i,c in enumerate(CHARS)}
VOCAB = len(C2I) + 1

def seq(t, m=300):
    s = [C2I.get(c,0) for c in t[:m]]
    return (s+[0]*m)[:m]

class CharCNN(nn.Module):
    def __init__(self, vocab, ncls):
        super().__init__()
        self.emb = nn.Embedding(vocab, 128, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(128, 64, k, padding=k//2), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveMaxPool1d(1))
            for k in [3,5,7]
        ])
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(64*3, ncls))
    def forward(self, x):
        x = self.emb(x).permute(0,2,1)
        x = torch.cat([c(x).squeeze(-1) for c in self.convs], dim=1)
        return self.fc(x)

# 加载数据
with open(OUT / 'No-01_待标注_3000_已标注.json', encoding='utf-8') as f:
    seed = json.load(f)

with open('data/stopwords.txt', encoding='utf-8') as f:
    SW = set(l.strip() for l in f if l.strip())

import jieba
def cut(t): return ' '.join(w for w in jieba.cut(t) if w.strip() and w not in SW)
def cln(t): return re.sub(r'\s+','',re.sub(r'[^一-鿿\w]','',t))
def bld(it):
    p = [it.get('question_title',''), it.get('question_content','')]
    for a in it.get('answers',[]):
        for d in a.get('dialogs',[]): p.append(d.get('content',''))
    return cut(cln(' '.join(p)))

txts = [bld(it) for it in seed]
s_levels = [it['labels']['label'][0] for it in seed]
y = np.array([{'1':0,'2':1,'3':2}[l] for l in s_levels])

print(f"训练 CharCNN Stage1: {len(txts)}条")

X = np.array([seq(t) for t in txts])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = CharCNN(VOCAB, 3)

# 类别权重（解决 S2/S3 样本不足）
from collections import Counter
cls_counts = Counter(y)
total = sum(cls_counts.values())
weights = [total / cls_counts[i] for i in range(3)]
weights = torch.tensor(weights, dtype=torch.float32)
print(f"类别权重: S1={weights[0]:.2f} S2={weights[1]:.2f} S3={weights[2]:.2f}")

opt = optim.Adam(model.parameters(), lr=0.001)
ld = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), 64, shuffle=True)
best = 0
criterion = nn.CrossEntropyLoss(weight=weights)

for ep in range(25):
    model.train()
    for bx, by in ld:
        opt.zero_grad()
        criterion(model(bx), by).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        a = accuracy_score(y_te, model(torch.tensor(X_te)).argmax(dim=1).numpy())
        if a > best:
            best = a
            torch.save(model.state_dict(), MODEL_DIR / 'char_cnn_stage1.pt')
    if (ep+1) % 5 == 0:
        print(f"  Epoch {ep+1}/25 acc={a:.4f}")

print(f"最佳: {best:.4f}")
print(f"模型已保存: {MODEL_DIR / 'char_cnn_stage1.pt'}")
