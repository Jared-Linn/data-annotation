#!/usr/bin/env python3
"""CharCNN + TF-IDF 特征融合"""
import json, re as re_m, time
from pathlib import Path
import numpy as np, jieba
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

OUT = Path('data/人工标注')
device = 'cpu'

CHARS = sorted(set('abcdefghijklmnopqrstuvwxyz0123456789的一是不了人在我有他这那中心理情绪压力焦虑抑郁恐惧强迫悲伤愤怒痛苦绝望死亡自杀攻击报复学业考试工作家庭关系婚姻恋爱朋友父母孩子教育成绩毕业考研就业睡梦哭吃喝玩花钱想知道看见'))
C2I = {c:i+1 for i,c in enumerate(CHARS)}

with open(OUT / 'No-01_待标注_3000_已标注.json', encoding='utf-8') as f:
    seed = json.load(f)

def cln(t): return re_m.sub(r'\s+','',re_m.sub(r'[^一-鿿\w]','',t))
with open('data/stopwords.txt', encoding='utf-8') as f:
    SW = set(l.strip() for l in f if l.strip())
def cut(t): return ' '.join(w for w in jieba.cut(t) if w.strip() and w not in SW)

def bld(it):
    p = [it.get('question_title',''), it.get('question_content','')]
    for a in it.get('answers',[]):
        for d in a.get('dialogs',[]): p.append(d.get('content',''))
    return cut(cln(' '.join(p)))

txts = [bld(it) for it in seed]
lbls = [it['labels']['label'] for it in seed]
s_levels = [l[0] for l in lbls]

def seq(t, m=300):
    s = [C2I.get(c,0) for c in t[:m]]
    return (s+[0]*m)[:m]

# CharCNN
class CharCNN(nn.Module):
    def __init__(s, vocab, ncls):
        super().__init__()
        s.emb = nn.Embedding(vocab, 128, padding_idx=0)
        s.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(128, 64, k, padding=k//2), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveMaxPool1d(1))
            for k in [3,5,7]
        ])
        s.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(64*3, ncls))
    def forward(s, x):
        x = s.emb(x).permute(0,2,1)
        x = torch.cat([c(x).squeeze(-1) for c in s.convs], dim=1)
        return s.fc(x)

# Stage 1
print("Stage 1: CharCNN 训练...")
X_c = np.array([seq(t) for t in txts])
y_s = np.array([{'1':0,'2':1,'3':2}[l] for l in s_levels])
X_tr, X_te, y_tr, y_te = train_test_split(X_c, y_s, test_size=0.2, random_state=42, stratify=y_s)

m = CharCNN(len(C2I)+1, 3)
opt = optim.Adam(m.parameters(), lr=0.001)
ld = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), 64, shuffle=True)
t0=time.time(); best=0
for ep in range(20):
    m.train()
    for bx,by in ld:
        opt.zero_grad(); nn.CrossEntropyLoss()(m(bx),by).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        a = accuracy_score(y_te, m(torch.tensor(X_te)).argmax(dim=1).numpy())
        if a>best: best=a
print(f"  CharCNN: {best:.4f}")

# 提取CNN特征
print("\n特征融合...")
m.eval()
feat_dim = 64*3
X_cnn = np.zeros((len(txts), feat_dim))
with torch.no_grad():
    for i in range(0, len(txts), 64):
        batch = torch.tensor(X_c[i:i+64])
        x = m.emb(batch).permute(0,2,1)
        convs = []
        for conv in m.convs:
            h = torch.relu(conv[0](x))
            convs.append(h.max(dim=2)[0])
        X_cnn[i:i+64] = torch.cat(convs, dim=1).numpy()

# TF-IDF
v = TfidfVectorizer(ngram_range=(1,1), max_features=5000, sublinear_tf=True)
X_tf = v.fit_transform(txts).toarray().astype(np.float32)

# 融合 + LR
X_f = np.concatenate([X_cnn, X_tf], axis=1)
X_tr_f, X_te_f, y_tr_f, y_te_f = train_test_split(X_f, y_s, test_size=0.2, random_state=42, stratify=y_s)
clf = LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced', random_state=42)
clf.fit(X_tr_f, y_tr_f)
acc = accuracy_score(y_te_f, clf.predict(X_te_f))

print(f"\n{'='*50}")
print("对比:")
print(f"  LR + TF-IDF:       81.67%")
print(f"  CharCNN:           {best*100:.2f}%")
print(f"  CharCNN+TF-IDF融合: {acc*100:.2f}%")
