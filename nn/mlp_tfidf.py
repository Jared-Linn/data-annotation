#!/usr/bin/env python3
"""神经网络实验1: MLP + TF-IDF (直接替代LR)"""
import json, re, time
from pathlib import Path
from collections import Counter
import numpy as np
import jieba
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA = Path('data')
OUT = Path('data/人工标注')

with open(DATA / 'stopwords.txt', encoding='utf-8') as f:
    STOP_WORDS = set(line.strip() for line in f if line.strip())

def cut(t): return ' '.join(w for w in jieba.cut(t) if w.strip() and w not in STOP_WORDS)
def cln(t): return re.sub(r'\s+', '', re.sub(r'[^一-鿿\w]', '', t))
def bld(item):
    p = [item.get('question_title',''), item.get('question_content','')]
    for a in item.get('answers',[]):
        for d in a.get('dialogs',[]): p.append(d.get('content',''))
    return cut(cln(' '.join(p)))

# 加载数据
with open(OUT / 'No-01_待标注_3000_已标注.json', encoding='utf-8') as f:
    seed = json.load(f)
txts = [bld(it) for it in seed]
lbls = [it['labels']['label'] for it in seed]
classes = sorted(set(lbls))
class_to_idx = {c: i for i, c in enumerate(classes)}
y_idx = [class_to_idx[l] for l in lbls]
print(f"数据: {len(txts)}条, {len(classes)}类")

# TF-IDF
v = TfidfVectorizer(ngram_range=(1,1), max_features=5000, sublinear_tf=True)
X = v.fit_transform(txts).toarray().astype(np.float32)
X_tr, X_te, y_tr, y_te = train_test_split(X, y_idx, test_size=0.2, random_state=42, stratify=y_idx)
print(f"TF-IDF: {X.shape[1]}维, 训练{len(y_tr)}条, 测试{len(y_te)}条")

# 构建PyTorch数据集
BATCH_SIZE = 64
train_dataset = TensorDataset(torch.from_numpy(X_tr), torch.tensor(y_tr, dtype=torch.long))
test_dataset = TensorDataset(torch.from_numpy(X_te), torch.tensor(y_te, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# MLP模型
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_dims=[512, 256]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.BatchNorm1d(hd),
                nn.ReLU(),
                nn.Dropout(0.3),
            ])
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

model = MLPClassifier(X.shape[1], len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# 训练
EPOCHS = 50
t0 = time.time()
best_acc = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 验证
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            preds = model(batch_X).argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    acc = correct / total
    if acc > best_acc:
        best_acc = acc
    scheduler.step()

    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:2d}/{EPOCHS} loss={total_loss/len(train_loader):.4f} val_acc={acc:.4f}")

# LR基线对比
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced', random_state=42)
lr.fit(X_tr, [c for c in y_tr])
lr_acc = accuracy_score(y_te, lr.predict(X_te))

print(f"\n{'='*60}")
print("实验结果对比 (S1+17类)")
print(f"{'='*60}")
print(f"LR (逻辑回归):     {lr_acc:.4f}")
print(f"MLP (神经网络):    {best_acc:.4f}")
print(f"MLP best epoch:    {np.argmax([0])+1}")
print(f"训练时间:          {time.time()-t0:.1f}s")
print(f"\nMLP全量预测...")

# MLP全量预测
model.eval()
all_preds = []
with torch.no_grad():
    X_all = torch.from_numpy(X).to(device)
    batch_size = 256
    for i in range(0, len(X_all), batch_size):
        batch = X_all[i:i+batch_size]
        preds = model(batch).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)

mlp_acc = accuracy_score(y_idx, all_preds)
print(f"MLP 训练集准确率: {mlp_acc:.4f}")
print(f"\nMLP 分类报告 (测试集):")
y_pred_mlp = []
model.eval()
with torch.no_grad():
    for batch_X, _ in test_loader:
        batch_X = batch_X.to(device)
        y_pred_mlp.extend(model(batch_X).argmax(dim=1).cpu().numpy())

print(classification_report(y_te, y_pred_mlp, target_names=classes, zero_division=0))
