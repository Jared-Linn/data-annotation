#!/usr/bin/env python3
"""Word2Vec + TextCNN/MLP 分类器"""
import json, re, time, sys
from pathlib import Path
from collections import Counter
import numpy as np
import jieba
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA = Path('data')
OUT = Path('data/人工标注')

with open(DATA / 'stopwords.txt', encoding='utf-8') as f:
    STOP_WORDS = set(line.strip() for line in f if line.strip())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def cut_words(t):
    return [w for w in jieba.cut(t) if w.strip() and w not in STOP_WORDS and len(w) > 1]

def cln(t):
    return re.sub(r'\s+', '', re.sub(r'[^一-鿿\w]', '', t))

def bld_words(item):
    """返回词列表"""
    p = [item.get('question_title',''), item.get('question_content','')]
    for a in item.get('answers',[]):
        for d in a.get('dialogs',[]): p.append(d.get('content',''))
    return cut_words(cln(' '.join(p)))

# ===== 1. 训练 Word2Vec =====
print("=" * 60)
print("Word2Vec 训练")
print("=" * 60)

# 用所有可用数据训练词向量
all_files = sorted(Path('data').glob('No*.json'))
print(f"加载 {len(all_files)} 个文件...")
all_sentences = []
for fp in all_files:
    try:
        with open(fp, encoding='utf-8') as f:
            for item in json.load(f):
                words = bld_words(item)
                if words:
                    all_sentences.append(words)
    except:
        pass

print(f"共 {len(all_sentences)} 条文本, {sum(len(s) for s in all_sentences)} 词")

from gensim.models import Word2Vec
t0 = time.time()
w2v = Word2Vec(all_sentences, vector_size=200, window=5, min_count=5,
               workers=4, sg=1, epochs=10, seed=42)
print(f"词表: {len(w2v.wv)} 词 | 耗时: {time.time()-t0:.1f}s")

# ===== 2. 准备分类数据 =====
print(f"\n{'='*60}")
print("分类数据准备")
print(f"{'='*60}")

with open(OUT / 'No-01_待标注_3000_已标注.json', encoding='utf-8') as f:
    seed = json.load(f)

classes = sorted(set(item['labels']['label'] for item in seed))
class_to_idx = {c: i for i, c in enumerate(classes)}
print(f"类别: {len(classes)}")

# 文本转词向量序列（用于TextCNN）和词向量平均（用于MLP）
def text_to_vector(words):
    """词向量平均 (用于MLP)"""
    vectors = [w2v.wv[w] for w in words if w in w2v.wv]
    if not vectors:
        return np.zeros(w2v.vector_size)
    return np.mean(vectors, axis=0)

def text_to_indices(words, max_len=200):
    """词转索引序列 (用于TextCNN)"""
    indices = [w2v.wv.key_to_index[w] for w in words[:max_len] if w in w2v.wv]
    if not indices:
        indices = [0]
    # padding
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    return indices[:max_len]

X_avg, X_seq, y = [], [], []
for item in seed:
    words = bld_words(item)
    X_avg.append(text_to_vector(words))
    X_seq.append(text_to_indices(words))
    y.append(class_to_idx[item['labels']['label']])

X_avg = np.array(X_avg, dtype=np.float32)
y = np.array(y)

# 划分
idx = np.arange(len(y))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
X_avg_tr, X_avg_te = X_avg[train_idx], X_avg[test_idx]
X_seq_tr = [X_seq[i] for i in train_idx]
X_seq_te = [X_seq[i] for i in test_idx]
y_tr, y_te = y[train_idx], y[test_idx]
n_classes = len(classes)

print(f"训练: {len(y_tr)}条, 测试: {len(y_te)}条")

# ===== 3. MLP (词向量平均) =====
print(f"\n{'='*60}")
print("MLP + Word2Vec 词向量平均")
print(f"{'='*60}")

class AvgMLP(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )
    def forward(self, x):
        return self.net(x)

def train_model(model, X_tr, y_tr, X_te, y_te, epochs=30, lr=0.001):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_tr) if isinstance(X_tr, np.ndarray) else X_tr,
        torch.from_numpy(y_tr) if isinstance(y_tr, np.ndarray) else y_tr,
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            X_te_t = torch.from_numpy(X_te).to(device) if isinstance(X_te, np.ndarray) else X_te.to(device)
            preds = model(X_te_t).argmax(dim=1).cpu().numpy()
            acc = accuracy_score(y_te, preds)
            if acc > best_acc:
                best_acc = acc
        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1:2d}/{epochs} val_acc={acc:.4f}")
    return best_acc

t0 = time.time()
mlp_acc = train_model(AvgMLP(w2v.vector_size, n_classes), X_avg_tr, y_tr, X_avg_te, y_te)
print(f"MLP+W2V 准确率: {mlp_acc:.4f} | 耗时: {time.time()-t0:.1f}s")

# ===== 4. TextCNN =====
print(f"\n{'='*60}")
print("TextCNN + Word2Vec (预训练词向量)")
print(f"{'='*60}")

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, max_len=200):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, 128, ks, padding=ks//2) for ks in [3,4,5]
        ])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128 * 3, n_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (B, E, L)
        convs = [torch.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        out = torch.cat(convs, dim=1)
        return self.fc(self.dropout(out))

# 构建词嵌入矩阵（预训练向量）
vocab_size = len(w2v.wv) + 1
embed_dim = w2v.vector_size

# 数据集
class TextDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = [torch.tensor(s, dtype=torch.long) for s in seqs]
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return self.seqs[i], self.labels[i]

tr_dataset = TextDataset(X_seq_tr, y_tr)
te_dataset = TextDataset(X_seq_te, y_te)
tr_loader = DataLoader(tr_dataset, batch_size=32, shuffle=True)
te_loader = DataLoader(te_dataset, batch_size=32)

# 初始化模型并加载预训练词向量
cnn = TextCNN(vocab_size, embed_dim, n_classes).to(device)
with torch.no_grad():
    for word, idx in w2v.wv.key_to_index.items():
        if idx + 1 < vocab_size:
            cnn.embedding.weight[idx + 1] = torch.from_numpy(w2v.wv[word])

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001, weight_decay=1e-4)

best_acc = 0
for epoch in range(20):
    cnn.train()
    for bx, by in tr_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(cnn(bx), by)
        loss.backward()
        optimizer.step()

    cnn.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for bx, by in te_loader:
            bx, by = bx.to(device), by.to(device)
            preds = cnn(bx).argmax(dim=1)
            correct += (preds == by).sum().item()
            total += by.size(0)
    acc = correct / total
    if acc > best_acc:
        best_acc = acc
    if (epoch+1) % 5 == 0:
        print(f"  Epoch {epoch+1:2d}/20 val_acc={acc:.4f}")

print(f"TextCNN 准确率: {best_acc:.4f}")

# ===== 5. 对比总结 =====
print(f"\n{'='*60}")
print("模型对比总结 (31类)")
print(f"{'='*60}")
print(f"{'模型':<30} {'准确率':>8}")
print("-" * 40)
print(f"{'LR + TF-IDF (基线)':<30} {'~0.45':>8}")
print(f"{'MLP + TF-IDF':<30} {'~0.50':>8}")
print(f"{'MLP + Word2Vec 词平均':<30} {mlp_acc:>8.4f}")
print(f"{'TextCNN + Word2Vec':<30} {best_acc:>8.4f}")
