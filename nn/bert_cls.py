#!/usr/bin/env python3
"""BERT 微调实验 (使用 HuggingFace Transformers)"""
import json, re, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

DATA = Path('data')
OUT = Path('data/人工标注')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 使用小批量测试
MAX_SAMPLES = 500  # BERT 太慢，先用 500 条试试
BATCH_SIZE = 8     # CPU 上 batch 要小
EPOCHS = 3

with open(OUT / 'No-01_待标注_3000_已标注.json', encoding='utf-8') as f:
    seed = json.load(f)

classes = sorted(set(item['labels']['label'] for item in seed))
class_to_idx = {c: i for i, c in enumerate(classes)}

# 取子集
data = seed[:MAX_SAMPLES]
texts = []
for item in data:
    p = [item.get('question_title',''), item.get('question_content','')]
    for a in item.get('answers',[]):
        for d in a.get('dialogs',[]): p.append(d.get('content',''))
    texts.append(' '.join(p)[:512])  # BERT 最大 512 token

labels = [class_to_idx[item['labels']['label']] for item in data]
n_classes = len(classes)

print(f"BERT 实验: {len(texts)}条, {n_classes}类, 设备={device}")

# 加载 tokenizer 和模型
print("加载 BERT...")
t0 = time.time()
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese', num_labels=n_classes,
    hidden_dropout_prob=0.1,
)
model.to(device)
print(f"  BERT 加载完成 ({time.time()-t0:.0f}s)")

# 编码
print("编码文本...")
t0 = time.time()
encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
print(f"  编码完成 ({time.time()-t0:.0f}s)")

# 划分
idx = np.arange(len(texts))
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

class TextDataset(Dataset):
    def __init__(self, encodings, labels, indices):
        self.input_ids = encodings['input_ids'][indices]
        self.attention_mask = encodings['attention_mask'][indices]
        self.labels = torch.tensor([labels[i] for i in indices])
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return self.input_ids[i], self.attention_mask[i], self.labels[i]

train_ds = TextDataset(encodings, labels, train_idx)
test_ds = TextDataset(encodings, labels, test_idx)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# 训练
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)

print(f"训练 BERT ({EPOCHS} epochs)...")
t0 = time.time()
best_acc = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, batch_labels in train_loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=batch_labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    # 评估
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attention_mask, batch_labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1).cpu()
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.size(0)
    acc = correct / total
    if acc > best_acc: best_acc = acc
    print(f"  Epoch {epoch+1}/{EPOCHS} loss={total_loss/len(train_loader):.4f} val_acc={acc:.4f}")

print(f"\nBERT 最佳准确率: {best_acc:.4f} (耗时: {time.time()-t0:.0f}s)")

# 对比
print(f"\n{'='*50}")
print("实验对比 (500条子集)")
print(f"{'='*50}")
print(f"  CharCNN (全量3000条):  ~0.47")
print(f"  BERT (500条子集):     {best_acc:.4f}")
print(f"\n备注: BERT 在 CPU 上极慢，500条耗时{time.time()-t0:.0f}s")
print(f"如需全量训练，建议使用 GPU 或减少训练数据")
