#!/usr/bin/env python3
"""
BERT 微调 — 使用 Deep v3 生成的标签
====================================

架构：
  用预训练 BERT 做文本分类
  在 3GB GPU 上优化：小 batch + 梯度累积 + 混合精度

用法：
  python -m nn.bert_finetune --subset 50000   # 先测 5 万条
  python -m nn.bert_finetune                   # 全量 25 万条（需 ~4h）

学习要点：
  - BERT 为什么比 CNN 更强？（双向注意力 + 预训练知识）
  - 小显存怎么训大模型？（梯度累积、混合精度、gradient checkpointing）
  - BERT 的输入格式：input_ids + attention_mask + token_type_ids
"""

import json
import time
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# ── 配置 ──
DATA_PATH = Path('data/人工标注/bert_training_data.json')
MODEL_DIR = Path('nn/models')
BERT_LOCAL = Path('nn/bert-model')  # 本地 BERT 模型文件
MODEL_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# BERT 配置
MAX_LEN = 128          # BERT 最长 512，但 128 已覆盖大部分对话
BATCH_SIZE = 8         # 小 batch 省显存（3GB 限制）
GRADIENT_ACCUM = 4     # 梯度累积步数 → 等效 batch = 8×4 = 32
EPOCHS = 3             # BERT 通常 2-4 epoch 就够了（再久过拟合）
WARMUP_RATIO = 0.1     # 学习率预热比例


# ═══════════════════════════════════════════════════════════
# 1. 数据集
# ═══════════════════════════════════════════════════════════

class TextClassificationDataset(Dataset):
    """
    BERT 数据集

    每个样本：
      input_ids:      词在 BERT 词表中的索引
      attention_mask: 哪些位置是真实 token（1）vs padding（0）
      label:          类别标签
    """
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        text = self.texts[i][:2000]  # 防止极端长文本

        # BERT tokenizer:
        # - 自动分词（中文按字分）
        # - 加 [CLS] 和 [SEP] 标记
        # - 转成 input_ids
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[i], dtype=torch.long),
        }


# ═══════════════════════════════════════════════════════════
# 2. 训练
# ═══════════════════════════════════════════════════════════

class BertTrainer:
    """
    BERT 训练器

    显存优化技巧：
    1. 梯度累积：小 batch 凑大 batch 效果
    2. 混合精度 (fp16)：显存减半，速度翻倍
    3. 学习率预热：BERT 微调需要小心地调整预训练权重
    """
    def __init__(self, model, device, lr=2e-5):
        self.model = model.to(device)
        self.device = device

        # AdamW：带权重衰减的 Adam，BERT 微调标配
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01,  # L2 正则化
        )

        # 混合精度: 用 autocast + GradScaler（fp32 模型，自动选 fp16 算子）
        self.scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    def train_epoch(self, loader, scheduler, gradient_accum=1):
        self.model.train()
        total_loss = 0
        n_batches = len(loader)

        for i, batch in enumerate(loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            # 混合精度前向：自动选择 fp16 算子（兼容新 torch API）
            with torch.amp.autocast('cuda', enabled=self.scaler is not None):
                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss / gradient_accum

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * gradient_accum

            # 梯度累积
            if (i + 1) % gradient_accum == 0 or (i + 1) == n_batches:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()
                scheduler.step()

        return total_loss / n_batches

    def evaluate(self, loader):
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(
                    input_ids, attention_mask=attention_mask, labels=labels
                )
                total_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        return acc, total_loss / len(loader), all_preds, all_labels


# ═══════════════════════════════════════════════════════════
# 3. Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=int, default=None,
                        help='取前N条快速实验')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch', type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    print("=" * 60)
    print("BERT 微调 — 心理咨询对话分类")
    if device.type == 'cuda':
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"设备: {device} (显存: {mem:.1f}GB)")
    else:
        print(f"设备: {device}")
    print("=" * 60)

    # ── 3.1 加载数据 ──
    print(f"\n▶ 加载数据: {DATA_PATH}")
    with open(DATA_PATH, encoding='utf-8') as f:
        data = json.load(f)

    if args.subset:
        data = data[:args.subset]

    texts = [d['text'] for d in data]
    labels_raw = [d['label'] for d in data]

    # 只做 S 级分类（3类），因为这才是主要改进点
    s_labels = [l[0] for l in labels_raw]  # '1','2','3'
    s_map = {'1': 0, '2': 1, '3': 2}
    y = np.array([s_map[l] for l in s_labels])

    n_classes = 3
    stats = Counter(s_labels)
    print(f"  总计: {len(texts)} 条")
    for k in ['1', '2', '3']:
        n = stats.get(k, 0)
        print(f"    {'S1日常困扰' if k=='1' else 'S2心理障碍' if k=='2' else 'S3紧急危机'}: {n}条 ({n/len(texts)*100:.1f}%)")

    # ── 3.2 划分 ──
    train_idx, test_idx = train_test_split(
        np.arange(len(texts)), test_size=5000, random_state=42, stratify=y
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=3000, random_state=42,
        stratify=y[train_idx]
    )
    print(f"\n  划分: 训练{len(train_idx)} / 验证{len(val_idx)} / 测试{len(test_idx)}")

    # ── 3.3 加载 BERT tokenizer（本地） ──
    print(f"\n▶ 加载 BERT tokenizer (本地)...")
    t0 = time.time()
    tokenizer = BertTokenizer.from_pretrained(str(BERT_LOCAL), local_files_only=True)
    print(f"  BERT 词表大小: {len(tokenizer)} ({time.time()-t0:.1f}s)")

    # ── 3.4 创建 DataLoader ──
    train_ds = TextClassificationDataset(
        [texts[i] for i in train_idx], y[train_idx], tokenizer, MAX_LEN
    )
    val_ds = TextClassificationDataset(
        [texts[i] for i in val_idx], y[val_idx], tokenizer, MAX_LEN
    )
    test_ds = TextClassificationDataset(
        [texts[i] for i in test_idx], y[test_idx], tokenizer, MAX_LEN
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch)

    # ── 3.5 加载 BERT 模型（fp32 + autocast 混合精度） ──
    print(f"\n▶ 加载 BERT 模型 (本地)...")
    t0 = time.time()
    torch.cuda.empty_cache()

    model = BertForSequenceClassification.from_pretrained(
        str(BERT_LOCAL),
        num_labels=n_classes,
        hidden_dropout_prob=0.1,
        local_files_only=True,
    )
    model.gradient_checkpointing_enable()
    model = model.to(device)

    # 测试一次前向
    dummy = tokenizer('测试', return_tensors='pt').to(device)
    with torch.no_grad():
        model(**dummy)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  BERT 参数: {n_params:,} ({time.time()-t0:.1f}s)")
    if device.type == 'cuda':
        mem = torch.cuda.memory_allocated() / 1024**2
        print(f"  模型显存: {mem:.0f}MB / 总 {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")

    # ── 3.6 训练 ──
    trainer = BertTrainer(model, device, lr=2e-5)
    total_steps = len(train_loader) * args.epochs // GRADIENT_ACCUM
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        trainer.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"\n▶ 开始训练 ({args.epochs} epochs)...")
    print(f"  batch={args.batch}, 梯度累积={GRADIENT_ACCUM}, "
          f"等效batch={args.batch * GRADIENT_ACCUM}")
    print(f"  总步数: {total_steps}, 预热: {warmup_steps}")

    best_acc = 0
    t_start = time.time()

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = trainer.train_epoch(
            train_loader, scheduler, gradient_accum=GRADIENT_ACCUM
        )
        val_acc, val_loss, _, _ = trainer.evaluate(val_loader)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_DIR / 'bert_best.pt')
            print(f"  ✓ 保存最佳模型 (acc={val_acc:.4f})")

        print(f"  Epoch {epoch+1}/{args.epochs} | "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} | "
              f"{time.time()-t0:.0f}s")

    # ── 3.7 测试集评估 ──
    model.load_state_dict(torch.load(MODEL_DIR / 'bert_best.pt'))
    test_acc, test_loss, preds, true = trainer.evaluate(test_loader)
    target_names = ['S1日常困扰', 'S2心理障碍', 'S3紧急危机']
    print(f"\n{'='*60}")
    print("📊 BERT 测试结果")
    print(f"{'='*60}")
    print(f"  测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  总训练时间: {time.time()-t_start:.0f}s")
    print(f"\n  分类报告:")
    print(classification_report(true, preds, target_names=target_names))

    # ── 3.8 对比 CharCNN ──
    print(f"\n{'='*60}")
    print("模型对比 (S1/S2/S3)")
    print(f"{'='*60}")
    print(f"  {'模型':<25} {'准确率':<10}")
    print(f"  {'-'*35}")
    print(f"  {'CharCNN (Original 250K)':<25} {'~0.6833':<10}")
    print(f"  {'CharCNN (Deep v3 50K)':<25} {'~0.7567':<10}")
    print(f"  {'BERT (refined labels)':<25} {test_acc:<10.4f}")

    # BERT 是否赢了
    bert_won = test_acc > 0.7567
    print(f"\n  {'✅ BERT 超越 CharCNN!' if bert_won else '⚠️ BERT 未超越 CharCNN，可能需更多数据/epochs'}")
    print(f"\n  模型已保存: {MODEL_DIR / 'bert_best.pt'}")


if __name__ == '__main__':
    main()
