#!/usr/bin/env python3
"""
Ensemble Voting — 多模型集成投票
================================

集成多个模型，用多数投票提升分类准确率。

策略:
  1. 多数投票（默认）- 所有模型投票，取多数
  2. S3 保守策略 — 仅当所有模型一致同意才判 S3，否则在 S1/S2 中取多数
     （提升 S3 precision，但 recall 可能略降）

用法:
  # 交互模式
  python -m nn.ensemble

  # 单条预测
  python -m nn.ensemble --text "最近压力很大，睡不着觉"

  # 测试集评估
  python -m nn.ensemble --eval

  # 训练缺失的模型
  python -m nn.ensemble --train-char-cnn
"""

import json
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report

from nn.config import (
    DEVICE, C2I, VOCAB_SIZE, MAX_LEN, MODEL_DIR, BERT_DIR,
    to_char_seq, load_labeled_data,
)

# ═══════════════════════════════════════════════════════════════
# 1. 模型定义（复用已有架构，保持一致）
# ═══════════════════════════════════════════════════════════════

S_LABELS = ['S1日常困扰', 'S2心理障碍', 'S3紧急危机']
S_MAP = {'1': 0, '2': 1, '3': 2}


class CharCNN(nn.Module):
    """原版 CharCNN（与 char_cnn.py 一致）"""
    def __init__(self, vocab_size, n_classes, embed_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, 64, k, padding=k // 2),
                nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveMaxPool1d(1),
            ) for k in [3, 5, 7]
        ])
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 * 3, n_classes),
        )

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        x = torch.cat([conv(x).squeeze(-1) for conv in self.convs], dim=1)
        return self.fc(x)


class ResidualConvBlock(nn.Module):
    """残差卷积块（与 char_cnn_deep.py 一致）"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, stride=1):
        super().__init__()
        padding = kernel_size // 2 * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class CharCNN_Deep_v3(nn.Module):
    """CharCNN v3 — 深度残差版（与 char_cnn_deep.py 一致）"""
    def __init__(self, vocab_size, n_classes, embed_dim=128):
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
        feat_dim = 128 + 256 + 256 + 128  # = 768
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 256),
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


# ═══════════════════════════════════════════════════════════════
# 2. 集成投票器
# ═══════════════════════════════════════════════════════════════

ENSEMBLE_DIR = MODEL_DIR / 'ensemble'
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

# 每个模型对应的保存路径（如果为 None 则需先训练）
MODEL_PATHS = {
    'CharCNN': MODEL_DIR / 'char_cnn_best.pt',
    'Deep v3': MODEL_DIR / 'char_cnn_deep_best.pt',
    'BERT':    MODEL_DIR / 'bert_best.pt',
}


class EnsembleVoter:
    """
    多模型集成投票器

    加载所有可用模型，统一接口做预测和投票。
    """

    def __init__(self, device=DEVICE, s3_conservative=True):
        self.device = device
        self.s3_conservative = s3_conservative  # S3 保守策略
        self.models = {}       # name → model
        self.tokenizer = None  # BERT tokenizer
        self.weights = {}      # name → weight (for weighted voting)

    # ── 加载 ────────────────────────────────────────────────

    def load_all(self):
        """加载所有已保存的模型，跳过缺失的"""
        print("▶ 加载集成模型...")
        self._load_char_cnn()
        self._load_deep_v3()
        self._load_bert()
        n = len(self.models)
        print(f"  ✓ 已加载 {n} 个模型: {list(self.models.keys())}")
        if n < 2:
            print("  ⚠ 至少需要 2 个模型才能投票，请先训练缺失的模型")
        return self.models

    def _load_char_cnn(self):
        path = MODEL_PATHS['CharCNN']
        if not path.exists():
            print("  ⚠ CharCNN 权重不存在，跳过")
            return
        model = CharCNN(VOCAB_SIZE, 3).to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        model.eval()
        self.models['CharCNN'] = model
        print(f"  ✓ CharCNN 加载完成")

    def _load_deep_v3(self):
        path = MODEL_PATHS['Deep v3']
        if not path.exists():
            print("  ⚠ Deep v3 权重不存在，跳过")
            return
        model = CharCNN_Deep_v3(VOCAB_SIZE, 3).to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        model.eval()
        self.models['Deep v3'] = model
        print(f"  ✓ Deep v3 加载完成")

    def _load_bert(self):
        path = MODEL_PATHS['BERT']
        if not path.exists():
            print("  ⚠ BERT 权重不存在，跳过")
            return
        try:
            from transformers import AutoTokenizer, BertForSequenceClassification
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(BERT_DIR), local_files_only=True
            )
            model = BertForSequenceClassification.from_pretrained(
                str(BERT_DIR), num_labels=3,
                local_files_only=True, ignore_mismatched_sizes=True,
            )
            # 加载微调权重
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            self.models['BERT'] = model
            print(f"  ✓ BERT 加载完成")
        except Exception as e:
            print(f"  ⚠ BERT 加载失败: {e}")

    # ── 预测 ────────────────────────────────────────────────

    def predict_single(self, text):
        """
        对单段文本预测，返回各模型结果和集成结果

        Returns:
            dict: {
                'text': str,
                'models': { name: {'label': int, 'prob': float, 'class': str} },
                'ensemble': {'label': int, 'class': str, 'confidence': float},
                'strategy': str,
            }
        """
        result = {'text': text, 'models': {}}

        for name, model in self.models.items():
            if name == 'BERT':
                label, prob = self._predict_bert(text)
            else:
                label, prob = self._predict_char_cnn(text, model)
            result['models'][name] = {
                'label': label,
                'prob': prob,
                'class': S_LABELS[label],
            }

        # 投票
        labels = [m['label'] for m in result['models'].values()]
        ensemble_label, strategy = self._vote(labels)

        result['ensemble'] = {
            'label': ensemble_label,
            'class': S_LABELS[ensemble_label],
        }
        result['strategy'] = strategy
        return result

    def predict_batch(self, texts, batch_size=64):
        """对批量文本预测，返回列表"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                results.append(self.predict_single(text))
        return results

    def _predict_bert(self, text):
        """BERT 单条预测"""
        encoding = self.tokenizer(
            text[:2000],
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt',
        ).to(self.device)
        with torch.no_grad():
            outputs = self.models['BERT'](
                encoding['input_ids'],
                attention_mask=encoding['attention_mask'],
            )
            probs = torch.softmax(outputs.logits, dim=1)
            label = probs.argmax(dim=1).item()
            prob = probs[0, label].item()
        return label, prob

    def _predict_char_cnn(self, text, model):
        """CharCNN 单条预测"""
        X = to_char_seq([text])
        X_t = torch.tensor(X).to(self.device)
        with torch.no_grad():
            logits = model(X_t)
            probs = torch.softmax(logits, dim=1)
            label = probs.argmax(dim=1).item()
            prob = probs[0, label].item()
        return label, prob

    def _vote(self, labels):
        """
        投票策略

        两种模式:
          1. 普通多数投票: 取最多模型同意的类别
          2. S3 保守策略: 仅当所有模型一致同意才判 S3，否则在 S1/S2 中取多数
        """
        if not labels:
            return 1, 'fallback (no models)'

        s3_count = sum(1 for l in labels if l == 2)  # 2 = S3

        if self.s3_conservative:
            # S3 保守: 所有模型一致同意才是 S3
            if s3_count == len(labels):
                return 2, 'S3-conservative (unanimous)'
            # 否则在 S1/S2 中取多数
            non_s3 = [l for l in labels if l != 2]
            majority = max(set(non_s3), key=non_s3.count)
            return majority, 'S3-conservative (majority of S1/S2)'
        else:
            # 普通多数投票
            majority = max(set(labels), key=labels.count)
            return majority, 'majority-vote'

    # ── 可用性 ──────────────────────────────────────────────

    @property
    def is_ready(self):
        return len(self.models) >= 2

    def list_available(self):
        """列出每个模型的状态"""
        total = len(MODEL_PATHS)
        available = sum(p.exists() for p in MODEL_PATHS.values())
        print(f"  模型状态 ({available}/{total} 可用):")
        for name, path in MODEL_PATHS.items():
            exists = path.exists()
            status = '✓ 已保存' if exists else '✗ 未训练'
            print(f"    {name:<12} {status}  ({path.name})")


# ═══════════════════════════════════════════════════════════════
# 3. 训练工具
# ═══════════════════════════════════════════════════════════════

def train_char_cnn(subset=50000, epochs=25):
    """训练 CharCNN Original 并保存"""
    print("=" * 60)
    print("训练 CharCNN Original")
    print("=" * 60)

    texts, s_labels, _ = load_labeled_data(subset=subset)
    y = np.array([S_MAP[l] for l in s_labels])
    print(f"  数据: {len(texts)} 条")

    X = to_char_seq(texts)

    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=5000, random_state=42, stratify=y
    )
    print(f"  训练: {len(X_tr)} / 验证: {len(X_val)}")

    model = CharCNN(VOCAB_SIZE, 3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    X_tr_t = torch.tensor(X_tr)
    y_tr_t = torch.tensor(y_tr)
    X_val_t = torch.tensor(X_val).to(DEVICE)
    y_val_t = torch.tensor(y_val).to(DEVICE)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), 128, shuffle=True)

    best_acc = 0
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_val_t).argmax(dim=1)
            acc = (preds == y_val_t).float().mean().item()
        if acc > best_acc:
            best_acc = acc
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  Epoch {ep+1:2d}/{epochs}  val_acc={acc:.4f}")

    print(f"  最佳验证准确率: {best_acc:.4f} ({time.time()-t0:.1f}s)")

    # 用最佳参数重新训练全量
    print("\n▶ 全量数据重新训练...")
    model = CharCNN(VOCAB_SIZE, 3).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    X_all_t = torch.tensor(X).to(DEVICE)
    y_all_t = torch.tensor(y).to(DEVICE)
    all_loader = DataLoader(TensorDataset(X_all_t, y_all_t), 128, shuffle=True)

    for ep in range(epochs):
        model.train()
        for bx, by in all_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()

    # 保存
    path = MODEL_PATHS['CharCNN']
    torch.save(model.state_dict(), path)
    print(f"  ✓ 模型已保存: {path}")

    return best_acc


def train_deep_v3(subset=50000, epochs=15):
    """训练 CharCNN Deep v3 并保存（快速版本，非完整对比）"""
    print("=" * 60)
    print("训练 CharCNN Deep v3")
    print("=" * 60)

    texts, s_labels, _ = load_labeled_data(subset=subset)
    y = np.array([S_MAP[l] for l in s_labels])
    print(f"  数据: {len(texts)} 条")

    X = to_char_seq(texts)

    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=5000, random_state=42, stratify=y
    )
    print(f"  训练: {len(X_tr)} / 验证: {len(X_val)}")

    model = CharCNN_Deep_v3(VOCAB_SIZE, 3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    X_tr_t = torch.tensor(X_tr)
    y_tr_t = torch.tensor(y_tr)
    X_val_t = torch.tensor(X_val).to(DEVICE)
    y_val_t = torch.tensor(y_val).to(DEVICE)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), 128, shuffle=True)

    best_acc = 0
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_val_t).argmax(dim=1)
            acc = (preds == y_val_t).float().mean().item()
        if acc > best_acc:
            best_acc = acc
        if (ep + 1) % 5 == 0:
            print(f"  Epoch {ep+1:2d}/{epochs}  val_acc={acc:.4f}")

    print(f"  最佳验证准确率: {best_acc:.4f} ({time.time()-t0:.1f}s)")

    # 全量训练
    print("\n▶ 全量数据重新训练...")
    model = CharCNN_Deep_v3(VOCAB_SIZE, 3).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    X_all_t = torch.tensor(X).to(DEVICE)
    y_all_t = torch.tensor(y).to(DEVICE)
    all_loader = DataLoader(TensorDataset(X_all_t, y_all_t), 128, shuffle=True)

    for ep in range(epochs):
        model.train()
        for bx, by in all_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            optimizer.step()

    # 保存
    path = MODEL_PATHS['Deep v3']
    torch.save(model.state_dict(), path)
    print(f"  ✓ 模型已保存: {path}")

    return best_acc


def train_bert(subset=None, epochs=3):
    """训练 BERT 并保存（简化版，参照 bert_finetune.py）"""
    print("=" * 60)
    print("训练 BERT 微调")
    print("=" * 60)

    try:
        from transformers import (
            BertTokenizer, BertForSequenceClassification,
            get_linear_schedule_with_warmup,
        )
        from torch.utils.data import Dataset, DataLoader
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("  ✗ 请先安装 transformers: pip install transformers")
        return None

    # 加载 BERT 数据
    data_path = Path('data/人工标注/bert_training_data.json')
    if not data_path.exists():
        print(f"  ✗ BERT 数据不存在: {data_path}")
        return None

    with open(data_path, encoding='utf-8') as f:
        data = json.load(f)

    if subset:
        data = data[:subset]

    texts = [d['text'] for d in data]
    labels_raw = [d['label'] for d in data]
    y = np.array([S_MAP[l[0]] for l in labels_raw])

    print(f"  数据: {len(texts)} 条")

    train_idx, val_idx = train_test_split(
        np.arange(len(texts)), test_size=5000, random_state=42, stratify=y
    )
    train_idx, _ = train_test_split(
        train_idx, test_size=3000, random_state=42, stratify=y[train_idx]
    )

    tokenizer = BertTokenizer.from_pretrained(str(BERT_DIR), local_files_only=True)

    class BERTDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, i):
            enc = self.tokenizer(
                self.texts[i][:2000], truncation=True,
                padding='max_length', max_length=self.max_len, return_tensors='pt',
            )
            return {
                'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'label': torch.tensor(self.labels[i], dtype=torch.long),
            }

    train_ds = BERTDataset([texts[i] for i in train_idx], y[train_idx], tokenizer)
    val_ds = BERTDataset([texts[i] for i in val_idx], y[val_idx], tokenizer)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8)

    model = BertForSequenceClassification.from_pretrained(
        str(BERT_DIR), num_labels=3, local_files_only=True,
    ).to(DEVICE)
    model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    best_acc = 0
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            with torch.amp.autocast('cuda', enabled=scaler is not None):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

            if scaler:
                scaler.scale(outputs.loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs.loss.backward()
                optimizer.step()

            optimizer.zero_grad()
            scheduler.step()

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        if acc > best_acc:
            best_acc = acc
        print(f"  Epoch {ep+1}/{epochs}  val_acc={acc:.4f}")

    path = MODEL_PATHS['BERT']
    torch.save(model.state_dict(), path)
    print(f"  ✓ BERT 模型已保存: {path} ({time.time()-t0:.1f}s)")

    return best_acc


# ═══════════════════════════════════════════════════════════════
# 4. 评估
# ═══════════════════════════════════════════════════════════════

def evaluate_ensemble(voter, subset=None):
    """在测试集上评估集成效果"""
    print("\n▶ 加载测试数据...")
    texts, s_labels, _ = load_labeled_data(subset=subset)
    y_true = np.array([S_MAP[l] for l in s_labels])

    print(f"  共 {len(texts)} 条")

    # 逐条预测
    print("▶ 集成预测中...")
    t0 = time.time()
    all_ensemble = []
    all_models = {name: [] for name in voter.models}

    for i, text in enumerate(texts):
        result = voter.predict_single(text)
        all_ensemble.append(result['ensemble']['label'])
        for name, m in result['models'].items():
            all_models[name].append(m['label'])

        if (i + 1) % 5000 == 0:
            print(f"  已处理 {i+1}/{len(texts)} ({time.time()-t0:.1f}s)")

    print(f"  完成 ({time.time()-t0:.1f}s)")

    # ── 各模型独立准确率 ──
    print(f"\n{'='*60}")
    print("📊 各模型独立准确率 vs 集成结果")
    print(f"{'='*60}")
    print(f"  {'模型':<16} {'准确率':<10} {'提升'}")
    print(f"  {'-'*45}")

    for name in voter.models:
        acc = accuracy_score(y_true, all_models[name])
        print(f"  {name:<16} {acc:.4f}")

    # 集成准确率
    ensemble_acc = accuracy_score(y_true, all_ensemble)
    print(f"  {'─'*45}")
    print(f"  {'集成投票':<16} {ensemble_acc:.4f}")

    # 完整报告
    print(f"\n  集成模型分类报告:")
    print(classification_report(
        y_true, all_ensemble, target_names=S_LABELS
    ))

    # 与最佳单模型的对比
    best_single_name = max(voter.models.keys(),
                           key=lambda n: accuracy_score(y_true, all_models[n]))
    best_single_acc = accuracy_score(y_true, all_models[best_single_name])
    gain = ensemble_acc - best_single_acc
    print(f"\n  最佳单模型: {best_single_name} ({best_single_acc:.4f})")
    print(f"  集成提升:   {gain:+.4f} ({gain*100:+.2f}%)")

    return ensemble_acc, best_single_acc


# ═══════════════════════════════════════════════════════════════
# 5. Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='集成投票 — 多模型分类',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m nn.ensemble                    # 交互模式
  python -m nn.ensemble --text "睡不着"    # 单条预测
  python -m nn.ensemble --eval             # 测试集评估
  python -m nn.ensemble --train-all        # 训练所有缺失模型
  python -m nn.ensemble --status           # 查看模型状态
        """,
    )
    parser.add_argument('--text', '-t', type=str, default=None,
                        help='输入文本，直接预测')
    parser.add_argument('--eval', action='store_true',
                        help='在测试集上评估')
    parser.add_argument('--subset', type=int, default=None,
                        help='取前N条数据快速实验')
    parser.add_argument('--train-all', action='store_true',
                        help='训练所有缺失的模型')
    parser.add_argument('--train-char-cnn', action='store_true',
                        help='训练 CharCNN Original')
    parser.add_argument('--no-s3-conservative', action='store_true',
                        help='关闭 S3 保守策略，使用普通多数投票')
    parser.add_argument('--status', action='store_true',
                        help='查看各模型保存状态')
    args = parser.parse_args()

    print("=" * 60)
    print("模型集成投票 — Ensemble Voting")
    print(f"设备: {DEVICE}")
    print("=" * 60)

    # ── 状态查看 ──
    if args.status:
        voter = EnsembleVoter()
        voter.list_available()
        return

    # ── 训练模式 ──
    if args.train_all:
        print("\n▶ 训练所有缺失模型...")
        for name, path in MODEL_PATHS.items():
            if not path.exists():
                print(f"\n  --- 训练 {name} ---")
                if name == 'CharCNN':
                    train_char_cnn(subset=args.subset)
                elif name == 'Deep v3':
                    train_deep_v3(subset=args.subset)
                elif name == 'BERT':
                    train_bert(subset=args.subset)
            else:
                print(f"  ✓ {name} 已存在，跳过")
        print("\n  ✅ 所有模型就绪")
        return

    if args.train_char_cnn:
        acc = train_char_cnn(subset=args.subset)
        print(f"\n  ✓ CharCNN 训练完成, val_acc={acc:.4f}")
        return

    # ── 加载模型 ──
    voter = EnsembleVoter(device=DEVICE, s3_conservative=not args.no_s3_conservative)
    voter.load_all()

    if not voter.is_ready:
        print("\n⚠ 可用模型不足，请先训练:")
        print("  python -m nn.ensemble --train-all")
        return

    # ── 单条预测 ──
    if args.text:
        print(f"\n▶ 输入: {args.text[:100]}")
        result = voter.predict_single(args.text)

        print(f"\n  各模型结果:")
        for name, m in result['models'].items():
            print(f"    {name:<12} → {m['class']} (p={m['prob']:.3f})")

        ens = result['ensemble']
        print(f"\n  🏆 集成结果: {ens['class']}  (strategy: {result['strategy']})")
        return

    # ── 测试集评估 ──
    if args.eval:
        evaluate_ensemble(voter, subset=args.subset)
        return

    # ── 交互模式 ──
    print("\n📝 交互模式 (输入 quit 退出)")
    print(f"   策略: {'S3 保守策略' if args.no_s3_conservative else '普通多数投票'}")
    print(f"   模型: {list(voter.models.keys())}")
    print()

    while True:
        try:
            text = input("输入文本 > ").strip()
            if not text:
                continue
            if text.lower() in ('quit', 'exit', 'q'):
                break

            result = voter.predict_single(text)

            models_line = ' | '.join(
                f"{n}: {m['class']}({m['prob']:.2f})"
                for n, m in result['models'].items()
            )
            print(f"  [{models_line}]")
            print(f"  🏆 集成 → {result['ensemble']['class']}  [{result['strategy']}]")
            print()

        except KeyboardInterrupt:
            break
        except EOFError:
            break

    print("bye.")


if __name__ == '__main__':
    main()
