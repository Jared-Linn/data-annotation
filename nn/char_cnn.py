#!/usr/bin/env python3
"""
字符级CNN + 对比基线 — 心理咨询对话分类
========================================

架构：两阶段分类
  Stage 1: S1/S2/S3 粗分类（3类）
  Stage 2: 各S下的子类细分类（31类）

对比模型：
  A. LR + TF-IDF（传统基线）
  B. MLP + TF-IDF（浅层神经网络）
  C. CharCNN（字符级卷积，无需分词）

用法：
  python -m nn.char_cnn                     # 全量(慢)
  python -m nn.char_cnn --subset 10000       # 取1万条快速实验

学习要点：
  - 两阶段分类为什么比直接31类效果好？
  - 字符级CNN vs 词级模型的区别
  - 过拟合判断：训练acc vs 验证acc
"""

import json
import re
import time
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ============================================================
# 0. 配置
# ============================================================

DATA_PATH = Path('data/人工标注/pseudo_labeled_all.json')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 字符表
# 原理：CharCNN 把文本看成字符序列，不需要分词
# 包含常用汉字 + 心理咨询常见词中的字
_CHARS = sorted(set(
    'abcdefghijklmnopqrstuvwxyz0123456789'
    '的一是不了人在我有他这那中心大小上到说会走时自家为以看好起学过如生动作发后出没开面'
    '心理情绪压力焦虑抑郁恐惧强迫悲伤愤怒痛苦绝望伤害死亡自杀攻击暴力报复学业考试工作'
    '家庭关系婚姻恋爱男女朋友父母孩子教育成绩毕业考研就业睡梦哭吃喝玩钱想知道看见听见'
))
C2I = {c: i + 1 for i, c in enumerate(_CHARS)}  # 0 留给 padding
VOCAB_SIZE = len(C2I) + 1
MAX_LEN = 300  # 最大字符数，超过截断，不足padding


# ============================================================
# 1. 数据加载
# ============================================================

def load_data(subset=None):
    """
    加载伪标签数据

    Args:
        subset: 取前N条（方便快速实验），None=全量

    Returns:
        texts: list[str]，原始对话文本
        s_labels: list[str]，S层级标签 ('1','2','3')
        full_labels: list[str]，完整子类标签 ('1.1','2.5',...)
    """
    print(f"加载数据: {DATA_PATH}")
    with open(DATA_PATH, encoding='utf-8') as f:
        data = json.load(f)

    if subset and subset < len(data):
        data = data[:subset]
        print(f"  取子集: {subset} 条")
    else:
        print(f"  全量: {len(data)} 条")

    texts = []
    s_labels = []
    full_labels = []

    for item in data:
        # 拼接对话文本
        parts = [
            item.get('question_title', ''),
            item.get('question_content', ''),
        ]
        for a in item.get('answers', []):
            for d in a.get('dialogs', []):
                parts.append(d.get('content', ''))
        text = ' '.join(parts)

        label = item['labels']['label']  # 如 '1.3', '2.1'
        texts.append(text)
        s_labels.append(label[0])       # '1'
        full_labels.append(label)        # '1.3'

    return texts, s_labels, full_labels


def build_char_sequences(texts, max_len=MAX_LEN):
    """
    将文本转为字符索引序列

    原理：每个字符查表得到索引，不足补0，超长截断。
    为什么不用分词？中文分词有误差，字符级可以保留所有信息。
    """
    sequences = []
    for text in texts:
        # 去空白
        text = re.sub(r'\s+', '', text)[:max_len]
        # 查表转索引
        seq = [C2I.get(c, 0) for c in text]
        # padding 到固定长度
        if len(seq) < max_len:
            seq += [0] * (max_len - len(seq))
        sequences.append(seq[:max_len])
    return np.array(sequences, dtype=np.int64)


def build_tfidf_features(texts, max_features=5000):
    """TF-IDF 特征提取"""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),      # 单字+双字组合
        max_features=max_features,
        sublinear_tf=True,       # 用 1+log(tf) 代替原始 tf，平滑高频词
    )
    X = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    return X, vectorizer


# ============================================================
# 2. 模型定义
# ============================================================

class MLP(nn.Module):
    """
    多层感知机 + TF-IDF 特征

    结构：512 → 256 → N类
    每个全连接层后跟 BatchNorm + ReLU + Dropout

    BatchNorm：加速收敛，缓解过拟合
    Dropout：随机丢弃神经元，防止过拟合
    """
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class CharCNN(nn.Module):
    """
    字符级卷积神经网络

    原理：
    1. Embedding: 每个字符映射为稠密向量 (字符索引 → 128维向量)
    2. Conv1d: 多个不同大小的卷积核在字符序列上滑动
       - 卷积核大小=3：捕捉3个字组成的模式（如"不开心"）
       - 卷积核大小=5：捕捉5个字组成的模式（如"心情不好"）
       - 卷积核大小=7：捕捉7个字组成的模式（如"最近压力很大"）
    3. MaxPooling: 取每个通道的最大值，保留最显著特征
    4. FC: 拼接所有卷积特征 → 分类

    优势：不需要分词，直接对字符序列建模，避免分词错误传播。
    """
    def __init__(self, vocab_size, n_classes, embed_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 多尺度卷积核：捕捉不同n-gram级别的模式
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, 64, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),  # 全局最大池化 → 每个通道1个值
            )
            for k in [3, 5, 7]  # 3种窗口大小
        ])

        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(64 * 3, n_classes),  # 3种卷积核 × 64通道
        )

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embed(x)             # → (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)        # → (batch, embed_dim, seq_len)  Conv1d需要通道维在第2维

        # 每个卷积核独立处理 → 池化 → 拼接
        features = []
        for conv in self.convs:
            h = conv(x)               # → (batch, 64, 1)
            features.append(h.squeeze(-1))  # → (batch, 64)

        x = torch.cat(features, dim=1)  # → (batch, 192)
        return self.fc(x)


# ============================================================
# 3. 训练工具函数
# ============================================================

def train_epoch(model, loader, criterion, optimizer):
    """训练一个epoch，返回平均loss"""
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # 梯度裁剪：防止梯度爆炸
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader):
    """评估模型，返回准确率"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    return correct / total, all_preds, all_labels


def train_model(model, X_train, y_train, X_val, y_val,
                epochs=30, batch_size=64, lr=0.001, weight_decay=1e-4,
                model_name='model'):
    """
    通用训练循环

    Args:
        weight_decay: L2正则化系数，越大越抑制过拟合

    Returns:
        best_acc: 最佳验证准确率
        best_state: 最佳模型参数
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 转为Tensor + DataLoader
    X_train_t = torch.tensor(X_train, dtype=torch.long if 'int' in str(X_train.dtype) else torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.long if 'int' in str(X_val.dtype) else torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size,
    )

    best_acc = 0
    best_state = None
    t0 = time.time()

    print(f"\n  训练 {model_name} ...")
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, criterion, optimizer)
        acc, _, _ = evaluate(model, val_loader)
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:2d}/{epochs} loss={loss:.4f} val_acc={acc:.4f}")

    print(f"  >> {model_name} 最佳: {best_acc:.4f} ({time.time()-t0:.1f}s)")

    # 恢复最佳参数
    if best_state:
        model.load_state_dict(best_state)

    return best_acc, model


# ============================================================
# 4. 主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=int, default=None,
                        help='取前N条数据快速实验')
    parser.add_argument('--epochs', type=int, default=25,
                        help='训练轮数')
    args = parser.parse_args()

    print("=" * 60)
    print("心理咨询对话分类 — 模型对比实验")
    print(f"设备: {device}")
    print("=" * 60)

    # ---- 4.1 加载数据 ----
    print("\n▶ 第1步: 加载数据")
    texts, s_labels, full_labels = load_data(subset=args.subset)
    n_total = len(texts)
    print(f"  总计: {n_total} 条对话")

    # ---- 4.2 Stage 1: S层级分类 (S1/S2/S3) ----
    print("\n" + "=" * 60)
    print("▶ 第2步: Stage 1 — S层级分类 (S1/S2/S3)")
    print("=" * 60)

    # 标签编码: '1'→0, '2'→1, '3'→2
    s_map = {'1': 0, '2': 1, '3': 2}
    y_s = np.array([s_map[l] for l in s_labels])

    # 统计分布
    dist = Counter(s_labels)
    for k in ['1', '2', '3']:
        name = ['S1日常困扰', 'S2心理障碍', 'S3紧急危机'][int(k) - 1]
        print(f"  {name}: {dist[k]:>6}条 ({dist[k]/n_total*100:5.1f}%)")

    # 划分: 训练70% / 验证15% / 测试15%
    # 先用70%训练，剩下30%分两半
    X_temp, X_test_s, y_temp, y_test_s = train_test_split(
        np.arange(n_total), y_s, test_size=0.15, random_state=42, stratify=y_s
    )
    X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, random_state=42, stratify=y_temp
    )
    print(f"  划分: 训练{len(X_train_s)} / 验证{len(X_val_s)} / 测试{len(X_test_s)}")

    ### ---- 4.2.A 基线: LR + TF-IDF ----
    print("\n--- 模型A: Logistic Regression + TF-IDF (基线) ---")

    # TF-IDF: 文本 → 稀疏特征向量
    # 原理：TF = 词频, IDF = log(总文档数/包含该词的文档数)
    # TF-IDF = TF × IDF，高值表示"在该文档中频繁出现但在整体中罕见的词"
    vec_s = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    X_tfidf_s = vec_s.fit_transform([texts[i] for i in X_temp]).toarray().astype(np.float32)

    # 进一步划分TF-IDF特征
    n_train = len(X_train_s)
    X_tr_tf = X_tfidf_s[:n_train]
    X_val_tf = X_tfidf_s[n_train:]

    t0 = time.time()
    lr_s = LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced', random_state=42)
    lr_s.fit(X_tr_tf, y_train_s)
    lr_pred = lr_s.predict(X_val_tf)
    lr_acc_s = accuracy_score(y_val_s, lr_pred)
    print(f"  LR + TF-IDF: {lr_acc_s:.4f} ({time.time()-t0:.1f}s)")
    print(classification_report(y_val_s, lr_pred, target_names=['S1', 'S2', 'S3']))

    ### ---- 4.2.B MLP + TF-IDF ----
    print("\n--- 模型B: MLP + TF-IDF ---")
    mlp_s = MLP(X_tfidf_s.shape[1], 3)
    best_mlp_s, _ = train_model(
        mlp_s, X_tr_tf, y_train_s, X_val_tf, y_val_s,
        epochs=args.epochs, model_name='MLP+TF-IDF (Stage1)',
    )

    ### ---- 4.2.C CharCNN ----
    print("\n--- 模型C: CharCNN (字符级) ---")
    # 用索引取数据，TF-IDF时已经shuffle了
    train_idx = X_train_s  # 已经是原始索引
    val_idx = X_val_s

    X_char_all = build_char_sequences(texts)
    X_tr_c = X_char_all[train_idx]
    X_val_c = X_char_all[val_idx]

    cnn_s = CharCNN(VOCAB_SIZE, 3)
    best_cnn_s, cnn_model_s = train_model(
        cnn_s, X_tr_c, y_train_s, X_val_c, y_val_s,
        epochs=args.epochs, model_name='CharCNN (Stage1)',
    )

    ### ---- Stage 1 汇总 ----
    print(f"\n  {'='*50}")
    print(f"  Stage 1 对比 (S1/S2/S3 3分类)")
    print(f"  {'='*50}")
    print(f"  {'模型':<25} {'准确率':>8}")
    print(f"  {'-'*35}")
    print(f"  {'LR + TF-IDF (基线)':<25} {lr_acc_s:>8.4f}")
    print(f"  {'MLP + TF-IDF':<25} {best_mlp_s:>8.4f}")
    print(f"  {'CharCNN (字符级)':<25} {best_cnn_s:>8.4f}")

    # ---- 4.3 Stage 2: S1子类分类 (17类) ----
    # 原理：S1样本最多，子类也最多(17类)，最具挑战性
    # S2(9类)和S3(5类)样本较少，留作后续优化

    print("\n" + "=" * 60)
    print("▶ 第3步: Stage 2 — S1子类分类 (17类)")
    print("=" * 60)

    # 筛选S1样本
    s1_indices = [i for i in range(n_total) if full_labels[i].startswith('1.')]
    s1_texts = [texts[i] for i in s1_indices]
    s1_sub_labels = [full_labels[i] for i in s1_indices]

    # 编码子类
    s1_classes = sorted(set(s1_sub_labels))
    s1_c2i = {c: i for i, c in enumerate(s1_classes)}
    y_s1 = np.array([s1_c2i[l] for l in s1_sub_labels])

    print(f"  S1样本: {len(s1_texts)}条, {len(s1_classes)}个子类")
    print(f"  子类: {', '.join(s1_classes[:5])} ...")

    # 划分
    s1_train, s1_val, y_s1_tr, y_s1_val = train_test_split(
        np.arange(len(s1_texts)), y_s1, test_size=0.2, random_state=42, stratify=y_s1
    )

    ### ---- Stage 2: LR基线 ----
    print("\n--- 模型A: LR + TF-IDF (基线) ---")
    vec_s1 = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    X_s1_tf = vec_s1.fit_transform([s1_texts[i] for i in s1_train]).toarray().astype(np.float32)
    X_s1_val_tf = vec_s1.transform([s1_texts[i] for i in s1_val]).toarray().astype(np.float32)

    t0 = time.time()
    lr_s1 = LogisticRegression(max_iter=3000, C=1.0, class_weight='balanced', random_state=42)
    lr_s1.fit(X_s1_tf, y_s1_tr)
    lr_s1_pred = lr_s1.predict(X_s1_val_tf)
    lr_s1_acc = accuracy_score(y_s1_val, lr_s1_pred)
    print(f"  LR + TF-IDF: {lr_s1_acc:.4f} ({time.time()-t0:.1f}s)")

    ### ---- Stage 2: CharCNN ----
    print("\n--- 模型C: CharCNN ---")
    X_c_s1 = build_char_sequences(s1_texts)
    X_s1_tr_c = X_c_s1[s1_train]
    X_s1_val_c = X_c_s1[s1_val]

    cnn_s1 = CharCNN(VOCAB_SIZE, len(s1_classes))
    best_cnn_s1, _ = train_model(
        cnn_s1, X_s1_tr_c, y_s1_tr, X_s1_val_c, y_s1_val,
        epochs=args.epochs, model_name='CharCNN (S1子类)',
    )

    ### ---- Stage 2 汇总 ----
    print(f"\n  {'='*50}")
    print(f"  Stage 2 对比 (S1子类 {len(s1_classes)}类)")
    print(f"  {'='*50}")
    print(f"  {'LR + TF-IDF':<25} {lr_s1_acc:>8.4f}")
    print(f"  {'CharCNN':<25} {best_cnn_s1:>8.4f}")

    # ---- 4.4 总结 ----
    print("\n" + "=" * 60)
    print("📊 实验结果总结")
    print("=" * 60)
    print(f"""
  Stage 1 (S1/S2/S3):
    LR + TF-IDF:     {lr_acc_s:.2%}
    MLP + TF-IDF:    {best_mlp_s:.2%}
    CharCNN:          {best_cnn_s:.2%}

  Stage 2 (S1子类 {len(s1_classes)}类):
    LR + TF-IDF:     {lr_s1_acc:.2%}
    CharCNN:          {best_cnn_s1:.2%}

  🔑 关键观察:
    - {'深度学习与传统方法差距明显' if max(best_cnn_s, best_mlp_s) > lr_acc_s + 0.01 else '深度学习方法与传统基线差距不大'}
    - {'CharCNN 在粗粒度分类上更强' if best_cnn_s > best_mlp_s else 'MLP 略优于 CharCNN'}
    - 子类分类难度远大于粗分类（更多类、更细粒度）
    - 实际应用中可混合使用：CharCNN做Stage1 + LR做Stage2
""")


if __name__ == '__main__':
    main()
