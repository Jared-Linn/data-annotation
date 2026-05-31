#!/usr/bin/env python3
"""
主动学习 - 训练循环
管理标注集 -> 训练 -> 采样 -> 伪标签 -> 重训 迭代
"""
import json, re, time
from pathlib import Path
from collections import Counter
import numpy as np, jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from .sampler import select_samples

DATA = Path('data')
OUT = Path('data/人工标注')

with open(DATA / 'stopwords.txt', encoding='utf-8') as f:
    STOP_WORDS = set(line.strip() for line in f if line.strip())


def _cut(t):
    return ' '.join(w for w in jieba.cut(t) if w.strip() and w not in STOP_WORDS)


def _cln(t):
    return re.sub(r'\s+', '', re.sub(r'[^一-鿿\w]', '', t))


def _bld(item):
    p = [item.get('question_title', ''), item.get('question_content', '')]
    for a in item.get('answers', []):
        for d in a.get('dialogs', []):
            p.append(d.get('content', ''))
    return _cut(_cln(' '.join(p)))


def _get_full(item):
    t = item.get('question_title', '') + ' ' + item.get('question_content', '')
    for a in item.get('answers', []):
        for d in a.get('dialogs', []):
            t += ' ' + d.get('content', '')
    return t


class ActiveLearningLoop:
    """
    主动学习循环

    用法:
        loop = ActiveLearningLoop()
        loop.initialize()                      # 加载已有标注
        loop.run_round(n_select=200)           # 选200条 -> 训练 -> 伪标签 -> 评估
        loop.summary()                         # 查看进展
    """

    def __init__(self, weight=5.0, strategy='least_confidence'):
        self.weight = weight
        self.strategy = strategy

        # 训练数据
        self.train_texts = []
        self.train_labels = []
        self.weights = []
        self.source = []  # 'human' or 'pseudo'

        # 未标注池
        self.pool_raw = []
        self.pool_texts = []

        # 模型
        self.vec = None
        self.clf = None

        # 历史
        self.history = []

    def initialize(self, max_train_samples=None, pool_files=None):
        """
        初始化: 加载已有标注 + 未标注池
        """
        print("=" * 60)
        print("主动学习初始化")
        print("=" * 60)

        # 加载3000条人工标注作为种子
        with open(OUT / 'No-01_待标注_3000_已标注.json', encoding='utf-8') as f:
            seed = json.load(f)

        train_ids = set()
        for item in seed:
            qid = item['question_id']
            self.train_texts.append(_bld(item))
            self.train_labels.append(item['labels']['label'])
            self.weights.append(1.0)
            self.source.append('human')
            train_ids.add(qid)

        print(f"  种子集: {len(seed)} 条人工标注")
        print(f"  类别: {len(set(self.train_labels))}/31")

        # 2. 构建未标注池（No-01剩余 + No-02 + No-03）
        if pool_files is None:
            pool_files = ['No-01.json', 'No-02.json', 'No-03.json']

        for fn in pool_files:
            with open(DATA / fn, encoding='utf-8') as f:
                items = json.load(f)
            for item in items:
                if item['question_id'] not in train_ids:
                    self.pool_raw.append(item)
                    self.pool_texts.append(_bld(item))

        print(f"  未标注池: {len(self.pool_raw)} 条")
        return self

    def train(self):
        """训练当前模型"""
        self.vec = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b',
                                   ngram_range=(1, 1), max_features=10000,
                                   min_df=1, max_df=0.9, sublinear_tf=True)
        X = self.vec.fit_transform(self.train_texts)
        sw = np.array(self.weights)
        self.clf = LogisticRegression(max_iter=3000, C=1.0, random_state=42,
                                       class_weight='balanced')

        try:
            X_tr, X_te, y_tr, y_te, sw_tr, sw_te = train_test_split(
                X, self.train_labels, sw, test_size=0.2, random_state=42,
                stratify=self.train_labels)
            self.clf.fit(X_tr, y_tr, sample_weight=sw_tr)
            acc = accuracy_score(y_te, self.clf.predict(X_te))
        except Exception:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, self.train_labels, test_size=0.2, random_state=42)
            self.clf.fit(X_tr, y_tr)
            acc = accuracy_score(y_te, self.clf.predict(X_te))

        return acc

    def run_round(self, n_select=200, conf_threshold=0.7, auto_label=True):
        """
        执行一轮主动学习

        参数:
            n_select: 本次选择样本数
            conf_threshold: 自动标注置信阈值
            auto_label: True=自动伪标签, False=打印review任务

        返回:
            round_info: dict
        """
        t0 = time.time()
        print(f"\n--- 主动学习第{len(self.history)+1}轮 (strategy={self.strategy}) ---")

        # 1. 训练
        acc = self.train()
        print(f"  训练: {len(self.train_texts)}条 | 准确率: {acc:.4f}")

        # 2. 对未标注池预测
        X_pool = self.vec.transform(self.pool_texts)
        preds = self.clf.predict(X_pool)
        probs = self.clf.predict_proba(X_pool)
        max_probs = probs.max(axis=1)

        # 3. 选择最不确定的样本
        selected_idx, scores = select_samples(probs, min(n_select, len(self.pool_texts)), self.strategy)

        # 4. 处理选中样本
        n_pseudo = 0
        n_review = 0
        for idx in selected_idx:
            item = self.pool_raw[idx]
            text = self.pool_texts[idx]
            pred = preds[idx]
            conf = max_probs[idx]

            if auto_label and conf >= conf_threshold:
                # 自动添加伪标签
                self.train_texts.append(text)
                self.train_labels.append(pred)
                self.weights.append(self.weight)
                self.source.append('pseudo')
                n_pseudo += 1
            else:
                # 需要人工审核
                n_review += 1

        # 5. 从池中移除选中样本
        keep = list(range(len(self.pool_raw)))
        for idx in sorted(selected_idx, reverse=True):
            keep.pop(idx)

        self.pool_raw = [self.pool_raw[i] for i in keep]
        self.pool_texts = [self.pool_texts[i] for i in keep]

        elapsed = time.time() - t0
        round_info = {
            'round': len(self.history) + 1,
            'train_size': len(self.train_texts),
            'pool_size': len(self.pool_raw),
            'accuracy': acc,
            'n_selected': len(selected_idx),
            'n_pseudo': n_pseudo,
            'n_review': n_review,
            'elapsed': f'{elapsed:.1f}s',
        }

        # 评估伪标签质量
        if n_pseudo > 0:
            pseudo_labels = [self.train_labels[-n_pseudo:]]
            pseudo_dist = Counter(pseudo_labels[0])
            round_info['pseudo_dist'] = dict(pseudo_dist.most_common(5))

        self.history.append(round_info)

        print(f"  选中: {len(selected_idx)}条 (伪标签{n_pseudo}, 需审核{n_review})")
        print(f"  池剩余: {len(self.pool_raw)}条 | 耗时: {elapsed:.1f}s")
        return round_info

    def summary(self):
        """打印总结"""
        print("\n" + "=" * 60)
        print("主动学习总结")
        print("=" * 60)

        if not self.history:
            print("  尚未运行任何轮次")
            return

        print(f"\n{'轮次':>4} {'训练集':>6} {'池大小':>6} {'准确率':>8} {'选中':>6} {'伪标签':>6} {'耗时':>8}")
        for h in self.history:
            print(f"  {h['round']:>3}  {h['train_size']:>5}  {h['pool_size']:>5}  "
                  f"{h['accuracy']:.4f}  {h['n_selected']:>5}  {h['n_pseudo']:>5}  {h['elapsed']:>8}")

        final = self.history[-1]
        print(f"\n最终:")
        print(f"  训练集: {final['train_size']}条 ({sum(1 for s in self.source if s=='human')}人工 + {sum(1 for s in self.source if s=='pseudo')}伪标签)")
        print(f"  剩余池: {final['pool_size']}条")
        print(f"  最新准确率: {final['accuracy']:.4f}")
