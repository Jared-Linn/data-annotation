#!/usr/bin/env python3
"""混淆矩阵深度分析 + 热力图"""
import json, re
from pathlib import Path
from collections import Counter
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

CHART_DIR = Path('analysis/output')
CHART_DIR.mkdir(parents=True, exist_ok=True)

DATA = Path('data')
OUT = Path('ml/output')

with open(DATA / 'stopwords.txt', encoding='utf-8') as f:
    STOP_WORDS = set(line.strip() for line in f if line.strip())

def cut(t):
    return ' '.join(w for w in jieba.cut(t) if w.strip() and w not in STOP_WORDS)
def cln(t):
    return re.sub(r'\s+', '', re.sub(r'[^一-鿿\w]', '', t))
def bld(item):
    p = [item.get('question_title',''), item.get('question_content','')]
    for a in item.get('answers',[]):
        for d in a.get('dialogs',[]): p.append(d.get('content',''))
    return cut(cln(' '.join(p)))


def plot_confusion_heatmap(cm, labels, title, save_path):
    """画混淆矩阵热力图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=cm.max()*0.8)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('预测', fontsize=11)
    ax.set_ylabel('真实', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    # 标注数值
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i][j]), ha='center', va='center',
                    fontsize=6, color='white' if cm[i][j] > cm.max()*0.5 else 'black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"  已保存: {save_path}")


def analyze_confusion():
    """加载模型 -> 预测 -> 混淆矩阵分析"""
    print("=" * 60)
    print("混淆矩阵深度分析")
    print("=" * 60)

    # 加载训练数据
    with open('data/人工标注/No-01_待标注_3000_已标注.json', encoding='utf-8') as f:
        seed = json.load(f)
    txts = [bld(it) for it in seed]
    lbls = [it['labels']['label'] for it in seed]

    # S层级
    s_levels = [l[0] for l in lbls]

    v = TfidfVectorizer(ngram_range=(1,1), max_features=10000, sublinear_tf=True)
    X = v.fit_transform(txts)
    X_tr, X_te, y_tr, y_te = train_test_split(X, s_levels, test_size=0.2, random_state=42, stratify=s_levels)
    c = LogisticRegression(class_weight='balanced', max_iter=3000, random_state=42)
    c.fit(X_tr, y_tr)
    y_pred = c.predict(X_te)

    # S层级混淆矩阵
    cm_s = confusion_matrix(y_te, y_pred, labels=['1','2','3'])
    print(f"\nS层级混淆矩阵 ({len(y_te)}条测试):")
    print(f"{'':>8} {'S1':>6} {'S2':>6} {'S3':>6}")
    for i, lbl in enumerate(['S1','S2','S3']):
        print(f"  {lbl:>6}" + ''.join(f'{cm_s[i,j]:6d}' for j in range(3)))

    plot_confusion_heatmap(cm_s, ['S1','S2','S3'], 'S层级混淆矩阵',
                           str(CHART_DIR / 'confusion_S_level.png'))

    # 各类目混淆矩阵（只显示有足够测试样本的类）
    v2 = TfidfVectorizer(ngram_range=(1,1), max_features=10000, sublinear_tf=True)
    X2 = v2.fit_transform(txts)
    c2 = LogisticRegression(class_weight='balanced', max_iter=3000, random_state=42)
    c2.fit(X2, lbls)

    # 全量预测分析各类间混淆
    all_pred = c2.predict(X2)
    confusion_pairs = Counter()
    for true_l, pred_l in zip(lbls, all_pred):
        if true_l != pred_l:
            confusion_pairs[(true_l, pred_l)] += 1

    print(f"\n各类间混淆 Top-15:")
    print(f"{'真实→预测':>12} {'次数':>6} {'占比':>6}")
    total_errors = sum(confusion_pairs.values())
    for (t, p), cnt in confusion_pairs.most_common(15):
        print(f"  {t}→{p}     {cnt:>4}  {cnt/total_errors*100:>5.1f}%")

    # 子类混淆热力图（只取前20个常见类）
    top_classes = [l for l, _ in Counter(lbls).most_common(20)]
    cm_sub = confusion_matrix(lbls, all_pred, labels=top_classes)
    top_names = [f'{l}' for l in top_classes]
    plot_confusion_heatmap(cm_sub, top_names, '子类混淆矩阵 (Top20)',
                           str(CHART_DIR / 'confusion_subclass.png'))


if __name__ == '__main__':
    analyze_confusion()
