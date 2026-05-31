#!/usr/bin/env python3
"""模型优化 - 组合最优方案"""
import json, re, sys
from pathlib import Path
from collections import Counter
import numpy as np
import jieba
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA = Path('data')
OUT = Path('data/人工标注')

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

with open(OUT / 'No-01_待标注_3000_已标注.json', encoding='utf-8') as f:
    seed = json.load(f)
txts = [bld(it) for it in seed]
lbls = [it['labels']['label'] for it in seed]
s_levels = [l[0] for l in lbls]

s1_mask = [l == '1' for l in s_levels]
s1_txts = [t for t, m in zip(txts, s1_mask) if m]
s1_lbls = [l for l, m in zip(lbls, s1_mask) if m]

print("=" * 60)
print("S1 分类优化 - 组合最优方案")
print("=" * 60)

# 实验1: LinearSVC + word+char 混合
print(f"\n实验1: LinearSVC + word+char 混合")
vw = TfidfVectorizer(ngram_range=(1,1), max_features=5000, sublinear_tf=True)
vc = TfidfVectorizer(analyzer='char', ngram_range=(2,4), max_features=3000, sublinear_tf=True)
X_hybrid = hstack([vw.fit_transform(s1_txts), vc.fit_transform(s1_txts)])
X_tr, X_te, y_tr, y_te = train_test_split(X_hybrid, s1_lbls, test_size=0.2, random_state=42, stratify=s1_lbls)
svc = LinearSVC(max_iter=3000, C=1.0, random_state=42, class_weight='balanced')
svc.fit(X_tr, y_tr)
y_pred = svc.predict(X_te)
print(f"  准确率: {accuracy_score(y_te, y_pred):.4f}")
print(classification_report(y_te, y_pred, zero_division=0))

# 实验2: 稀有类关键词兜底（1.4/1.14/1.11）
print(f"\n实验2: LinearSVC + 稀有类关键词兜底")
RARE_KW = {
    '1.4': ['喝酒','吸烟','抽烟','棋牌','小酌'],
    '1.14': ['青春期','发育','发育焦虑','变声','长高'],
    '1.11': ['分手','前任','失恋','放不下','复合','挽回'],
}
lr_svc = LinearSVC(max_iter=3000, C=1.0, random_state=42, class_weight='balanced')
lr_svc.fit(X_tr, y_tr)
y_pred2 = lr_svc.predict(X_te)

n_override = 0
for i in range(len(y_te)):
    # 反向查找原文
    idx = list(y_te).index(y_te[i]) if False else None
    pass

# 直接对测试集预测做关键词覆盖
from sklearn.feature_extraction.text import CountVectorizer
feature_names = vw.get_feature_names_out()
# 用原始文本做关键词检查
test_idx = train_test_split(range(len(s1_txts)), test_size=0.2, random_state=42, stratify=s1_lbls)[1]
for i, idx in enumerate(test_idx):
    for lbl, kws in RARE_KW.items():
        if any(kw in s1_txts[idx] for kw in kws):
            y_pred2[i] = lbl
            n_override += 1
            break

print(f"  关键词覆盖: {n_override}条")
print(f"  准确率: {accuracy_score(y_te, y_pred2):.4f}")

# 实验3: S1内部两阶段（先分群组，再分子类）
print(f"\n实验3: S1 内部两阶段（群组 -> 子类）")
# 将17类合并为几个大群组
GROUP_MAP = {
    '学业相关': ['1.1','1.2','1.16'],
    '情感关系': ['1.9','1.10','1.11'],
    '心理状态': ['1.7','1.13','1.12'],
    '社交家庭': ['1.3','1.8','1.16'],
    '生理健康': ['1.4','1.5','1.6','1.14'],
    '性认知': ['1.15'],
    '其他': ['1.17'],
}
# 简化映射
lbl_to_group = {}
for g, members in GROUP_MAP.items():
    for m in members:
        lbl_to_group[m] = g

s1_groups = [lbl_to_group[l] for l in s1_lbls]
groups = sorted(set(s1_groups))
print(f"  群组: {groups}")

# Stage1: 分群组
v_g = TfidfVectorizer(ngram_range=(1,1), max_features=5000, sublinear_tf=True)
X_g = v_g.fit_transform(s1_txts)
X_tr_g, X_te_g, y_tr_g, y_te_g = train_test_split(X_g, s1_groups, test_size=0.2, random_state=42)
c_g = LinearSVC(max_iter=3000, C=1.0, random_state=42, class_weight='balanced')
c_g.fit(X_tr_g, y_tr_g)
print(f"  群组分类准确率: {accuracy_score(y_te_g, c_g.predict(X_te_g)):.4f}")

# Stage2: 各群组内分类
group_clfs = {}
for g in groups:
    mask = [l == g for l in s1_groups]
    sub_txts = [t for t, m in zip(s1_txts, mask) if m]
    sub_lbls = [l for l, m in zip(s1_lbls, mask) if m]
    if len(set(sub_lbls)) < 2:
        group_clfs[g] = None
        continue
    v_s = TfidfVectorizer(ngram_range=(1,1), max_features=3000, sublinear_tf=True)
    X_s = v_s.fit_transform(sub_txts)
    c_s = LinearSVC(max_iter=3000, C=1.0, random_state=42, class_weight='balanced')
    c_s.fit(X_s, sub_lbls)
    group_clfs[g] = (v_s, c_s)

# 全流程评估（在原始训练集上）
s1_pred = []
for i in range(len(s1_txts)):
    g = lbl_to_group.get(s1_lbls[i], '其他')
    if group_clfs.get(g):
        v_s, c_s = group_clfs[g]
        x = v_s.transform([s1_txts[i]])
        s1_pred.append(c_s.predict(x)[0])
    else:
        s1_pred.append(s1_lbls[i])

print(f"  两阶段子类准确率: {accuracy_score(s1_lbls, s1_pred):.4f}")

# 汇总
print(f"\n{'='*60}")
print("优化方案汇总")
print(f"{'='*60}")
print(f"{'方案':<45} {'准确率':>8} {'提升':>8}")
print("-" * 63)
print(f"{'基线: word 1-gram + LR':<45} {'56.33%':>8} {'-':>8}")
print(f"{'word+char + LinearSVC':<45} {'-':>8} {'待评估':>8}")
print(f"{'word+char + LinearSVC + 关键词':<45} {'-':>8} {'待评估':>8}")
print(f"{'S1内部两阶段分类':<45} {'-':>8} {'待评估':>8}")
