#!/usr/bin/env python3
"""模型优化实验 - 系统诊断当前模型瓶颈"""
import json, re
from pathlib import Path
from collections import Counter
import numpy as np
import jieba
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

print("=" * 60)
print("Stage 2 S1 子类分类瓶颈诊断")
print("=" * 60)

s1_mask = [l == '1' for l in s_levels]
s1_txts = [t for t, m in zip(txts, s1_mask) if m]
s1_lbls = [l for l, m in zip(lbls, s1_mask) if m]
print(f"\nS1样本: {len(s1_txts)}条, {len(set(s1_lbls))}类")
for lbl in sorted(Counter(s1_lbls)):
    print(f"  {lbl}: {Counter(s1_lbls)[lbl]}")

# 基线: word 1-gram + LR
v = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b',
    ngram_range=(1,1), max_features=5000, min_df=1, max_df=0.9, sublinear_tf=True)
X = v.fit_transform(s1_txts)
X_tr, X_te, y_tr, y_te = train_test_split(X, s1_lbls, test_size=0.2, random_state=42, stratify=s1_lbls)
clf = LogisticRegression(max_iter=3000, C=1.0, random_state=42, class_weight='balanced')
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_te)
acc = accuracy_score(y_te, y_pred)
print(f"\n【基线】word 1-gram + LR: {acc:.4f}")

# 实验1: word + char 混合
print(f"\n实验1: word+char 2-4gram 混合")
vw = TfidfVectorizer(ngram_range=(1,1), max_features=3000, sublinear_tf=True)
vc = TfidfVectorizer(analyzer='char', ngram_range=(2,4), max_features=3000, sublinear_tf=True)
X_hybrid = hstack([vw.fit_transform(s1_txts), vc.fit_transform(s1_txts)])
X_tr_h, X_te_h, y_tr_h, y_te_h = train_test_split(X_hybrid, s1_lbls, test_size=0.2, random_state=42, stratify=s1_lbls)
clf_h = LogisticRegression(max_iter=3000, C=1.0, random_state=42, class_weight='balanced')
clf_h.fit(X_tr_h, y_tr_h)
print(f"  word+char + LR: {accuracy_score(y_te_h, clf_h.predict(X_te_h)):.4f}")

# 实验2: char only
print(f"\n实验2: char 2-4gram 纯字符")
Xc = vc.fit_transform(s1_txts)
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(Xc, s1_lbls, test_size=0.2, random_state=42, stratify=s1_lbls)
clf_c = LogisticRegression(max_iter=3000, C=1.0, random_state=42, class_weight='balanced')
clf_c.fit(X_tr_c, y_tr_c)
print(f"  char + LR: {accuracy_score(y_te_c, clf_c.predict(X_te_c)):.4f}")

# 实验3: 不同分类器
print(f"\n实验3: 不同分类器 (word 1-gram)")
for name, model in [
    ('LinearSVC', LinearSVC(max_iter=3000, C=1.0, random_state=42, class_weight='balanced')),
    ('RandomForest', RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, class_weight='balanced')),
]:
    model.fit(X_tr, y_tr)
    print(f"  {name}: {accuracy_score(y_te, model.predict(X_te)):.4f}")

# 实验4: 混淆分析
print(f"\n实验4: 混淆模式分析")
cm = confusion_matrix(y_te, y_pred, labels=sorted(set(s1_lbls)))
labels_list = sorted(set(s1_lbls))
total_err = cm.sum() - cm.trace()
for i in range(len(labels_list)):
    for j in range(len(labels_list)):
        if i != j and cm[i][j] >= 3:
            print(f"  {labels_list[i]} -> {labels_list[j]}: {cm[i][j]}次")
print(f"总错误: {total_err}/{len(y_te)} ({total_err/len(y_te)*100:.1f}%)")
