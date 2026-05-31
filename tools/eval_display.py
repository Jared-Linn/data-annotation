#!/usr/bin/env python3
"""运行此脚本，全选终端输出 -> 截图 -> 贴到Word"""
import json, re, glob
from pathlib import Path
from collections import Counter
import numpy as np, jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

SEED = 42
with open('data/stopwords.txt', encoding='utf-8') as f:
    STOP_WORDS = set(line.strip() for line in f if line.strip())

def cut_ws(t):
    return ' '.join(w for w in jieba.cut(t) if w.strip() and w not in STOP_WORDS)
def cln(t):
    return re.sub(r'\s+', '', re.sub(r'[^一-鿿\w]', '', t))
def bld(item):
    p = [item.get('question_title',''), item.get('question_content','')]
    for a in item.get('answers',[]):
        for d in a.get('dialogs',[]): p.append(d.get('content',''))
    return cut_ws(cln(' '.join(p)))

OUT = Path('data/人工标注')
with open(OUT / 'No-01_待标注_3000_已标注.json', encoding='utf-8') as f:
    seed = json.load(f)
base_map = {item['question_id']: item['labels']['label'] for item in seed}
annotated = sorted(glob.glob(str(OUT / '*_已标注.json')))
annotated = [f for f in annotated if '3000_已标注' not in f]
corr = {}
for fp in annotated:
    with open(fp, encoding='utf-8') as f:
        for item in json.load(f):
            qid, nl = item['question_id'], item['labels']['label']
            if nl and nl != base_map.get(qid,''): corr[qid] = nl
merged = dict(base_map); merged.update(corr)

with open('data/No-01.json', encoding='utf-8') as f:
    full_data = json.load(f)

train_texts, train_labels = [], []
for item in full_data:
    qid = item['question_id']
    if qid in merged:
        train_texts.append(bld(item))
        train_labels.append(merged[qid])
train_s = [l[0] for l in train_labels]

print("=" * 60)
print("【一致性检验】5折交叉验证 (S层级)")
v = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b',
    ngram_range=(1,1), max_features=10000, min_df=1, max_df=0.9, sublinear_tf=True)
X = v.fit_transform(train_texts)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
clf = LogisticRegression(max_iter=3000, C=1.0, random_state=SEED, class_weight='balanced')
scores = cross_val_score(clf, X, train_s, cv=cv, scoring='accuracy')
print(f"  各折准确率: {[f'{s:.4f}' for s in scores]}")
print(f"  平均准确率: {scores.mean():.4f}")
print(f"  标准差:     {scores.std():.4f}")
print(f"  结论: 模型稳定" if scores.std() < 0.02 else "结论: 稳定性一般")

print("\n" + "=" * 60)
print("【分类报告 + 混淆矩阵】")
X_tr, X_te, y_tr, y_te = train_test_split(X, train_s, test_size=0.2, random_state=SEED, stratify=train_s)
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_te)
print(classification_report(y_te, y_pred, target_names=['S1','S2','S3'], digits=4))
cm = confusion_matrix(y_te, y_pred, labels=['1','2','3'])
print(f"{'':>8} {'S1':>6} {'S2':>6} {'S3':>6}")
for i, lbl in enumerate(['S1','S2','S3']):
    print(f"  {lbl:>6}" + ''.join(f'{cm[i,j]:6d}' for j in range(3)))

print("\n" + "=" * 60)
print("【三份数据最终分布】")
for tgt in ['No-01','No-02','No-03']:
    with open(OUT / f'{tgt}_最终版.json', encoding='utf-8') as f:
        items = json.load(f)
    labels = [item['labels']['label'] for item in items]
    n = len(labels)
    s1 = sum(1 for l in labels if l.startswith('1.'))/n*100
    s2 = sum(1 for l in labels if l.startswith('2.'))/n*100
    s3 = sum(1 for l in labels if l.startswith('3.'))/n*100
    cls = len(set(labels))
    s3c = len([k for k in set(labels) if k.startswith('3.')])
    print(f"  {tgt}: S1={s1:.1f}%  S2={s2:.1f}%  S3={s3:.1f}%  类={cls}/31  S3子类={s3c}/5")
print("=" * 60)
print("\n截图方法: 选中全部文字 -> Alt+PrintScreen -> 贴到Word")
