#!/usr/bin/env python3
"""
重训 v2: 人工修正 + 原始3000人标 -> 合并 -> 加权重训 -> 全量预测
"""
import json, re
from pathlib import Path
from collections import Counter
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

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

DATA = Path('data')
OUT = Path('data/人工标注')

# 1. 原始3000人标
with open(OUT / 'No-01_待标注_3000_已标注.json', encoding='utf-8') as f:
    seed = json.load(f)
print(f"原始人标: {len(seed)} 条")

base_map = {item['question_id']: item['labels']['label'] for item in seed}

# 2. 人工修正 (250条)
with open(OUT / 'No-01_待修正_极低_part1of28_已标注.json', encoding='utf-8') as f:
    corrected = json.load(f)
print(f"人工修正: {len(corrected)} 条")

correction_map = {}
for item in corrected:
    qid = item['question_id']
    new_label = item['labels']['label']
    if new_label and new_label != base_map.get(qid, ''):
        correction_map[qid] = new_label
print(f"标签变更: {len(correction_map)} 条")

# 3. 合并
merged = dict(base_map)
merged.update(correction_map)

with open(DATA / 'No-01.json', encoding='utf-8') as f:
    full_data = json.load(f)

train_texts, train_labels, train_weights = [], [], []
n_corrected = 0
for item in full_data:
    qid = item['question_id']
    if qid in merged:
        train_texts.append(bld(item))
        train_labels.append(merged[qid])
        w = 15.0 if qid in correction_map else 1.0
        train_weights.append(w)
        if qid in correction_map:
            n_corrected += 1

print(f"\n训练集: {len(train_texts)} 条 ({n_corrected}条修正 weight=15, {len(train_texts)-n_corrected}条 weight=1)")
print(f"类别数: {len(set(train_labels))}/31")

# 4. 训练
vec = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b',
    ngram_range=(1,3), max_features=10000, min_df=1, max_df=0.9, sublinear_tf=True)
X = vec.fit_transform(train_texts)
sw = np.array(train_weights)

try:
    X_tr, X_te, y_tr, y_te, sw_tr, sw_te = train_test_split(
        X, train_labels, sw, test_size=0.2, random_state=SEED, stratify=train_labels)
except:
    X_tr, X_te, y_tr, y_te, sw_tr, sw_te = train_test_split(
        X, train_labels, sw, test_size=0.2, random_state=SEED)

clf = LogisticRegression(max_iter=3000, C=1.0, random_state=SEED, class_weight='balanced')
clf.fit(X_tr, y_tr, sample_weight=sw_tr)

# 5. 评估
y_pred = clf.predict(X_te)
acc = accuracy_score(y_te, y_pred)
print(f"\n测试准确率: {acc:.4f} ({len(y_te)} 条)")

s_map = {'1':'S1','2':'S2','3':'S3'}
cm = confusion_matrix([s_map[l[0]] for l in y_te], [s_map[l[0]] for l in y_pred], labels=['S1','S2','S3'])
print(f"\n混淆矩阵 (S层级):")
print(f"{'':>8} {'S1':>6} {'S2':>6} {'S3':>6}")
for i, lbl in enumerate(['S1','S2','S3']):
    print(f"  {lbl:>6}" + ''.join(f'{cm[i,j]:6d}' for j in range(3)))

corrected_test = [i for i in range(len(y_te)) if sw_te[i] >= 15]
if corrected_test:
    ca = accuracy_score([y_te[i] for i in corrected_test], [y_pred[i] for i in corrected_test])
    print(f"修正样本准确率: {ca:.4f} ({len(corrected_test)} 条)")

# 6. 全量预测
print(f"\n--- 全量预测 No-01 (8366条) ---")

# 加载旧模型结果对比
old_preds = None
old_path = OUT / 'No-01_半监督_全量标注.json'
if old_path.exists():
    with open(old_path, encoding='utf-8') as f:
        old_data = json.load(f)
    old_preds = [item['labels']['label'] for item in old_data]

X_full = vec.transform([bld(item) for item in full_data])
new_preds = clf.predict(X_full)

for i, item in enumerate(full_data):
    item['labels'] = {'label': new_preds[i]}

out_path = OUT / 'No-01_最终标注.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(full_data, f, ensure_ascii=False, indent=2)

d = Counter(new_preds)
s1 = sum(v for k,v in d.items() if k.startswith('1.'))
s2 = sum(v for k,v in d.items() if k.startswith('2.'))
s3 = sum(v for k,v in d.items() if k.startswith('3.'))
s3c = sorted([k for k in d if k.startswith('3.')])
print(f"S1={s1}({s1/8366*100:.1f}%) S2={s2}({s2/8366*100:.1f}%) S3={s3}({s3/8366*100:.1f}%)")
print(f"类数: {len(d)}/31 | S3子类: {len(s3c)}/5 {s3c}")

if old_preds:
    changes = sum(1 for i in range(8366) if old_preds[i] != new_preds[i])
    print(f"与半监督模型对比: {changes} 条标签改变")

print(f"\n输出: {out_path}")
joblib.dump(clf, OUT / 'No-01_最终标注_model.pkl')
joblib.dump(vec, OUT / 'No-01_最终标注_vectorizer.pkl')
