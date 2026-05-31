#!/usr/bin/env python3
"""
两阶段分类 + S3关键词兜底
Stage1: S1/S2/S3 层级
Stage2: 各层内子类
"""
import json, re, glob
from pathlib import Path
from collections import Counter
import numpy as np, jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

SEED = 42
with open('data/stopwords.txt', encoding='utf-8') as f:
    STOP_WORDS = set(line.strip() for line in f if line.strip())

S3_KW = {
    '3.1': ['正在自杀','跳楼','上吊','割腕','服药自杀','在自杀'],
    '3.2': ['想自杀','自杀计划','准备死','安排后事','写遗书','计划自杀'],
    '3.3': ['自残','划手','割手','烫自己','伤害身体','自伤','划伤'],
    '3.4': ['打人','杀人','伤人','持刀','攻击','暴力','持械'],
    '3.5': ['报复','报仇','杀人计划','干掉','弄死','同归于尽'],
}

def cut_ws(t):
    return ' '.join(w for w in jieba.cut(t) if w.strip() and w not in STOP_WORDS)

def cln(t):
    return re.sub(r'\s+', '', re.sub(r'[^一-鿿\w]', '', t))

def bld(item):
    p = [item.get('question_title',''), item.get('question_content','')]
    for a in item.get('answers',[]):
        for d in a.get('dialogs',[]): p.append(d.get('content',''))
    return cut_ws(cln(' '.join(p)))

def get_full_text(item):
    t = item.get('question_title','') + ' ' + item.get('question_content','')
    for a in item.get('answers',[]):
        for d in a.get('dialogs',[]): t += ' ' + d.get('content','')
    return t

def detect_s3(text):
    for lbl, kws in S3_KW.items():
        if any(kw in text for kw in kws):
            return lbl
    return None

DATA = Path('data')
OUT = Path('data/人工标注')

print("=" * 60)
print("两阶段分类 + S3关键词兜底")
print("=" * 60)

# 加载训练数据
with open(OUT / 'No-01_待标注_3000_已标注.json', encoding='utf-8') as f:
    seed = json.load(f)
base_map = {item['question_id']: item['labels']['label'] for item in seed}

annotated = sorted(glob.glob(str(OUT / '*_已标注.json')))
annotated = [f for f in annotated if '3000_已标注' not in f]
all_corrections = {}
for fpath in annotated:
    with open(fpath, encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        qid = item['question_id']
        nl = item['labels']['label']
        if nl and nl != base_map.get(qid, ''):
            all_corrections[qid] = nl

merged = dict(base_map)
merged.update(all_corrections)
print(f"\n训练数据: {len(merged)}条, {len(set(merged.values()))}/31类")

with open(DATA / 'No-01.json', encoding='utf-8') as f:
    full_data = json.load(f)

# 构建训练集
train_texts, train_labels, train_weights = [], [], []
for item in full_data:
    qid = item['question_id']
    if qid in merged:
        train_texts.append(bld(item))
        train_labels.append(merged[qid])
        train_weights.append(5.0 if qid in all_corrections else 1.0)

train_s_level = [l[0] for l in train_labels]
print(f"S层级分布: {Counter(train_s_level)}")

# ===== Stage 1 =====
print("\n" + "-"*40)
print("Stage 1: S1/S2/S3 层级分类")
print("-"*40)

v1 = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b',
    ngram_range=(1,3), max_features=10000, min_df=1, max_df=0.9, sublinear_tf=True)
X1 = v1.fit_transform(train_texts)
sw = np.array(train_weights)

X_tr1, X_te1, y_tr1, y_te1 = train_test_split(
    X1, train_s_level, test_size=0.2, random_state=SEED, stratify=train_s_level)
sw_tr1, sw_te1 = train_test_split(
    sw, test_size=0.2, random_state=SEED, stratify=train_s_level)

c1 = LogisticRegression(max_iter=3000, C=1.0, random_state=SEED, class_weight='balanced')
c1.fit(X_tr1, y_tr1, sample_weight=sw_tr1)
p1 = c1.predict(X_te1)
acc1 = accuracy_score(y_te1, p1)
print(f"准确率: {acc1:.4f} ({len(y_te1)}条)")

cm = confusion_matrix(y_te1, p1, labels=['1','2','3'])
print(f"{'':>8} {'S1':>6} {'S2':>6} {'S3':>6}")
for i, lbl in enumerate(['S1','S2','S3']):
    print(f"  {lbl:>6}" + ''.join(f'{cm[i,j]:6d}' for j in range(3)))

# ===== Stage 2 =====
print("\n" + "-"*40)
print("Stage 2: 子类分类器")
print("-"*40)

classifiers = {}
vecs = {}
for level, name in [('1','S1'), ('2','S2'), ('3','S3')]:
    mask = [l == level for l in train_s_level]
    sub_texts = [t for t, m in zip(train_texts, mask) if m]
    sub_labels = [l for l, m in zip(train_labels, mask) if m]
    sub_weights = [w for w, m in zip(train_weights, mask) if m]

    if len(set(sub_labels)) < 2:
        print(f"  {name}: 跳过 (仅{len(set(sub_labels))}类)")
        classifiers[level] = None
        vecs[level] = None
        continue

    v = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b',
        ngram_range=(1,3), max_features=5000, min_df=1, max_df=0.9, sublinear_tf=True)
    X = v.fit_transform(sub_texts)
    sw_sub = np.array(sub_weights)

    try:
        X_tr, X_te, y_tr, y_te, sw_tr2, sw_te2 = train_test_split(
            X, sub_labels, sw_sub, test_size=0.2, random_state=SEED, stratify=sub_labels)
    except:
        X_tr, X_te, y_tr, y_te, sw_tr2, sw_te2 = train_test_split(
            X, sub_labels, sw_sub, test_size=0.2, random_state=SEED)

    c2 = LogisticRegression(max_iter=3000, C=1.0, random_state=SEED, class_weight='balanced')
    c2.fit(X_tr, y_tr, sample_weight=sw_tr2)
    p2 = c2.predict(X_te)
    acc2 = accuracy_score(y_te, p2)

    classifiers[level] = c2
    vecs[level] = v
    print(f"  {name} ({len(set(sub_labels))}类): {acc2:.4f} ({len(y_te)}条)")

# ===== 全量预测 =====
print("\n" + "-"*40)
print("全量预测 No-01")
print("-"*40)

X1_full = v1.transform([bld(item) for item in full_data])
s_pred = c1.predict(X1_full)

final_labels = []
for i, item in enumerate(full_data):
    level = s_pred[i]

    if classifiers.get(level) and vecs.get(level):
        x = vecs[level].transform([bld(item)])
        sub = classifiers[level].predict(x)[0]
    else:
        sub = level + '.17' if level == '1' else level + '.9'

    # S3兜底
    s3 = detect_s3(get_full_text(item))
    if s3:
        sub = s3

    final_labels.append(sub)

d = Counter(final_labels)
s1 = sum(v for k,v in d.items() if k.startswith('1.'))
s2 = sum(v for k,v in d.items() if k.startswith('2.'))
s3 = sum(v for k,v in d.items() if k.startswith('3.'))
s3c = sorted([k for k in d if k.startswith('3.')])

with open(OUT / 'No-01_半监督_全量标注.json', encoding='utf-8') as f:
    old_data = json.load(f)
old_preds = [item['labels']['label'] for item in old_data]
changes = sum(1 for i in range(8366) if old_preds[i] != final_labels[i])

# 逐类对比
print(f"\nS1={s1}({s1/8366*100:.1f}%) S2={s2}({s2/8366*100:.1f}%) S3={s3}({s3/8366*100:.1f}%)")
print(f"类: {len(d)}/31 | S3子类: {len(s3c)}/5 {s3c}")
print(f"vs半监督: {changes}条改变")

print(f"\n逐类对比 (半监督 -> 两阶段):")
all_lbls = sorted(set(list(Counter(old_preds).keys()) + list(d.keys())))
for lbl in all_lbls:
    o = Counter(old_preds).get(lbl, 0)
    n = d.get(lbl, 0)
    if abs(n - o) > 50:
        print(f"  {lbl}: {o} -> {n} ({'+' if n>o else ''}{n-o})")

# 保存模型（先保存，避免后续代码报错丢失）
joblib.dump(c1, OUT / 'stage1_model.pkl')
joblib.dump(v1, OUT / 'stage1_vec.pkl')
for lvl in ['1','2','3']:
    if classifiers.get(lvl):
        joblib.dump(classifiers[lvl], OUT / f'stage2_{lvl}_model.pkl')
        joblib.dump(vecs[lvl], OUT / f'stage2_{lvl}_vec.pkl')
print("模型保存完成")

# 输出
for i, item in enumerate(full_data):
    item['labels'] = {'label': final_labels[i]}

out_path = OUT / 'No-01_两阶段_最终标注.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(full_data, f, ensure_ascii=False, indent=2)
print(f"\n输出: {out_path}")
