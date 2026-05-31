#!/usr/bin/env python3
"""
半监督自训练 v3: 轻量版 — 每轮每类 top-30，最多3轮
"""
import json, re, os, random, time
from pathlib import Path
from collections import Counter
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

with open('data/stopwords.txt', encoding='utf-8') as f:
    STOP_WORDS = set(line.strip() for line in f if line.strip())

def cut_ws(text):
    return ' '.join(w for w in jieba.cut(text) if w.strip() and w not in STOP_WORDS)

def cln(text):
    return re.sub(r'\s+', '', re.sub(r'[^一-鿿\w]', '', text))

def bld(item):
    p = [item.get('question_title',''), item.get('question_content','')]
    for a in item.get('answers',[]):
        for d in a.get('dialogs',[]): p.append(d.get('content',''))
    return cut_ws(cln(' '.join(p)))

DATA = Path('data')
OUT = Path('data/人工标注')
OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()

# 加载种子
with open(OUT / 'No-01_待标注_3000_已标注.json', encoding='utf-8') as f:
    seed = json.load(f)
tx = [bld(item) for item in seed]
ly = [item['labels']['label'] for item in seed]
src = ['human'] * len(tx)
print(f"种子: {len(tx)}条, {len(set(ly))}类")

# 加载未标注
unlab = {}
for n in ['No-02','No-03']:
    with open(DATA / f'{n}.json', encoding='utf-8') as f:
        unlab[n] = json.load(f)
    print(f"{n}: {len(unlab[n])}条未标注")

# 自训练 - 每轮每类 top-K
for rnd, topk in enumerate([50, 30, 20], 1):
    print(f"\n--- 第{rnd}轮 topk={topk} ---")

    v = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b',
        ngram_range=(1,3), max_features=10000, min_df=1, max_df=0.9, sublinear_tf=True)
    X = v.fit_transform(tx)
    c = LogisticRegression(max_iter=3000, C=1.0, random_state=SEED, class_weight='balanced')
    c.fit(X, ly)

    new = 0
    for tgt in ['No-02','No-03']:
        its = unlab[tgt]
        Xu = v.transform([bld(it) for it in its])
        pr = c.predict(Xu)
        pb = c.predict_proba(Xu).max(axis=1)

        # 按类选 top-k
        cand = {}
        for i in range(len(its)):
            cand.setdefault(pr[i], []).append((i, pb[i]))

        added = 0
        for lbl in c.classes_:
            items = sorted(cand.get(lbl,[]), key=lambda x:-x[1])[:topk]
            for idx,_ in items:
                tx.append(bld(its[idx]))
                ly.append(lbl)
                src.append(f'pseudo_{tgt}')
                added += 1
                new += 1
        print(f"  {tgt}: +{added}")

    if new == 0:
        print("  收敛")
        break

print(f"\n最终: {len(tx)}条 (人工{src.count('human')} + 伪{src.count('pseudo_No-02')+src.count('pseudo_No-03')})")

# 最终模型
print("\n全量预测 No-01/02/03 ...")
vf = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b',
    ngram_range=(1,3), max_features=10000, min_df=1, max_df=0.9, sublinear_tf=True)
Xf = vf.fit_transform(tx)
cf = LogisticRegression(max_iter=3000, C=1.0, random_state=SEED, class_weight='balanced')
cf.fit(Xf, ly)

for tgt in ['No-01','No-02','No-03']:
    with open(DATA / f'{tgt}.json', encoding='utf-8') as f:
        its = json.load(f)
    Xt = vf.transform([bld(it) for it in its])
    pred = cf.predict(Xt)
    for i, it in enumerate(its):
        it['labels'] = {'label': pred[i]}

    op = OUT / f'{tgt}_半监督_全量标注.json'
    with open(op, 'w', encoding='utf-8') as f:
        json.dump(its, f, ensure_ascii=False, indent=2)

    d = Counter(pred)
    s1 = sum(v for k,v in d.items() if k.startswith('1.'))
    s2 = sum(v for k,v in d.items() if k.startswith('2.'))
    s3 = sum(v for k,v in d.items() if k.startswith('3.'))
    n = len(its)
    s3c = sorted([k for k in d if k.startswith('3.')])
    print(f"{tgt}: {n}条 {len(d)}/31类 S3{len(s3c)}/5 {s3c}")
    print(f"  S1={s1/n*100:.1f}% S2={s2/n*100:.1f}% S3={s3/n*100:.1f}%")

print(f"\n耗时: {(time.time()-t0)/60:.1f}分钟")
print(f"输出: {OUT}/")
