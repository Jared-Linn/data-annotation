#!/usr/bin/env python3
"""微调实验: 不同特征+模型组合"""
import json, re, glob
from pathlib import Path
from collections import Counter
import numpy as np, jieba
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

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

# 加载数据
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

with open(DATA / 'No-01.json', encoding='utf-8') as f:
    full_data = json.load(f)

train_texts, train_labels, train_weights = [], [], []
for item in full_data:
    qid = item['question_id']
    if qid in merged:
        train_texts.append(bld(item))
        train_labels.append(merged[qid])
        train_weights.append(5.0 if qid in all_corrections else 1.0)

train_s_level = [l[0] for l in train_labels]
sw = np.array(train_weights)

print("微调实验: S层级分类\n")

results = []

def eval_model(name, vec, clf, X, y, sw):
    try:
        X_tr, X_te, y_tr, y_te, sw_tr, sw_te = train_test_split(
            X, y, sw, test_size=0.2, random_state=SEED, stratify=y)
    except:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=SEED)
        sw_tr = None
    clf.fit(X_tr, y_tr, sample_weight=sw_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))
    results.append((name, acc, clf, vec))
    print(f"  {name}: {acc:.4f}")
    return clf, vec

# 实验1: word unigram
v1 = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b',
    ngram_range=(1,1), max_features=10000, min_df=1, max_df=0.9, sublinear_tf=True)
X1 = v1.fit_transform(train_texts)
eval_model('LR+word1gram', v1, LogisticRegression(max_iter=3000, C=1.0, random_state=SEED, class_weight='balanced'), X1, train_s_level, sw)
eval_model('NB+word1gram', v1, MultinomialNB(alpha=0.1), X1, train_s_level, sw)

# 实验2: word 1-3gram (当前)
v3 = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b',
    ngram_range=(1,3), max_features=10000, min_df=1, max_df=0.9, sublinear_tf=True)
X3 = v3.fit_transform(train_texts)
eval_model('LR+word1-3gram', v3, LogisticRegression(max_iter=3000, C=1.0, random_state=SEED, class_weight='balanced'), X3, train_s_level, sw)
eval_model('NB+word1-3gram', v3, MultinomialNB(alpha=0.1), X3, train_s_level, sw)

# 实验3: char 2-4gram
vc = TfidfVectorizer(analyzer='char', ngram_range=(2,4), max_features=10000, min_df=1, max_df=0.9, sublinear_tf=True)
Xc = vc.fit_transform(train_texts)
eval_model('LR+char2-4gram', vc, LogisticRegression(max_iter=3000, C=1.0, random_state=SEED, class_weight='balanced'), Xc, train_s_level, sw)
eval_model('NB+char2-4gram', vc, MultinomialNB(alpha=0.1), Xc, train_s_level, sw)

# 实验4: word 1-3gram + char 2-4gram 拼接
Xwc = hstack([X3, Xc])
# 自定义 eval 因为需要特殊 vectorizer
X_tr, X_te, y_tr, y_te, sw_tr, sw_te = train_test_split(
    Xwc, train_s_level, sw, test_size=0.2, random_state=SEED, stratify=train_s_level)
clf_wc = LogisticRegression(max_iter=3000, C=1.0, random_state=SEED, class_weight='balanced')
clf_wc.fit(X_tr, y_tr, sample_weight=sw_tr)
acc_wc = accuracy_score(y_te, clf_wc.predict(X_te))
results.append(('LR+word+char', acc_wc, clf_wc, ('hybrid', v3, vc)))
print(f"  LR+word+char: {acc_wc:.4f}")

# 结果排名
print(f"\n{'='*50}")
print(f"排名:")
for i, (name, acc, _, _) in enumerate(sorted(results, key=lambda x: -x[1])):
    print(f"  #{i+1} {name}: {acc:.4f}")

best = max(results, key=lambda x: x[1])
print(f"\n最佳: {best[0]} = {best[1]:.4f}")

# 用最佳方案 + 关键词兜底重建全量
print(f"\n{'='*50}")
print(f"用最佳方案预测全量")
print(f"{'='*50}")

S3_KW = {
    '3.1': ['正在自杀','跳楼','上吊','割腕','服药自杀','在自杀'],
    '3.2': ['想自杀','自杀计划','准备死','安排后事','写遗书','计划自杀'],
    '3.3': ['自残','划手','割手','烫自己','伤害身体','自伤','划伤'],
    '3.4': ['打人','杀人','伤人','持刀','攻击','暴力','持械'],
    '3.5': ['报复','报仇','杀人计划','干掉','弄死','同归于尽'],
}

best_name, best_acc, best_clf, best_vec = best

for tgt in ['No-01','No-02','No-03']:
    with open(DATA / f'{tgt}.json', encoding='utf-8') as f:
        items = json.load(f)

    if best_name == 'LR+word+char':
        _, v3_best, vc_best = best_vec
        X_full = hstack([v3_best.transform([bld(it) for it in items]),
                         vc_best.transform([bld(it) for it in items])])
    else:
        X_full = best_vec.transform([bld(it) for it in items])

    s_pred = best_clf.predict(X_full)

    # 关键词补稀有类
    for i, item in enumerate(items):
        txt = item.get('question_title','') + ' ' + item.get('question_content','')
        for a in item.get('answers',[]):
            for d in a.get('dialogs',[]): txt += ' ' + d.get('content','')

        # S3关键词优先
        s3 = None
        for lbl, kws in S3_KW.items():
            if any(kw in txt for kw in kws):
                s3 = lbl; break
        if s3:
            s_pred[i] = s3

    d = Counter(s_pred)
    s1 = sum(v for k,v in d.items() if k.startswith('1.'))
    s2 = sum(v for k,v in d.items() if k.startswith('2.'))
    s3 = sum(v for k,v in d.items() if k.startswith('3.'))
    s3c = sorted([k for k in d if k.startswith('3.')])
    n = len(items)

    print(f"\n{tgt}: S1={s1/n*100:.1f}% S2={s2/n*100:.1f}% S3={s3/n*100:.1f}% 类={len(d)}/31 S3子类={len(s3c)}/5")

    for i, item in enumerate(items):
        item['labels'] = {'label': s_pred[i]}
    op = OUT / f'{tgt}_微调_最终标注.json'
    with open(op, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

print(f"\n输出: {OUT}/")
