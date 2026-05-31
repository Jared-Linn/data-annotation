#!/usr/bin/env python3
"""最终版: LR+word1gram + Stage2子类 + S3兜底 + 稀有类补全"""
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
STOP_WORDS = set("""
的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你
会 着 没有 看 好 自己 这 他 她 它 们 那 么 什么 怎么 因为 所以
""".split())

S3_KW = {
    '3.1': ['正在自杀','跳楼','上吊','割腕','服药自杀','在自杀'],
    '3.2': ['想自杀','自杀计划','准备死','安排后事','写遗书','计划自杀'],
    '3.3': ['自残','划手','割手','烫自己','伤害身体','自伤','划伤'],
    '3.4': ['打人','杀人','伤人','持刀','攻击','暴力','持械'],
    '3.5': ['报复','报仇','杀人计划','干掉','弄死','同归于尽'],
}

# 稀缺类关键词兜底（Stage 2 覆盖不到的稀有类）
RARE_KW = {
    '1.4': ['喝酒','吸烟','抽烟','棋牌','小酌','偶尔喝酒'],
    '1.14': ['青春期','发育','发育焦虑','青春困惑','变声','长高','月经','遗精'],
    '2.3': ['躁郁','双相','情绪两极','亢奋','精力旺盛','不睡觉','思维跳跃','冲动消费'],
    '2.5': ['恐慌','濒死','窒息','惊恐发作','panic','急性焦虑','突然心悸','濒死感'],
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
def get_full(item):
    t = item.get('question_title','') + ' ' + item.get('question_content','')
    for a in item.get('answers',[]):
        for d in a.get('dialogs',[]): t += ' ' + d.get('content','')
    return t

DATA = Path('data')
OUT = Path('data/人工标注')
OUT.mkdir(parents=True, exist_ok=True)

# 加载数据
with open(OUT / 'No-01_待标注_3000_已标注.json', encoding='utf-8') as f:
    seed = json.load(f)
base_map = {item['question_id']: item['labels']['label'] for item in seed}

annotated = sorted(glob.glob(str(OUT / '*_已标注.json')))
annotated = [f for f in annotated if '3000_已标注' not in f]
all_corrections = {}
for fpath in annotated:
    with open(fpath, encoding='utf-8') as f:
        d = json.load(f)
    for item in d:
        qid = item['question_id']
        nl = item['labels']['label']
        if nl and nl != base_map.get(qid, ''):
            all_corrections[qid] = nl

merged = dict(base_map)
merged.update(all_corrections)
print(f"训练数据: {len(merged)}条, {len(set(merged.values()))}类")

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
n_total = len(train_texts)
n_corr = sum(1 for w in train_weights if w > 1)
print(f"训练: {n_total}条 ({n_corr}修正 w=5)")

# ===== Stage 1: LR + word 1gram =====
print("\nStage 1: LR+word1gram")
v1 = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b',
    ngram_range=(1,1), max_features=10000, min_df=1, max_df=0.9, sublinear_tf=True)
X1 = v1.fit_transform(train_texts)
X_tr, X_te, y_tr, y_te, sw_tr, sw_te = train_test_split(X1, train_s_level, sw, test_size=0.2, random_state=SEED, stratify=train_s_level)
c1 = LogisticRegression(max_iter=3000, C=1.0, random_state=SEED, class_weight='balanced')
c1.fit(X_tr, y_tr, sample_weight=sw_tr)
p1 = c1.predict(X_te)
acc1 = accuracy_score(y_te, p1)
print(f"  准确率: {acc1:.4f} ({len(y_te)}条)")

cm = confusion_matrix(y_te, p1, labels=['1','2','3'])
print(f"{'':>8} {'S1':>6} {'S2':>6} {'S3':>6}")
for i, lbl in enumerate(['S1','S2','S3']):
    print(f"  {lbl:>6}" + ''.join(f'{cm[i,j]:6d}' for j in range(3)))

# ===== Stage 2 =====
print("\nStage 2: 子类分类器")
classifiers, vecs = {}, {}
for level, name in [('1','S1'),('2','S2'),('3','S3')]:
    mask = [l == level for l in train_s_level]
    sub_t = [t for t,m in zip(train_texts,mask) if m]
    sub_l = [l for l,m in zip(train_labels,mask) if m]
    sub_w = [w for w,m in zip(train_weights,mask) if m]
    if len(set(sub_l)) < 2:
        classifiers[level] = None; vecs[level] = None
        print(f"  {name}: 跳过"); continue
    v = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b',
        ngram_range=(1,1), max_features=5000, min_df=1, max_df=0.9, sublinear_tf=True)
    X = v.fit_transform(sub_t)
    sw_sub = np.array(sub_w)
    try:
        X_tr, X_te2, y_tr, y_te2 = train_test_split(X, sub_l, test_size=0.2, random_state=SEED, stratify=sub_l)
        sw_tr2,_ = train_test_split(sw_sub, test_size=0.2, random_state=SEED, stratify=sub_l)
    except:
        X_tr, X_te2, y_tr, y_te2 = train_test_split(X, sub_l, test_size=0.2, random_state=SEED)
        sw_tr2,_ = train_test_split(sw_sub, test_size=0.2, random_state=SEED)
    c = LogisticRegression(max_iter=3000, C=1.0, random_state=SEED, class_weight='balanced')
    c.fit(X_tr, y_tr, sample_weight=sw_tr2)
    acc2 = accuracy_score(y_te2, c.predict(X_te2))
    classifiers[level] = c; vecs[level] = v
    print(f"  {name} ({len(set(sub_l))}类): {acc2:.4f}")

# 保存模型
joblib.dump(c1, OUT / 'final_stage1.pkl')
joblib.dump(v1, OUT / 'final_stage1_vec.pkl')
for lvl in ['1','2','3']:
    if classifiers.get(lvl):
        joblib.dump(classifiers[lvl], OUT / f'final_stage2_{lvl}.pkl')
        joblib.dump(vecs[lvl], OUT / f'final_stage2_{lvl}_vec.pkl')
print("\n模型已保存")

# ===== 全量预测 =====
print("\n" + "=" * 55)
print("全量预测 No-01/02/03")
print("=" * 55)

for tgt in ['No-01','No-02','No-03']:
    with open(DATA / f'{tgt}.json', encoding='utf-8') as f:
        items = json.load(f)

    X_full = v1.transform([bld(it) for it in items])
    s_pred = c1.predict(X_full)
    final = []

    for i, it in enumerate(items):
        full = get_full(it)
        level = s_pred[i]

        # 1) S3关键词优先
        s3 = None
        for lbl, kws in S3_KW.items():
            if any(kw in full for kw in kws):
                s3 = lbl; break
        if s3:
            final.append(s3)
            continue

        # 2) 稀有类关键词补全
        for lbl, kws in RARE_KW.items():
            if any(kw in full for kw in kws):
                if lbl[0] == level:  # 同层级才补
                    final.append(lbl)
                    break
        else:
            # 3) Stage 2 预测
            if classifiers.get(level) and vecs.get(level):
                x = vecs[level].transform([bld(it)])
                final.append(classifiers[level].predict(x)[0])
            else:
                final.append(level + '.17' if level == '1' else level + '.9')

    # 统计
    d = Counter(final)
    s1 = sum(v for k,v in d.items() if k.startswith('1.'))
    s2 = sum(v for k,v in d.items() if k.startswith('2.'))
    s3 = sum(v for k,v in d.items() if k.startswith('3.'))
    s3c = sorted([k for k in d if k.startswith('3.')])
    n = len(items)

    # 和半监督基线对比
    old_path = OUT / f'{tgt}_半监督_全量标注.json'
    changes = 0
    if old_path.exists():
        with open(old_path, encoding='utf-8') as f:
            old = [x['labels']['label'] for x in json.load(f)]
        changes = sum(1 for i in range(n) if old[i] != final[i])

    print(f"\n{tgt}: {n}条 | {len(d)}/31类 | S3子类{len(s3c)}/5 {s3c}")
    print(f"  S1={s1}({s1/n*100:.1f}%) S2={s2}({s2/n*100:.1f}%) S3={s3}({s3/n*100:.1f}%)" + (f" | vs基线 {changes}条改变" if changes else ""))

    # 展示缺失类
    missing = [f'{i}.{j}' for i in ['1','2','3'] for j in range(1,18) if i=='1' and f'1.{j}' not in d] + \
              [f'{i}.{j}' for i in ['2','3'] for j in range(1,10) if i=='2' and f'2.{j}' not in d] + \
              [f'{i}.{j}' for i in ['3'] for j in range(1,6) if i=='3' and f'3.{j}' not in d]
    if missing:
        print(f"  缺失类: {missing}")

    for i, it in enumerate(items):
        it['labels'] = {'label': final[i]}
    op = OUT / f'{tgt}_最终版.json'
    with open(op, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"  输出: {op}")

print(f"\n全部完成! 文件在 {OUT}/")
