#!/usr/bin/env python3
"""补全: 多模型对比(4) + 分类报告(6) + dialog tags(加分项)"""
import json, re, glob
from pathlib import Path
from collections import Counter
import numpy as np, jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

SEED = 42
STOP_WORDS = set("""
的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你
会 着 没有 看 好 自己 这 他 她 它 们 那 么 什么 怎么 因为 所以
""".split())

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
OUT.mkdir(parents=True, exist_ok=True)

# 加载数据
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
            if nl and nl != base_map.get(qid, ''): corr[qid] = nl
merged = dict(base_map); merged.update(corr)

with open(DATA / 'No-01.json', encoding='utf-8') as f:
    full_data = json.load(f)

train_texts, train_labels = [], []
for item in full_data:
    qid = item['question_id']
    if qid in merged:
        train_texts.append(bld(item))
        train_labels.append(merged[qid])

train_s = [l[0] for l in train_labels]

# ============================================================
# PART 1 & 2: 多模型对比 + 分类报告
# ============================================================
print("=" * 70)
print("PART 1 & 2: 多模型对比 + 分类报告")
print("=" * 70)

vec = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b',
    ngram_range=(1,1), max_features=10000, min_df=1, max_df=0.9, sublinear_tf=True)
X = vec.fit_transform(train_texts)
X_tr, X_te, y_tr, y_te = train_test_split(X, train_s, test_size=0.2, random_state=SEED, stratify=train_s)

models = [
    ('LR (逻辑回归)', LogisticRegression(max_iter=3000, C=1.0, random_state=SEED, class_weight='balanced')),
    ('NB (朴素贝叶斯)', MultinomialNB(alpha=0.1)),
    ('SVM (线性核)', LinearSVC(max_iter=3000, C=1.0, random_state=SEED, class_weight='balanced')),
]

results = []
for name, clf in models:
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    acc = accuracy_score(y_te, pred)
    cm = confusion_matrix(y_te, pred, labels=['1','2','3'])
    results.append((name, acc, pred, cm))
    print(f"\n{'─'*50}")
    print(f"{name}  | 准确率: {acc:.4f}")
    print(f"{'─'*50}")
    print(f"混淆矩阵 (S层级):")
    print(f"{'':>8} {'S1':>6} {'S2':>6} {'S3':>6}")
    for i, lbl in enumerate(['S1','S2','S3']):
        print(f"  {lbl:>6}" + ''.join(f'{cm[i,j]:6d}' for j in range(3)))
    print(f"\n分类报告:")
    print(classification_report(y_te, pred, target_names=['S1','S2','S3'], zero_division=0))

# ============================================================
# Stage 2 分类报告
# ============================================================
print("\n" + "=" * 70)
print("Stage 2 子类分类报告")
print("=" * 70)

# 用最佳模型(LR)做Stage 2详细报告
for level, name in [('1','S1'),('2','S2'),('3','S3')]:
    mask = [l == level for l in train_s]
    sub_t = [t for t,m in zip(train_texts,mask) if m]
    sub_l = [l for l,m in zip(train_labels,mask) if m]
    if len(set(sub_l)) < 2: continue
    v = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b',
        ngram_range=(1,1), max_features=5000, min_df=1, max_df=0.9, sublinear_tf=True)
    Xs = v.fit_transform(sub_t)
    try:
        X_st, X_sv, y_st, y_sv = train_test_split(Xs, sub_l, test_size=0.2, random_state=SEED, stratify=sub_l)
    except:
        X_st, X_sv, y_st, y_sv = train_test_split(Xs, sub_l, test_size=0.2, random_state=SEED)
    clf = LogisticRegression(max_iter=3000, C=1.0, random_state=SEED, class_weight='balanced')
    clf.fit(X_st, y_st)
    pred = clf.predict(X_sv)
    acc = accuracy_score(y_sv, pred)
    print(f"\n{name} ({len(set(sub_l))}类) 准确率: {acc:.4f}")
    print(classification_report(y_sv, pred, zero_division=0))

# ============================================================
# PART 3: Dialog Tags 标注
# ============================================================
print("\n" + "=" * 70)
print("PART 3: Dialog Tags 标注 (加分项)")
print("=" * 70)

# 关键词规则
KNOWLEDGE_KW = [
    '建议','方法','可以试试','原因是','心理学','认知','行为','情绪管理',
    '放松训练','深呼吸','冥想','正念','心理咨询','治疗','调理','改善',
    '调整','缓解','克服','面对','接受','理解','倾听','共情','支持',
    '尝试','练习','记录','反思','分析','思考','感受','觉察','意识到',
    '鼓励','肯定','认可','尊重','信任','安全感','边界','健康',
    '专业','评估','诊断','量表','症状','干预','疏导','沟通','表达',
    '探索','发现','改变','成长','学习','适应','规划','目标','行动',
    '欢迎点击头像','可以私信','预约','咨询师','咨询',
]
NEGATIVE_KW = [
    '你怎么','你应该','你不该','这不对','有病','严重','糟糕','真烦',
    '没救了','放弃吧','你不行','你错了','你太','你总是','你从来',
    '指责','批评','抱怨','埋怨','贬低','羞辱','嘲笑','讽刺',
    '冷漠','无视','不耐烦','嫌弃','讨厌','恶心','受不了',
]
MEANINGLESS_KW = [
    '你好','新年好','春节好','节日快乐','祝您','祝福','谢谢','感谢',
    '欢迎','嗯','哦','好的','是的','对','知道了','收到','明白',
    '送花','献上','拥抱','握手','微笑','图片',
]

def tag_dialog(content):
    """对单条 dialog 内容打 tags"""
    tags = []
    if any(kw in content for kw in KNOWLEDGE_KW):
        tags.append('knowledge')
    if any(kw in content for kw in NEGATIVE_KW):
        tags.append('negative')
    if any(kw in content for kw in MEANINGLESS_KW):
        tags.append('meaningless')
    if not tags:
        # 默认：有实质内容但不是专业知识的，归为知识类
        if len(content) > 10:
            tags.append('knowledge')
        else:
            tags.append('meaningless')
    return tags

tag_stats = Counter()
total_dialogs = 0

for tgt in ['No-01','No-02','No-03']:
    with open(DATA / f'{tgt}.json', encoding='utf-8') as f:
        items = json.load(f)

    for item in items:
        for ans in item.get('answers', []):
            for d in ans.get('dialogs', []):
                content = d.get('content', '')
                tags = tag_dialog(content)
                d['tags'] = tags
                tag_stats.update(tags)
                total_dialogs += 1

    # 输出带tags的版本
    op = OUT / f'{tgt}_带标签.json'
    with open(op, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

print(f"总标注对话条数: {total_dialogs}")
print(f"Tags分布:")
for tag, cnt in sorted(tag_stats.items(), key=lambda x: -x[1]):
    print(f"  {tag}: {cnt} ({cnt/total_dialogs*100:.1f}%)")

# 展示一条示例
with open(DATA / 'No-01.json', encoding='utf-8') as f:
    sample = json.load(f)[0]
print(f"\n示例 (question_id={sample['question_id']}):")
for ans in sample.get('answers', [])[:1]:
    for d in ans.get('dialogs', [])[:2]:
        print(f"  content: {d['content'][:60]}...")
        print(f"  tags: {d.get('tags', [])}")

print(f"\n带tags文件已输出到 {OUT}/")
print("完成!")
