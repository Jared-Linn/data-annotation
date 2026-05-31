#!/usr/bin/env python3
"""
半监督自训练 v2: 按类选 top-K 高置信预测（不用绝对值阈值）
"""
import json, re, os, random
from pathlib import Path
from collections import Counter
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

STOP_WORDS = set("""
的 了 在 是 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你
会 着 没有 看 好 自己 这 他 她 它 们 那 么 什么 怎么 因为 所以
如果 但 是 但 可以 还 为 又 能 而 或 之 与 及 等 被 把 让 向
从 对 将 用 以 比 按 照 跟 和 同 被 把 让 给 为 所 得 地 着
过 了 呢 吗 啊 呀 吧 么 哦 嗯 哈 呵 嗨 喂 啦 嘛 哪 咋 喔 呗
这 那 哪 谁 怎样 哪儿 那里 这里 那些 这些 什么 怎么 如何 为何
个 只 些 点 样 种 回 次 遍 下 里 外 前 后 左 右 东 西 南 北
已 已经 曾经 刚 刚刚 正 正在 将 将要 就 便 才 再 又 也 还
很 非常 十分 太 最 极 更 越 稍 略 几 多么 尤其 甚至 几乎
不 没 没有 未 别 勿 休 不要 不用 甭 看 稍 无须 未曾 尚未
别 难道 究竟 到底 何必 何苦 何不 何况 况且 而且 并且 或者
还是 但是 然而 不过 只是 可是 却 则 然 而 虽然 尽管 即使
因为 所以 因此 于是 从而 以致 以至于 由于 既然 鉴于 为了
如果 假如 假若 假使 倘若 要是 若 若非 不然 否则 要不 要不是
只要 除非 无论 不论 不管 任凭 哪怕 纵使 就算 就是 即使
除了 此外 另外 关于 对于 至于 针对 通过 根据 凭借 依照 按照
遵照 本着 经过 沿着 顺着 朝着 往 向 朝 从 自 自从 打 由 到
在 于 当 以 用 拿 把 将 被 叫 让 给 替 为 对 跟 同 比 与
和 及 以及 并 并且 而 而且 或 或者 还是 要么 不只 不仅 不但
""".split())

def cut_with_stop(text):
    return ' '.join(w for w in jieba.cut(text) if w.strip() and w not in STOP_WORDS)

def clean(text):
    return re.sub(r'\s+', '', re.sub(r'[^一-鿿\w]', '', text))

def build_text(item):
    parts = [item.get('question_title', ''), item.get('question_content', '')]
    for ans in item.get('answers', []):
        for d in ans.get('dialogs', []):
            parts.append(d.get('content', ''))
    return cut_with_stop(clean(' '.join(parts)))

DATA_DIR = Path('data')
OUT_DIR = Path('data/人工标注')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ========== 加载种子 ==========
print("=" * 70)
print("半监督自训练 v2: 按类选 top-K")
print("=" * 70)

with open(OUT_DIR / 'No-01_待标注_3000_已标注.json', encoding='utf-8') as f:
    seed_raw = json.load(f)

seed_texts = [build_text(item) for item in seed_raw]
seed_labels = [item['labels']['label'] for item in seed_raw]
print(f"种子: {len(seed_raw)}条, {len(set(seed_labels))}类")

# ========== 加载未标注 ==========
targets = ['No-02', 'No-03']
unlabeled = {}
for t in targets:
    with open(DATA_DIR / f'{t}.json', encoding='utf-8') as f:
        unlabeled[t] = json.load(f)
    print(f"{t}: {len(unlabeled[t])}条未标注")

# ========== 自训练 ==========
# 训练集
train_texts = list(seed_texts)
train_labels = list(seed_labels)
train_source = ['human'] * len(seed_texts)

# 每轮每类新增的伪标签数（逐轮递减）
TOP_K_PER_CLASS = [300, 200, 150, 100, 50]

for rnd, top_k in enumerate(TOP_K_PER_CLASS, 1):
    print(f"\n--- 第{rnd}轮 (top-k={top_k}) 训练集={len(train_texts)}条 ---")

    vec = TfidfVectorizer(
        analyzer='word', token_pattern=r'(?u)\b\w+\b',
        ngram_range=(1, 3), max_features=10000,
        min_df=1, max_df=0.9, sublinear_tf=True,
    )
    X_train = vec.fit_transform(train_texts)
    clf = LogisticRegression(max_iter=3000, C=1.0, random_state=SEED, class_weight='balanced')
    clf.fit(X_train, train_labels)

    total_new = 0
    for tgt in targets:
        items = unlabeled[tgt]
        # 跳过已被选过的
        already_selected = set()
        for s in train_source:
            if tgt in s:
                already_selected.add(id(s))

        X_unlab = vec.transform([build_text(item) for item in items])
        preds = clf.predict(X_unlab)
        probs = clf.predict_proba(X_unlab)
        max_probs = probs.max(axis=1)
        classes = clf.classes_

        class_candidates = {c: [] for c in classes}
        for i in range(len(items)):
            class_candidates[preds[i]].append((i, max_probs[i]))

        added = 0
        for c in classes:
            candidates = sorted(class_candidates.get(c, []), key=lambda x: -x[1])
            take = min(len(candidates), top_k)
            for idx, conf in candidates[:take]:
                train_texts.append(build_text(items[idx]))
                train_labels.append(c)
                train_source.append(f'pseudo_{tgt}_r{rnd}')
                added += 1
                total_new += 1
        print(f"  {tgt}: +{added}条")

    if total_new == 0:
        print("  无新增，收敛")
        break

print(f"\n最终训练集: {len(train_texts)}条 (人工{sum(1 for s in train_source if s=='human')} + 伪{sum(1 for s in train_source if 'pseudo' in s)})")

# ========== 最终模型 ==========
print("\n" + "=" * 70)
print("最终模型: 全量预测 No-01/02/03")
print("=" * 70)

final_vec = TfidfVectorizer(
    analyzer='word', token_pattern=r'(?u)\b\w+\b',
    ngram_range=(1, 3), max_features=10000,
    min_df=1, max_df=0.9, sublinear_tf=True,
)
X_final = final_vec.fit_transform(train_texts)
final_clf = LogisticRegression(max_iter=3000, C=1.0, random_state=SEED, class_weight='balanced')
final_clf.fit(X_final, train_labels)

for tgt in ['No-01', 'No-02', 'No-03']:
    with open(DATA_DIR / f'{tgt}.json', encoding='utf-8') as f:
        items = json.load(f)

    X = final_vec.transform([build_text(item) for item in items])
    preds = final_clf.predict(X)

    for i, item in enumerate(items):
        item['labels'] = {'label': preds[i]}

    out_path = OUT_DIR / f'{tgt}_半监督_全量标注.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    dist = Counter(preds)
    s1 = sum(v for k, v in dist.items() if k.startswith('1.'))
    s2 = sum(v for k, v in dist.items() if k.startswith('2.'))
    s3 = sum(v for k, v in dist.items() if k.startswith('3.'))
    n = len(items)
    s3_cls = sorted([k for k in dist if k.startswith('3.')])
    print(f"\n{tgt}: {n}条 | {len(dist)}/31类 | S3子类{len(s3_cls)}/5 {s3_cls}")
    print(f"  S1={s1}({s1/n*100:.1f}%) S2={s2}({s2/n*100:.1f}%) S3={s3}({s3/n*100:.1f}%)")

    # 与纯种子模型对比（从输出文件读取）
    if tgt == 'No-01':
        with open(OUT_DIR / 'No-01_待标注_3000_已标注.json', encoding='utf-8') as f:
            human = json.load(f)
        h_dist = Counter(item['labels']['label'] for item in human)
        print(f"\n  --- No-01 人工(3000) vs 半监督(8366) 分布 ---")
        print(f"  {'类目':>5} {'人工%':>6} {'半监督%':>8}")
        all_l = sorted(set(list(h_dist.keys()) + list(dist.keys())))
        for lbl in all_l:
            hp = h_dist.get(lbl, 0) / 3000 * 100
            sp = dist.get(lbl, 0) / 8366 * 100
            diff = sp - hp
            sign = '+' if diff > 0 else ''
            print(f"  {lbl:>4} {hp:5.1f}% {sp:7.1f}% ({sign}{diff:+.1f})")

print(f"\n输出目录: {OUT_DIR}/")
