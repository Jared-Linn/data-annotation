#!/usr/bin/env python3
"""
重训脚本：加载 web 工具修正后的 JSON → 加权重训 → 输出最终标注
用法:
  python retrain.py --corrected data/人工标注/No-01_待修正_已标注.json \
                    --output data/人工标注/No-01_最终标注.json \
                    --weight 15
"""
import json, re, argparse
from pathlib import Path
from collections import Counter
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

SEED = 42

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

def cut_ws(t):
    return ' '.join(w for w in jieba.cut(t) if w.strip() and w not in STOP_WORDS)

def cln(t):
    return re.sub(r'\s+', '', re.sub(r'[^一-鿿\w]', '', t))

def bld(item):
    p = [item.get('question_title',''), item.get('question_content','')]
    for a in item.get('answers',[]):
        for d in a.get('dialogs',[]): p.append(d.get('content',''))
    return cut_ws(cln(' '.join(p)))


def main():
    parser = argparse.ArgumentParser(description='加权重训：用 web 修正结果优化模型')
    parser.add_argument('--corrected', required=True, help='web 工具修正后的 JSON')
    parser.add_argument('--output', required=True, help='输出文件路径')
    parser.add_argument('--weight', type=float, default=15.0, help='修正样本权重 (默认 15)')
    parser.add_argument('--original', help='原始数据 JSON (如不指定，从 corrected 文件推断)')
    args = parser.parse_args()

    # 加载修正数据
    corrected_path = Path(args.corrected)
    with open(corrected_path, encoding='utf-8') as f:
        corrected_data = json.load(f)

    print(f"修正数据: {len(corrected_data)} 条")

    # 构建训练数据
    texts = [bld(item) for item in corrected_data]
    labels = [item['labels']['label'] for item in corrected_data]

    # 统计
    n_empty = sum(1 for l in labels if not l)
    if n_empty > 0:
        print(f"⚠ 警告: {n_empty} 条未标注，已跳过")
        # 过滤
        valid = [(t, l) for t, l in zip(texts, labels) if l]
        texts = [v[0] for v in valid]
        labels = [v[1] for v in valid]

    # 找修正过的样本（与原始预测对比）
    n_corrected = 0
    for item in corrected_data:
        old = item.get('_old_label')
        new = item['labels']['label']
        if old and new and old != new:
            n_corrected += 1

    print(f"标签已修改: {n_corrected} 条")
    print(f"有效训练: {len(texts)} 条, {len(set(labels))} 类")

    # 标签分布
    dist = Counter(labels)
    print("\n标签分布:")
    for lbl in sorted(dist):
        print(f"  {lbl}: {dist[lbl]}")

    # TF-IDF
    vec = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b',
        ngram_range=(1,3), max_features=10000, min_df=1, max_df=0.9, sublinear_tf=True)
    X = vec.fit_transform(texts)

    # 权重: 修正过的样本 weight, 其余 1
    sw = np.ones(len(labels))
    if '_old_label' in corrected_data[0]:
        for i, item in enumerate(corrected_data):
            if item.get('labels',{}).get('label') and item.get('_old_label') and \
               item['labels']['label'] != item['_old_label']:
                sw[i] = args.weight

    # 划分测试
    try:
        X_tr, X_te, y_tr, y_te, sw_tr, sw_te = train_test_split(
            X, labels, sw, test_size=0.2, random_state=SEED, stratify=labels)
        clf = LogisticRegression(max_iter=3000, C=1.0, random_state=SEED, class_weight='balanced')
        clf.fit(X_tr, y_tr, sample_weight=sw_tr)
    except Exception as e:
        print(f"分层分割失败: {e}, 改用普通分割")
        X_tr, X_te, y_tr, y_te, sw_tr, sw_te = train_test_split(
            X, labels, sw, test_size=0.2, random_state=SEED)
        clf = LogisticRegression(max_iter=3000, C=1.0, random_state=SEED, class_weight='balanced')
        clf.fit(X_tr, y_tr, sample_weight=sw_tr)

    # 评估
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"\n测试准确率: {acc:.4f} ({len(y_te)} 条)")

    # 混淆矩阵 (S层级)
    s_map = {'1':'S1','2':'S2','3':'S3'}
    y_te_s = [s_map[l[0]] for l in y_te]
    y_pr_s = [s_map[l[0]] for l in y_pred]
    cm = confusion_matrix(y_te_s, y_pr_s, labels=['S1','S2','S3'])
    print("\n混淆矩阵 (S层级):")
    print(f"{'':>8} {'S1':>6} {'S2':>6} {'S3':>6}")
    for i, lbl in enumerate(['S1','S2','S3']):
        row = f"  {lbl:>6}" + ''.join(f'{cm[i,j]:6d}' for j in range(3))
        print(row)

    # 修正子集准确率
    corrected_in_test = [i for i in range(len(y_te)) if sw_te[i] >= args.weight]
    if corrected_in_test:
        h_acc = accuracy_score([y_te[i] for i in corrected_in_test],
                               [y_pred[i] for i in corrected_in_test])
        print(f"修正样本测试集准确率: {h_acc:.4f} ({len(corrected_in_test)} 条)")

    print(f"\n分类报告:\n")
    print(classification_report(y_te, y_pred, zero_division=0))

    # 全量预测
    all_pred = clf.predict(X)
    for i, item in enumerate(corrected_data):
        item['labels'] = {'label': all_pred[i]}

    out_path = Path(args.output)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(corrected_data, f, ensure_ascii=False, indent=2)

    # 最终分布
    dist_final = Counter(all_pred)
    s1 = sum(v for k,v in dist_final.items() if k.startswith('1.'))
    s2 = sum(v for k,v in dist_final.items() if k.startswith('2.'))
    s3 = sum(v for k,v in dist_final.items() if k.startswith('3.'))
    n = len(all_pred)
    s3_cls = sorted([k for k in dist_final if k.startswith('3.')])
    print(f"\n最终标注分布:")
    print(f"  S1={s1}({s1/n*100:.1f}%) S2={s2}({s2/n*100:.1f}%) S3={s3}({s3/n*100:.1f}%)")
    print(f"  类数: {len(dist_final)}/31")
    print(f"  S3子类: {len(s3_cls)}/5 {s3_cls}")
    print(f"\n输出: {out_path}")

    # 保存模型
    import joblib
    model_path = out_path.parent / f"{out_path.stem}_model.pkl"
    vec_path = out_path.parent / f"{out_path.stem}_vectorizer.pkl"
    joblib.dump(clf, model_path)
    joblib.dump(vec, vec_path)
    print(f"模型: {model_path}")


if __name__ == '__main__':
    main()
