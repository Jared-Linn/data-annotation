#!/usr/bin/env python3
"""各类目关键词特征分析 - 看模型靠什么词做决策"""
import json, re
from pathlib import Path
from collections import Counter
import numpy as np
import jieba

DATA = Path('data')
OUT = Path('ml/output')

with open(DATA / 'stopwords.txt', encoding='utf-8') as f:
    STOP_WORDS = set(line.strip() for line in f if line.strip())

def cut(t):
    return [w for w in jieba.cut(t) if w.strip() and w not in STOP_WORDS and len(w) > 1]

def cln(t):
    return re.sub(r'\s+', '', re.sub(r'[^一-鿿\w]', '', t))


# 类目中文名
LABEL_NAMES = {
    '1.1':'学业烦恼','1.2':'校园职场','1.3':'家庭矛盾','1.4':'轻度消遣','1.5':'亲友离世',
    '1.6':'短期失眠','1.7':'现实压力','1.8':'社交矛盾','1.9':'亲密关系','1.10':'离异后续',
    '1.11':'分手情绪','1.12':'自我探索','1.13':'低自尊','1.14':'青春期','1.15':'性认知',
    '1.16':'亲子日常','1.17':'其他S1',
    '2.1':'抑郁','2.2':'焦虑','2.3':'双相','2.4':'PTSD','2.5':'恐慌','2.6':'饮食障碍',
    '2.7':'强迫','2.8':'成瘾','2.9':'其他S2',
    '3.1':'正在自杀','3.2':'自杀计划','3.3':'自残','3.4':'伤害他人','3.5':'报复',
}


def analyze_keywords(in_file='No-01_最终版.json'):
    """分析各类目高频关键词"""
    with open(OUT / in_file, encoding='utf-8') as f:
        data = json.load(f)

    # 按类收集文本
    class_texts = {}
    for item in data:
        lbl = item['labels']['label']
        text = cln(item.get('question_title','') + ' ' + item.get('question_content',''))
        for a in item.get('answers',[]):
            for d in a.get('dialogs',[]):
                text += ' ' + d.get('content','')
        class_texts.setdefault(lbl, []).append(text)

    print("=" * 60)
    print("各类目高频关键词 Top-15")
    print("=" * 60)

    for lbl in sorted(class_texts.keys()):
        words = []
        for text in class_texts[lbl]:
            words.extend(cut(text))
        word_counts = Counter(words)
        total = sum(word_counts.values())
        name = LABEL_NAMES.get(lbl, lbl)
        n_samples = len(class_texts[lbl])

        print(f"\n{lbl} {name} ({n_samples}条):")
        for word, cnt in word_counts.most_common(15):
            ratio = cnt / n_samples
            bar = '#' * int(ratio * 20)
            print(f"  {word:>8}  {cnt:>4}次 ({ratio:.0%}) {bar}")

    return class_texts


def compare_similar_classes(pairs=None):
    """对比易混淆类的关键词差异"""
    if pairs is None:
        pairs = [('1.7','2.1'), ('1.9','1.11'), ('2.2','2.5'), ('3.2','3.5')]

    with open(OUT / 'No-01_最终版.json', encoding='utf-8') as f:
        data = json.load(f)

    class_texts = {}
    for item in data:
        lbl = item['labels']['label']
        text = cln(item.get('question_title','') + ' ' + item.get('question_content',''))
        class_texts.setdefault(lbl, []).append(text)

    print("\n" + "=" * 60)
    print("易混淆类关键词对比")
    print("=" * 60)

    for lbl_a, lbl_b in pairs:
        name_a = LABEL_NAMES.get(lbl_a, lbl_a)
        name_b = LABEL_NAMES.get(lbl_b, lbl_b)

        words_a = Counter()
        for t in class_texts.get(lbl_a, []):
            words_a.update(cut(t))
        words_b = Counter()
        for t in class_texts.get(lbl_b, []):
            words_b.update(cut(t))

        total_a = sum(words_a.values()) or 1
        total_b = sum(words_b.values()) or 1

        # 计算区分度：在A中比例高但在B中比例低的关键词
        diff_words = {}
        all_words = set(words_a.keys()) | set(words_b.keys())
        for w in all_words:
            rate_a = words_a.get(w, 0) / total_a
            rate_b = words_b.get(w, 0) / total_b
            if abs(rate_a - rate_b) > 0.001:
                diff_words[w] = rate_a - rate_b

        top_diff = sorted(diff_words.items(), key=lambda x: -abs(x[1]))[:10]

        print(f"\n{lbl_a} {name_a}  vs  {lbl_b} {name_b}:")
        print(f"  {'关键词':>8} {'在A中频率':>10} {'在B中频率':>10} {'偏向':>6}")
        for word, diff in top_diff:
            fa = words_a.get(word, 0) / total_a * 100
            fb = words_b.get(word, 0) / total_b * 100
            bias = 'A' if diff > 0 else 'B'
            print(f"  {word:>8}  {fa:>6.2f}%        {fb:>6.2f}%      →{bias}")


if __name__ == '__main__':
    analyze_keywords()
    compare_similar_classes()
