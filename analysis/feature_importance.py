#!/usr/bin/env python3
"""模型决策特征重要性 - 每个分类靠什么词做决策"""
import json, re
from pathlib import Path
from collections import Counter
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

CHART_DIR = Path('analysis/output')
CHART_DIR.mkdir(parents=True, exist_ok=True)

DATA = Path('data')
OUT = Path('data/人工标注')

with open(DATA / 'stopwords.txt', encoding='utf-8') as f:
    STOP_WORDS = set(line.strip() for line in f if line.strip())

def cut(t):
    return ' '.join(w for w in jieba.cut(t) if w.strip() and w not in STOP_WORDS)
def cln(t):
    return re.sub(r'\s+', '', re.sub(r'[^一-鿿\w]', '', t))
def bld(item):
    p = [item.get('question_title',''), item.get('question_content','')]
    for a in item.get('answers',[]):
        for d in a.get('dialogs',[]): p.append(d.get('content',''))
    return cut(cln(' '.join(p)))

LABEL_NAMES = {
    '1.1':'学业','1.2':'职场','1.3':'家庭','1.4':'消遣','1.5':'离世',
    '1.6':'失眠','1.7':'压力','1.8':'社交','1.9':'亲密','1.10':'离异',
    '1.11':'分手','1.12':'自我','1.13':'低自尊','1.14':'青春期','1.15':'性',
    '1.16':'亲子','1.17':'其他S1',
    '2.1':'抑郁','2.2':'焦虑','2.3':'双相','2.4':'PTSD','2.5':'恐慌',
    '2.6':'饮食','2.7':'强迫','2.8':'成瘾','2.9':'其他S2',
    '3.1':'自杀','3.2':'自杀计划','3.3':'自残','3.4':'伤人','3.5':'报复',
}


def plot_top_features(coef, feature_names, class_label, top_n=20, save_path=None):
    """画特征重要性条形图"""
    top_pos = np.argsort(coef)[-top_n:]
    top_neg = np.argsort(coef)[:top_n]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

    # 正向特征（促进该分类）
    ax1.barh(range(top_n), coef[top_pos], color='#4CAF50', edgecolor='white')
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels([feature_names[i] for i in top_pos], fontsize=9)
    ax1.set_title(f'"{class_label}" 的正向特征词', fontsize=11, fontweight='bold')
    ax1.invert_yaxis()

    # 负向特征（抑制该分类）
    ax2.barh(range(top_n), coef[top_neg], color='#f44336', edgecolor='white')
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels([feature_names[i] for i in top_neg], fontsize=9)
    ax2.set_title(f'"{class_label}" 的负向特征词', fontsize=11, fontweight='bold')
    ax2.invert_yaxis()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def analyze_feature_importance(target_labels=None):
    """分析指定类目的特征重要性"""
    with open(OUT / 'No-01_待标注_3000_已标注.json', encoding='utf-8') as f:
        seed = json.load(f)

    txts = [bld(it) for it in seed]
    lbls = [it['labels']['label'] for it in seed]

    v = TfidfVectorizer(ngram_range=(1,1), max_features=5000, sublinear_tf=True, min_df=2)
    X = v.fit_transform(txts)
    feature_names = v.get_feature_names_out()

    c = LogisticRegression(class_weight='balanced', max_iter=3000, C=1.0, random_state=42, multi_class='multinomial')
    c.fit(X, lbls)

    if target_labels is None:
        # 分析最受关注的几个类
        target_labels = ['2.1', '3.2', '1.7', '1.9']

    print("=" * 60)
    print("模型决策特征 Top-15")
    print("=" * 60)

    for lbl in target_labels:
        if lbl not in c.classes_:
            continue
        idx = list(c.classes_).index(lbl)
        coef = c.coef_[idx]
        name = LABEL_NAMES.get(lbl, lbl)

        print(f"\n{lbl} {name} 的特征词:")
        top_pos = np.argsort(coef)[-15:]
        top_neg = np.argsort(coef)[:10]
        print(f"  促进该分类: ", end='')
        for i in top_pos[::-1]:
            print(f"{feature_names[i]}({coef[i]:.2f})", end=' ')
        print()
        print(f"  抑制该分类: ", end='')
        for i in top_neg:
            print(f"{feature_names[i]}({coef[i]:.2f})", end=' ')
        print()

        # 画图
        save_path = CHART_DIR / f'features_{lbl}.png'
        plot_top_features(coef, feature_names, f'{lbl} {name}', 20, save_path)


def analyze_s3_keywords():
    """S3关键词有效性分析"""
    print("\n" + "=" * 60)
    print("S3关键词有效性分析")
    print("=" * 60)

    # 检查最终版中的S3分布
    with open(OUT / 'No-01_最终版.json', encoding='utf-8') as f:
        data = json.load(f)

    s3_kw = ['自杀','跳楼','上吊','割腕','自残','划手','伤人','杀人','持刀','报复','报仇','弄死']
    kw_stats = {}
    for kw in s3_kw:
        count = 0
        s3_count = 0
        for item in data:
            txt = item.get('question_title','') + item.get('question_content','')
            for a in item.get('answers',[]):
                for d in a.get('dialogs',[]): txt += ' ' + d.get('content','')
            if kw in txt:
                count += 1
                if item['labels']['label'].startswith('3.'):
                    s3_count += 1
        if count > 0:
            kw_stats[kw] = (count, s3_count, s3_count/count*100)

    print(f"{'关键词':>8} {'出现次数':>8} {'标为S3':>8} {'S3占比':>8}")
    for kw, (c, s3, pct) in sorted(kw_stats.items(), key=lambda x: -x[1][2]):
        bar = '#' * int(pct / 5)
        print(f"  {kw:>6}  {c:>5}次   {s3:>5}次   {pct:>5.1f}%  {bar}")


if __name__ == '__main__':
    analyze_feature_importance()
    analyze_s3_keywords()
