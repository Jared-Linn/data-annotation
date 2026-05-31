#!/usr/bin/env python3
"""数据分布可视化 - 对话长度、层级比例、各类数量"""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

CHART_DIR = Path('research/data_analysis/output')
CHART_DIR.mkdir(parents=True, exist_ok=True)

DATA = Path('data')
OUT = Path('data/人工标注')

LABEL_NAMES_S = {'1':'S1 日常困扰', '2':'S2 中度障碍', '3':'S3 紧急危机'}
LABEL_NAMES = {
    '1.1':'学业','1.2':'职场','1.3':'家庭','1.4':'消遣','1.5':'离世',
    '1.6':'失眠','1.7':'压力','1.8':'社交','1.9':'亲密','1.10':'离异',
    '1.11':'分手','1.12':'自我','1.13':'低自尊','1.14':'青春期','1.15':'性',
    '1.16':'亲子','1.17':'其他','2.1':'抑郁','2.2':'焦虑','2.3':'双相',
    '2.4':'PTSD','2.5':'恐慌','2.6':'饮食','2.7':'强迫','2.8':'成瘾',
    '2.9':'其他','3.1':'自杀','3.2':'自杀计划','3.3':'自残','3.4':'伤人','3.5':'报复',
}


def plot_all_distributions():
    """生成所有分布图"""
    # 加载数据
    with open(OUT / 'No-01_待标注_3000_已标注.json', encoding='utf-8') as f:
        human = json.load(f)
    with open(OUT / 'No-01_最终版.json', encoding='utf-8') as f:
        final = json.load(f)

    human_labels = [it['labels']['label'] for it in human]
    final_labels = [it['labels']['label'] for it in final]

    # 1. S层级分布对比（人工 vs 最终）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    s_names = ['S1 日常困扰', 'S2 中度障碍', 'S3 紧急危机']

    h_s = Counter(l[0] for l in human_labels)
    f_s = Counter(l[0] for l in final_labels)
    h_vals = [h_s.get('1',0), h_s.get('2',0), h_s.get('3',0)]
    f_vals = [f_s.get('1',0), f_s.get('2',0), f_s.get('3',0)]

    colors = ['#42A5F5', '#FFA726', '#EF5350']
    ax1.pie(h_vals, labels=s_names, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('人工标注分布 (3000条)', fontsize=12, fontweight='bold')

    ax2.pie(f_vals, labels=s_names, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('模型最终分布 (8366条)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(CHART_DIR / 'viz_S_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  viz_S_distribution.png 已保存")

    # 2. 各类目数量分布（Top 20）
    fig, ax = plt.subplots(figsize=(12, 8))
    dist = Counter(final_labels)
    top20 = dist.most_common(20)
    lbls = [f'{l} {LABEL_NAMES.get(l,"")}' for l, _ in top20]
    vals = [v for _, v in top20]

    colors_bar = ['#42A5F5']*17 + ['#FFA726']*9 + ['#EF5350']*5
    colors_top = [colors_bar[i] for i in range(len(top20))]
    bars = ax.barh(range(len(top20)), vals, color=colors_top, edgecolor='white')
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(lbls, fontsize=9)
    ax.set_xlabel('样本数', fontsize=11)
    ax.set_title('各类目数量分布 (Top 20)', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(CHART_DIR / 'viz_class_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  viz_class_distribution.png 已保存")

    # 3. 对话轮数分布
    fig, ax = plt.subplots(figsize=(10, 5))
    dialog_counts = []
    for item in final:
        n = sum(len(a.get('dialogs', [])) for a in item.get('answers', []))
        s = item['labels']['label'][0]
        dialog_counts.append((s, min(n, 20)))  # 截断到20

    s1_len = [d[1] for d in dialog_counts if d[0] == '1']
    s2_len = [d[1] for d in dialog_counts if d[0] == '2']
    s3_len = [d[1] for d in dialog_counts if d[0] == '3']

    ax.hist(s1_len, bins=20, alpha=0.5, label='S1', color='#42A5F5', density=True)
    ax.hist(s2_len, bins=20, alpha=0.5, label='S2', color='#FFA726', density=True)
    ax.hist(s3_len, bins=20, alpha=0.5, label='S3', color='#EF5350', density=True)
    ax.set_xlabel('对话轮数', fontsize=11)
    ax.set_ylabel('密度', fontsize=11)
    ax.set_title('S1/S2/S3 对话轮数分布对比', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(CHART_DIR / 'viz_dialog_length.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  viz_dialog_length.png 已保存")


if __name__ == '__main__':
    plot_all_distributions()
