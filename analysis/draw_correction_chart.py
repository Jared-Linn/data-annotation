#!/usr/bin/env python3
"""生成人工修正对比图"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 左图: 修正前后数量变化 Top 10
categories = [
    '1.9 亲密关系', '1.1 学业烦恼', '1.17 其他', '1.2 校园职场',
    '1.16 亲子日常', '1.8 社交矛盾', '1.12 自我探索', '1.7 现实压力',
    '2.1 抑郁障碍', '1.13 低自尊'
]
before = [726, 366, 361, 230, 98, 148, 53, 91, 104, 76]
after  = [396, 136, 308, 69, 73, 162, 84, 377, 320, 334]

y = range(len(categories))
height = 0.35

ax1.barh([i + height/2 for i in y], before, height, label='修正前(模型预测)',
         color='#FF8A80', edgecolor='white', linewidth=0.5)
ax1.barh([i - height/2 for i in y], after, height, label='修正后(人工标注)',
         color='#81D4FA', edgecolor='white', linewidth=0.5)

for i, (b, a) in enumerate(zip(before, after)):
    diff = a - b
    sign = '+' if diff > 0 else ''
    ax1.text(max(b, a) + 15, i, f'{b}->{a} ({sign}{diff})', va='center', fontsize=8, color='#333')

ax1.set_yticks(range(len(categories)))
ax1.set_yticklabels(categories, fontsize=9)
ax1.set_xlabel('标注数量', fontsize=10)
ax1.set_title('修正前后各类目数量对比 (Top 10)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='lower right')
ax1.invert_yaxis()
for spine in ax1.spines.values():
    spine.set_visible(False)

# 右图: S层级分布变化
labels_b = ['S1', 'S2', 'S3']
sizes_b = [77.4, 16.5, 6.0]
sizes_a = [74.1, 15.3, 10.6]

x = np.arange(len(labels_b))
width = 0.35

bars1 = ax2.bar(x - width/2, sizes_b, width, label='修正前', color='#FF8A80', edgecolor='white')
bars2 = ax2.bar(x + width/2, sizes_a, width, label='修正后', color='#81D4FA', edgecolor='white')

for bar, val in zip(bars1, sizes_b):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val}%', ha='center', fontsize=10, fontweight='bold')
for bar, val in zip(bars2, sizes_a):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val}%', ha='center', fontsize=10, fontweight='bold')

ax2.set_xticks(x)
ax2.set_xticklabels(labels_b, fontsize=11, fontweight='bold')
ax2.set_ylabel('占比 (%)', fontsize=10)
ax2.set_title('S层级分布变化', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.set_ylim(0, 90)
for spine in ax2.spines.values():
    spine.set_visible(False)

ax2.annotate('S3 +4.6%', xy=(2+width/2, 10.6), xytext=(2+width/2, 20),
            ha='center', fontsize=10, color='#EF5350', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#EF5350', lw=1.5))

fig.suptitle('人工修正效果对比（3351条修正，weight=5加权重训）', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('tools/人工修正对比图.png', dpi=200, bbox_inches='tight')
print('已保存: tools/人工修正对比图.png')
