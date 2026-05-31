#!/usr/bin/env python3
"""生成tags标注分布图"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# 数据
tags = ['knowledge\n(专业知识)', 'meaningless\n(无实质内容)', 'negative\n(负面回复)']
counts = [93975, 75277, 6087]
pcts = [73.7, 59.0, 4.8]
colors = ['#4CAF50', '#FF9800', '#f44336']

# 左: 柱状图
bars = ax1.barh(tags, counts, color=colors, height=0.5, edgecolor='white', linewidth=1.5)
for bar, count, pct in zip(bars, counts, pcts):
    ax1.text(bar.get_width() + 1000, bar.get_y() + bar.get_height()/2,
             f'{count:,}条 ({pct}%)', ha='left', va='center', fontsize=10, fontweight='bold')
ax1.set_xlim(0, 110000)
ax1.set_xlabel('对话数量', fontsize=10)
ax1.tick_params(labelsize=9)
for spine in ax1.spines.values():
    spine.set_visible(False)

# 右: 韦恩图式说明（用文字说明多重标签）
ax2.axis('off')

# 画三个重叠圆表示多标签关系
from matplotlib.patches import Circle
c1 = Circle((3.2, 3), 2.2, color='#4CAF50', alpha=0.3, ec='#4CAF50', linewidth=2)
c2 = Circle((5.2, 3), 2.2, color='#FF9800', alpha=0.3, ec='#FF9800', linewidth=2)
c3 = Circle((4.2, 1.5), 2.2, color='#f44336', alpha=0.3, ec='#f44336', linewidth=2)
ax2.add_patch(c1)
ax2.add_patch(c2)
ax2.add_patch(c3)

ax2.text(3.2, 4.2, 'knowledge', ha='center', fontsize=10, fontweight='bold', color='#2E7D32')
ax2.text(5.2, 4.2, 'meaningless', ha='center', fontsize=10, fontweight='bold', color='#E65100')
ax2.text(4.2, 0.5, 'negative', ha='center', fontsize=10, fontweight='bold', color='#C62828')
ax2.text(4.2, 2.8, '一条对话\n可同时标\n多个tags', ha='center', fontsize=9, color='#666')

ax2.set_xlim(0, 8)
ax2.set_ylim(0, 5.5)

fig.suptitle('对话标签（tags）标注结果', fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('tools/tags标注分布图.png', dpi=200, bbox_inches='tight')
print('已保存: tools/tags标注分布图.png')
