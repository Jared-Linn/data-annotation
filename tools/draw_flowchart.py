#!/usr/bin/env python3
"""生成预测流程图"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

# 使用中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

# 颜色
c_input = '#E3F2FD'   # 浅蓝
c_stage1 = '#FFF3E0'  # 浅橙
c_stage2 = '#F3E5F5'  # 浅紫
c_fallback = '#FFCDD2'# 浅红
c_output = '#C8E6C9'  # 浅绿
c_arrow = '#666666'
c_text = '#333333'

def box(x, y, w, h, color, text, fontsize=10):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                     facecolor=color, edgecolor='#999999', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=c_text)
    return (x + w/2, y + h)  # 返回底部中点

def arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.5))

def arrow_label(x1, y1, x2, y2, label):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c_arrow, lw=1.5))
    ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, label, ha='center', va='bottom',
            fontsize=8, color='#666', fontstyle='italic')

# 标题
ax.text(5, 7.6, '两阶段分类预测流程', ha='center', va='center', fontsize=16, fontweight='bold', color=c_text)

# 输入
box(3.5, 6.5, 3, 0.7, c_input, '输入 JSON\nNo-01 / No-02 / No-03\n(各8366条)', 9)

# Stage 1
arrow(5, 6.5, 5, 5.3)
box(2, 4.3, 6, 0.9, c_stage1, 'Stage 1: S层级分类\nLR + TF-IDF → S1 / S2 / S3\n准确率 81.67%', 9)

# 三个分支
arrow_label(3, 4.3, 3, 3.2, 'S1')
arrow_label(5, 4.3, 5, 3.2, 'S2')
arrow_label(7, 4.3, 7, 3.2, 'S3')

# Stage 2 - 三个并行
box(0.5, 2.2, 3, 0.9, c_stage2, 'Stage2-S1: 子类分类\n17类 → 1.1~1.17\n准确率 49.8%', 8)
box(3.5, 2.2, 3, 0.9, c_stage2, 'Stage2-S2: 子类分类\n9类 → 2.1~2.9\n准确率 65.8%', 8)
box(6.5, 2.2, 3, 0.9, c_stage2, 'Stage2-S3: 子类分类\n5类 → 3.1~3.5\n准确率 61.9%', 8)

# 汇聚到 S3兜底
arrow(2, 2.2, 2, 1.2)
arrow(5, 2.2, 5, 1.2)
arrow(8, 2.2, 8, 1.2)

ax.text(2, 1.5, '↓', ha='center', va='center', fontsize=14, color='#999')
ax.text(5, 1.5, '↓', ha='center', va='center', fontsize=14, color='#999')
ax.text(8, 1.5, '↓', ha='center', va='center', fontsize=14, color='#999')

# S3 兜底
box(2.5, 0.8, 5, 0.6, c_fallback, 'S3关键词兜底（自杀/自残/暴力等强信号覆盖）', 8.5)

arrow(5, 0.8, 5, 0.1)

# 输出
box(3, -0.4, 4, 0.6, c_output, '输出: No-01/02/03_最终版.json\n31/31类 | S3子类5/5', 9)

# 图例
legend_y = -1.2
ax.text(1, legend_y, '■', fontsize=12, color=c_input, fontweight='bold')
ax.text(1.3, legend_y, '输入', fontsize=8, color=c_text)
ax.text(3, legend_y, '■', fontsize=12, color=c_stage1, fontweight='bold')
ax.text(3.3, legend_y, '层级分类', fontsize=8, color=c_text)
ax.text(5, legend_y, '■', fontsize=12, color=c_stage2, fontweight='bold')
ax.text(5.3, legend_y, '子类分类', fontsize=8, color=c_text)
ax.text(7, legend_y, '■', fontsize=12, color=c_fallback, fontweight='bold')
ax.text(7.3, legend_y, 'S3兜底', fontsize=8, color=c_text)
ax.text(8.8, legend_y, '■', fontsize=12, color=c_output, fontweight='bold')
ax.text(9.1, legend_y, '输出', fontsize=8, color=c_text)

plt.tight_layout()
plt.savefig('tools/预测流程图.png', dpi=200, bbox_inches='tight')
print("已保存: tools/预测流程图.png")
