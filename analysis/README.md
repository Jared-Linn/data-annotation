# 数据分析 (analysis)

数据洞察、模型分析和可视化图表。

## 工具

| 脚本 | 功能 |
|------|------|
| `class_profile.py` | 各类目高频关键词 + 易混淆类对比 |
| `confusion_analyzer.py` | 混淆矩阵热力图 + 类间混淆分析 |
| `feature_importance.py` | 模型决策特征重要性 + S3关键词有效性 |
| `distribution_viz.py` | S分布/类目数量/对话轮数可视化 |

## 运行

```bash
# 全部分析
python -m analysis.run_analysis

# 单项
python -m analysis.run_analysis --mode keywords
python -m analysis.run_analysis --mode confusion
python -m analysis.run_analysis --mode features
python -m analysis.run_analysis --mode viz
```

## 产出图表

所有图表输出到 `analysis/output/`。
