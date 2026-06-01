# 传统机器学习 (ML)

基于 jieba + TF-IDF + LogisticRegression 的心理咨询对话三级分类流水线。

## 流水线

| 脚本 | 功能 |
|------|------|
| `pipeline_a.py` | 初始自动标注（启发式+LR） |
| `final_pipeline.py` | **最终版**: 两阶段分类 + S3兜底 |
| `two_stage.py` | 两阶段分类架构训练 |
| `refine_loop.py` | 修正循环（prepare/apply） |
| `auto_correction.py` | 启发式自动修正 |
| `auto_correct.py` | LLM(DeepSeek)自动修正 |
| `rebuild_final.py` | 一键重建最终标注 |

## 核心流程

```
原始数据 → 两阶段分类 → 输出最终标注
               │
        Stage1: S1/S2/S3 (LR, 81.67%)
        Stage2: 子类分类 (LR/SVC, 50~66%)
        S3关键词兜底 → 31/31类覆盖
```

## 运行方式

```bash
# 最终版全量预测（输出到 ml/output/）
python -m ml.final_pipeline

# 两阶段训练
python -m ml.two_stage
```

## 输出

| 路径 | 内容 |
|------|------|
| `ml/output/*_最终版.json` | 最终标注结果 |
| `ml/output/*_带标签.json` | 含 dialog tags |
| `ml/models/` | 训练好的模型文件 |

## 优化工具

| 脚本 | 功能 |
|------|------|
| `diagnose.py` | S1子类瓶颈诊断 |
| `optimize.py` | 特征/模型组合优化实验 |
| `tune.py` | 超参微调实验 |
| `self_train.py` | 半监督自训练 |

## 依赖

```bash
pip install jieba numpy scikit-learn joblib
```
