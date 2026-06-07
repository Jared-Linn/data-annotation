# 数据目录

## 目录结构

```
data/
├── README.md                  ← 本文档
├── stopwords.txt              ← 停用词表（程序自动生成）
├── No2.json ~ No37.json       ← 原始对话数据（未标注）
├── 人工标注/                   ← 标注数据（自动生成）
│   ├── No2_已标注.json ~ No37_已标注.json
│   ├── pseudo_labeled_all.json         ← 初始伪标签（关键词匹配）
│   ├── pseudo_labeled_refined.json     ← 精炼标签（Deep v3 预测）
│   └── bert_training_data.json         ← BERT 训练格式
```

## 数据说明

### 原始数据 (`No*.json`)

每条对话结构：

```json
{
  "source": "psy525",
  "question_id": "psy525_110208",
  "question_title": "网贷欠了两万现在无力偿还...",
  "question_content": "刚开始接触贷款金额不大...",
  "answers": [
    {
      "answer_user_id": "psy525_杨波",
      "answer_identity": "国家二级心理咨询师",
      "dialogs": [
        {"role": "answer_psy525_杨波", "content": "1、说与不说对于你的困境有帮助吗?"},
        {"role": "user_psy525_Mom", "content": "说的话家里会帮我偿还..."}
      ]
    }
  ]
}
```

### 标注数据 (`人工标注/*.json`)

在原始数据基础上增加 `labels` 字段：

```json
{
  "...原始字段...": "...",
  "labels": {"label": "1.3"}
}
```

## 标签体系

| 层级 | 含义 | 子类 |
|------|------|------|
| S3 紧急危机 | 重度 | 3.1自杀 3.2自杀计划 3.3自残 3.4伤人 3.5报复 |
| S2 中度障碍 | 中度 | 2.1抑郁 2.2焦虑 2.3双相 2.4PTSD 2.5恐慌 2.6饮食障碍 2.7强迫 2.8成瘾 2.9其他 |
| S1 日常困扰 | 轻度 | 1.1学业 1.2职场 1.3家庭 ... 1.17其他（共17类） |

## 注意

- 原始数据文件较大（单文件 25~60MB），不纳入 git 版本控制
- 标注数据由脚本自动生成，从 `nn/pseudo_label.py` 或 `nn/generate_labels.py` 运行
- `.gitignore` 已排除 `data/No*.json` 和 `data/人工标注/*.json`
