# 心理咨询对话三级标签自动标注系统

基于 NLP 方法对心理咨询对话进行 **S1 / S2 / S3** 三级分类标注，覆盖 31 个子类。

---

## 🚀 快速开始

```bash
# 1. 创建环境（conda 或 pip）
conda env create -f environment.yml
conda activate data-annotation

# 或用 pip
pip install -r requirements.txt

# 2. 生成伪标签（需要 data/ 下有 No*.json 原始数据）
python -m nn.pseudo_label

# 3. 跑基线实验
python -m nn.char_cnn --subset 50000
```

---

## 📁 项目结构

```
data-annotation/
├── nn/                    神经网络实验 ★ 当前工作重点
│   ├── config.py           共享配置（路径、字符表、工具函数）
│   ├── pseudo_label.py     关键词匹配 → 初始伪标签
│   ├── char_cnn.py         CharCNN + MLP + LR 基线对比
│   ├── char_cnn_deep.py    残差CharCNN 深度改进版
│   ├── self_train.py       Self-Training 迭代改善标签
│   ├── generate_labels.py  用模型生成全量精炼标签
│   ├── bert_finetune.py    BERT 微调
│   ├── bert_cls.py         BERT 实验（旧版）
│   ├── mlp_tfidf.py        MLP + TF-IDF 实验
│   ├── word2vec_cls.py     Word2Vec + TextCNN
│   ├── fusion.py           CharCNN + TF-IDF 特征融合
│   ├── train_save.py       训练并保存模型
│   ├── hybrid_pipeline.py  CharCNN + ML 混合流水线
│   ├── README.md           模块说明
│   ├── 训练流程.md           完整实验记录
│   ├── models/             训练好的模型权重（.pt）
│   └── bert-model/         本地 BERT 文件
│
├── ml/                    传统机器学习流水线（jieba + TF-IDF）
├── analysis/              数据洞察 & 可视化
├── active_learning/       主动学习扩增
├── web/                   Web 标注工具
├── docs/                  文档 & 实验报告
│
├── data/                  数据（.gitignore 排除大文件）
│   ├── No*.json            原始对话数据
│   ├── stopwords.txt       停用词表
│   ├── 人工标注/            标注结果（自动生成）
│   └── README.md           数据格式说明
│
├── requirements.txt        pip 依赖
├── environment.yml         conda 环境配置
├── .gitignore
└── README.md
```

---

## 🔬 两阶段分类架构

```
输入对话
    │
    ▼
Stage 1: 粗分类
────────────────
  S1 日常困扰 (43%)  ──→  Stage 2: 17个子类（学业/职场/家庭...）
  S2 心理障碍 (50%)  ──→  Stage 2:  9个子类（抑郁/焦虑/双相...）
  S3 紧急危机 ( 7%)  ──→  Stage 2:  5个子类（自杀/自残/伤人...）
```

---

## 📊 模型表现

| 模型 | Stage1 (S级) | Stage2 (S1子类) | 参数量 | 速度 |
|------|-------------|----------------|--------|------|
| LR + TF-IDF (基线) | 40.3% | 22.7% | — | 10s |
| **CharCNN** | **75.0%** ✅ | **69.4%** | **142K** | **2.5min** |
| **CharCNN Deep v3** (残差) | **75.7%** ✅ | — | **2.1M** | **20min** |
| **BERT (全量242K, 3epoch) 🏆** | **77.24%** | — | **102M** | **30min** |

> ℹ️ 详细实验记录见 [`nn/训练流程.md`](nn/训练流程.md)。BERT 远程服务器训练，RTX 3080 Ti 12GB。

---

## 🔧 开发指南

### 我想...

#### 从头跑一遍
```bash
python -m nn.pseudo_label                          # 1. 生成伪标签
python -m nn.char_cnn --subset 50000                # 2. 基线
python -m nn.char_cnn_deep --subset 50000           # 3. Deep v3
python -m nn.self_train --subset 50000 --rounds 3   # 4. Self-Training
python -m nn.generate_labels                        # 5. 精炼全量标签
```

#### 只用最快的方式推理
就用 `nn/models/char_cnn_deep_best.pt`（Deep v3，75.7%），参考 `nn/generate_labels.py` 加载预测。

#### 改进模型
- **改进伪标签** → 编辑 `nn/pseudo_label.py` 扩大关键词库
- **改进 CharCNN** → 编辑 `nn/char_cnn_deep.py` 加层/调参
- **跑 BERT** → `python -m nn.bert_finetune`（需要 GPU 4GB+）

#### 理解数据
→ 见 [`data/README.md`](data/README.md)

#### 看完整实验报告
→ 见 [`nn/训练流程.md`](nn/训练流程.md)

---

## 🏷️ 标签体系

| 层级 | 子类 |
|------|------|
| S3 紧急危机 | 3.1自杀 3.2自杀计划 3.3自残 3.4伤人 3.5报复 |
| S2 中度障碍 | 2.1抑郁 2.2焦虑 2.3双相 2.4PTSD 2.5恐慌 2.6饮食障碍 2.7强迫 2.8成瘾 2.9其他 |
| S1 日常困扰 | 1.1学业 1.2职场 1.3家庭 1.4消遣 1.5离世 1.6失眠 1.7压力 1.8社交 1.9亲密关系 1.10离异 1.11分手 1.12自我探索 1.13低自尊 1.14青春期 1.15性认知 1.16亲子 1.17其他 |

---

## 📝 环境

```bash
conda env create -f environment.yml
conda activate data-annotation
```

核心依赖：PyTorch (CUDA) + transformers + scikit-learn + jieba + gensim
