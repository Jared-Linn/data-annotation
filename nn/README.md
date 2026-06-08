# 神经网络实验 (NN)

基于 PyTorch 的神经网络分类器，对心理咨询对话进行 S1/S2/S3 三级分类。

## 模型列表

| 文件 | 说明 | 运行 |
|------|------|------|
| `char_cnn.py` | CharCNN + MLP + LR 基线对比 | `python -m nn.char_cnn --subset 50000` |
| `char_cnn_deep.py` | 残差CharCNN v3/v4 对比 | `python -m nn.char_cnn_deep --subset 50000` |
| `self_train.py` | Self-Training 迭代改善标签 | `python -m nn.self_train --rounds 3` |
| `pseudo_label.py` | 关键词伪标签生成器 | `python -m nn.pseudo_label` |
| `generate_labels.py` | 用最佳模型生成全量标签 | `python -m nn.generate_labels` |
| `bert_finetune.py` | BERT 微调分类 | `python -m nn.bert_finetune --subset 50000` |
| `mlp_tfidf.py` | MLP + TF-IDF 实验 | `python -m nn.mlp_tfidf` |
| `word2vec_cls.py` | Word2Vec + TextCNN 实验 | `python -m nn.word2vec_cls` |
| `bert_cls.py` | BERT 微调实验(旧版) | `python -m nn.bert_cls` |
| `fusion.py` | CharCNN + TF-IDF 特征融合 | `python -m nn.fusion` |
| `hybrid_pipeline.py` | CharCNN + ML 混合流水线 | `python -m nn.hybrid_pipeline` |
| `train_save.py` | 训练并保存 CharCNN 模型 | `python -m nn.train_save` |

## 实验结果

### Stage 1 (S1/S2/S3 三级分类)

| 模型 | 数据量 | 准确率 | 参数量 | 训练时间 |
|------|--------|--------|--------|---------|
| LR + TF-IDF | 5万 | 40.3% | — | 10s |
| MLP + TF-IDF | 5万 | 49.2% | 1.3M | 15s |
| CharCNN Original | 5万 | 75.0% | 142K | 2.5min |
| CharCNN Deep v3 | 5万 | 75.7% | 2.1M | 20min |
| **BERT (全量242K)** | **25万** | **77.24% 🏆** | **102M** | **~30min** |

### Stage 2 (S1 子类 17类)

| 模型 | 准确率 |
|------|--------|
| LR + TF-IDF | 22.7% |
| CharCNN | **69.4%** |

### BERT 最终测试结果 (全量244K, 3epoch)

| 类别 | precision | recall | f1-score |
|------|-----------|--------|----------|
| S1 日常困扰 | 0.74 | 0.80 | 0.77 |
| S2 心理障碍 | 0.80 | 0.77 | 0.79 |
| S3 紧急危机 | 0.78 | 0.55 | 0.64 |
| **总体** | **0.77** | **0.77** | **0.77** |

## 完整流程

→ [训练流程文档](训练流程.md)

## 环境

```bash
conda activate data-annotation
pip install torch transformers scikit-learn jieba gensim joblib matplotlib
```
