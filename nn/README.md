# 神经网络实验 (NN)

基于 PyTorch 的神经网络分类器实验，探索深度学习在心理咨询对话分类上的效果。

## 实验记录

| 模型 | 准确率(31类) | 速度 | 说明 |
|------|-------------|------|------|
| MLP + TF-IDF | ~50% | 30s | 全连接网络，改进有限 |
| CharCNN v1 | ~47% | 76s | 字符级卷积，无需分词 |
| CharCNN v2 (Stage1) | **83.33%** | 120s | ✓ 超越 LR(81.67%) |
| CharCNN v2 (S1子类) | 52.74% | 60s | 仍低于 LinearSVC(59%) |
| Word2Vec + TextCNN | - | >10min | CPU 训练过重 |
| BERT fine-tune | - | >5min/500条 | 需 GPU 加速 |

## 当前结论

**最优方案**: CharCNN Stage1 + LinearSVC Stage2 + S3兜底

- CharCNN 在粗粒度分类（S1/S2/S3）上优于 LR
- TF-IDF + LinearSVC 在细粒度子类分类上仍有优势
- BERT 精度潜力最大，但 CPU 不可行

## 运行方式

```bash
# 字符级CNN（两阶段）
python -m nn.char_cnn

# CharCNN v2 优化版
python -m nn.char_cnn_tune

# MLP + TF-IDF
python -m nn.mlp_tfidf

# BERT 微调（需GPU）
python -m nn.bert_cls

# Word2Vec + TextCNN（需gensim）
python -m nn.word2vec_cls
```

## 依赖

```bash
pip install torch numpy scikit-learn
# 可选
pip install transformers   # BERT
pip install gensim         # Word2Vec
```
