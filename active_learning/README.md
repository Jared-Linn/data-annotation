# 主动学习 (active_learning)

自动挑选最有价值的样本进行标注，提高标注效率。

## 采样策略

| 策略 | 原理 | 适用场景 |
|------|------|---------|
| least_confidence | 最低最大概率 | S3 召回优先 |
| margin | top1-top2 概率差 | 易混淆类 |
| entropy | 信息熵最大 | 多类边界 |
| random | 随机采样 | 基线对比 |

## 运行

```bash
# 单轮实验
python -m active_learning.run --n 300 --rounds 5

# 多策略对比
python -m active_learning.run --mode compare --rounds 4

# 查看未标注池
python -m active_learning.run --mode inspect
```
