#!/usr/bin/env python3
"""
主动学习 - 样本选择策略
提供多种不确定性采样方法：
- least_confidence: 最低最大概率
- margin_sampling:   top-1与top-2概率差
- entropy:           最高熵
- random:            随机基线
"""
import numpy as np
from scipy.stats import entropy


def least_confidence(probs):
    """最低置信度: 1 - max(prob)"""
    return 1 - probs.max(axis=1)


def margin_sampling(probs):
    """边界采样: top1_prob - top2_prob (越小越不确定)"""
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]
    return 1 - margin  # 越大越不确定


def entropy_sampling(probs):
    """熵: 越大越不确定"""
    # 防止 log(0)
    probs = np.clip(probs, 1e-10, 1)
    return entropy(probs, axis=1)


def random_sampling(probs):
    """随机采样（基线）"""
    return np.random.rand(probs.shape[0])


STRATEGIES = {
    'least_confidence': least_confidence,
    'margin': margin_sampling,
    'entropy': entropy_sampling,
    'random': random_sampling,
}


def select_samples(probs, n_select, strategy='least_confidence'):
    """
    从概率矩阵中选择最不确定的 n_select 个样本

    参数:
        probs: shape=(n_samples, n_classes) 预测概率矩阵
        n_select: 选择数量
        strategy: 采样策略

    返回:
        indices: 被选中的样本索引 (按不确定度降序)
        scores:  对应的不确定度分数
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"未知策略: {strategy}，可选: {list(STRATEGIES.keys())}")

    score_fn = STRATEGIES[strategy]
    scores = score_fn(probs)

    # 按不确定度降序排列
    indices = np.argsort(scores)[::-1]
    selected = indices[:n_select]

    return selected, scores[selected]
