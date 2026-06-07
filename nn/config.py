"""
nn 模块共享配置
===============

所有脚本统一从此文件导入路径和常量，避免重复定义。
修改某处（如字符表、路径）时，只需改这里。
"""

import re
from pathlib import Path

import numpy as np
import torch

# ── 路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
LABELED_DIR = DATA_DIR / '人工标注'
MODEL_DIR = PROJECT_ROOT / 'nn' / 'models'
BERT_DIR = PROJECT_ROOT / 'nn' / 'bert-model'

# 自动创建目录
MODEL_DIR.mkdir(parents=True, exist_ok=True)
BERT_DIR.mkdir(parents=True, exist_ok=True)

# ── 设备 ──
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 字符表 ──
# 覆盖心理咨询常见字，所有 CharCNN 模型共用
_CHARS = sorted(set(
    'abcdefghijklmnopqrstuvwxyz0123456789'
    '的一是不了人在我有他这那中心大小上到说会走时自家为以看好起学过如生动作发后出没开面'
    '心理情绪压力焦虑抑郁恐惧强迫悲伤愤怒痛苦绝望伤害死亡自杀攻击暴力报复学业考试工作'
    '家庭关系婚姻恋爱男女朋友父母孩子教育成绩毕业考研就业睡梦哭吃喝玩钱想知道看见听见'
))
C2I = {c: i + 1 for i, c in enumerate(_CHARS)}
VOCAB_SIZE = len(C2I) + 1  # +1 给 padding 占位符 0
MAX_LEN = 200  # 对话截断长度，大部分 200 字足够

def to_char_seq(texts, max_len=MAX_LEN, dtype=np.int32):
    """
    文本列表 → 字符索引矩阵

    用法:
        X = to_char_seq(['你好世界', '今天天气不错'])
        # → shape (2, 200) 的 int32 矩阵
    """
    X = np.zeros((len(texts), max_len), dtype=dtype)
    for i, t in enumerate(texts):
        t = re.sub(r'\s+', '', t)[:max_len]
        for j, c in enumerate(t):
            X[i, j] = C2I.get(c, 0)
    return X


def build_text(item):
    """从 JSON item 拼接完整对话文本"""
    parts = [
        item.get('question_title', ''),
        item.get('question_content', ''),
    ]
    for a in item.get('answers', []):
        for d in a.get('dialogs', []):
            parts.append(d.get('content', ''))
    return ' '.join(parts)


def load_labeled_data(data_path=None, subset=None):
    """
    加载标注数据，返回 (texts, s_labels, full_labels)

    s_labels: '1'/'2'/'3'
    full_labels: '1.1'/'2.5'/...
    """
    if data_path is None:
        data_path = LABELED_DIR / 'pseudo_labeled_all.json'
    with open(data_path, encoding='utf-8') as f:
        data = json.load(f)
    if subset:
        data = data[:subset]
    texts, s_labels, full_labels = [], [], []
    for item in data:
        texts.append(build_text(item))
        label = item['labels']['label']
        s_labels.append(label[0])
        full_labels.append(label)
    return texts, s_labels, full_labels

import json  # noqa: E402 (放在后面避免循环import)
