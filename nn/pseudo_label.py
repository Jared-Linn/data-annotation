#!/usr/bin/env python3
"""
伪标签生成器 — 关键词匹配 + 规则推断
====================================

原理：心理咨询对话中，特定词汇强烈暗示某些类别。
例如出现 "自杀" → 几乎肯定是 S3.1。
先用关键词做初步标注，后续可用这些伪标签训练神经网络。

策略（按优先级）：
  1. S3 关键词匹配 → 最紧急，关键词最明确
  2. S2 关键词匹配 → 中度障碍，心理学术语明确
  3. S1 关键词匹配 → 日常困扰，关键词较泛
  4. 无匹配 → 默认 S1.17（其他）

输出格式（与旧标注文件一致）：
  {"labels": {"label": "1.1"}, ...原始字段...}
"""

import json
import re
import os
from pathlib import Path
from collections import Counter

DATA_DIR = Path('data')
OUT_DIR = DATA_DIR / '人工标注'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. 关键词词典
# ============================================================
# 每个类别配一组关键词，匹配任意一个即触发
# 关键词越长/越具体 → 准确率越高

S3_KEYWORDS = {
    '3.1': ['正在自杀', '跳楼', '上吊', '割腕', '服药自杀', '在自杀', '立刻死', '马上去死'],
    '3.2': ['自杀计划', '计划自杀', '准备死', '写遗书', '遗书', '计划死亡', '怎么死好', '什么死法'],
    '3.3': ['自残', '划手', '割手', '烫自己', '伤害身体', '自伤', '划伤自己', '割自己', '伤害自己'],
    '3.4': ['打人', '杀人', '伤人', '持刀', '攻击别人', '暴力倾向', '持械', '想打人', '想杀人'],
    '3.5': ['报复', '报仇', '同归于尽', '弄死他', '弄死她', '干掉', '让他死', '报复社会'],
}

S2_KEYWORDS = {
    '2.1': [
        '抑郁症', '抑郁', '情绪低落', '提不起劲', '没意思', '没兴趣',
        '开心不起来', '消沉', '低落', '忧郁', '不想动', '没动力',
    ],
    '2.2': [
        '焦虑症', '焦虑', '紧张', '不安', '心慌', '担心', '坐立不安',
        '惶惶', '忐忑', '焦虑不安', '心神不宁',
    ],
    '2.3': ['双相', '躁狂', '躁郁', '情绪极端', '情绪波动大', '时而高涨时而低落'],
    '2.4': [
        '创伤后', 'PTSD', '心理阴影', '噩梦', '闪回', '创伤',
        '那件事忘不了', '反复想起', '害怕回忆',
    ],
    '2.5': ['惊恐', '恐慌发作', '窒息感', '濒死感', '心跳加速', '惊恐发作', '突然害怕'],
    '2.6': [
        '厌食', '暴食', '催吐', '进食障碍', '体重', '减肥', '身材',
        '吃不下', '暴饮暴食', '吃了吐', '怕胖',
    ],
    '2.7': ['强迫症', '强迫', '反复检查', '控制不住', '洁癖', '反复想', '重复做'],
    '2.8': ['上瘾', '成瘾', '沉迷', '戒不掉', '网瘾', '酒瘾', '烟瘾', '赌博'],
    '2.9': [],  # 其他 — 关键词匹配不到时 fallback
}

S1_KEYWORDS = {
    '1.1': [
        '学业', '学习', '考试', '成绩', '考研', '毕业', '作业',
        '上课', '读书', '复习', '挂科', '补考', '升学', '高考',
        '上学', '学校', '大学', '专业',
    ],
    '1.2': [
        '工作', '职场', '同事', '领导', '老板', '辞职', '找工作',
        '就业', '面试', '跳槽', '加班', '工资', '升职', '职业',
        '失业', '打工',
    ],
    '1.3': ['家庭', '父母', '家人', '爸妈', '亲戚', '家里', '父亲', '母亲', '弟弟', '妹妹', '哥哥', '姐姐'],
    '1.4': [
        '消遣', '娱乐', '游戏', '电影', '旅游', '爱好', '兴趣', '打游戏',
        '看剧', '听歌', '运动', '健身', '跑步', '小说',
    ],
    '1.5': ['去世', '离世', '过世', '丧', '逝世', '走了', '离开人世', '死了', '死亡'],
    '1.6': ['失眠', '睡不着', '睡眠', '熬夜', '早起', '做梦', '噩梦', '睡不好', '入睡困难'],
    '1.7': ['压力', '累', '疲惫', '疲劳', '喘不过气', '崩溃', '身心俱疲', '好累', '心累', '无力'],
    '1.8': [
        '社交', '朋友', '人际', '交流', '沟通', '孤独', '内向', '社恐',
        '社交恐惧', '不合群', '没朋友', '交朋友', '说话',
    ],
    '1.9': [
        '恋爱', '男朋友', '女朋友', '对象', '老公', '老婆', '感情',
        '交往', '约会', '结婚', '伴侣', '爱人', '配偶', '谈朋友',
    ],
    '1.10': ['离婚', '离异', '分开', '前夫', '前妻', '婚姻失败', '打离婚', '解除婚姻'],
    '1.11': ['分手', '失恋', '前任', '放弃感情', '结束关系', '被甩', '提分手'],
    '1.12': [
        '自我', '迷茫', '意义', '人生', '方向', '未来', '选择', '不知道干什么',
        '找不到方向', '困惑', '我是谁', '活着为了什么',
    ],
    '1.13': ['自卑', '自信', '否定自己', '没用', '差劲', '不够好', '看不起自己', '讨厌自己'],
    '1.14': ['青春期', '叛逆', '成长', '成熟', '少年', '青少年', '长大', '叛逆期'],
    '1.15': ['性取向', '同性', '同性恋', 'LGBT', '出柜', '性', '性别', '跨性别'],
    '1.16': [
        '孩子', '儿子', '女儿', '教育', '管教', '亲子', '育儿', '宝宝',
        '娃', '带孩子', '小孩', '家长',
    ],
    '1.17': [],  # 其他 — fallback
}

# ============================================================
# 2. 工具函数
# ============================================================

def build_text(item):
    """从 JSON item 提取完整对话文本"""
    parts = [item.get('question_title', ''), item.get('question_content', '')]
    for a in item.get('answers', []):
        for d in a.get('dialogs', []):
            parts.append(d.get('content', ''))
    return ' '.join(parts)


def match_keywords(text, kw_dict):
    """
    关键词匹配核心逻辑
    返回: (label, matched_keyword) 或 (None, None)

    原理：对整个文本（标题+内容+对话）做子串匹配。
    中文 NLP 中，字符级子串匹配比分词更可靠（分词可能切碎关键词）。
    """
    for label, kws in kw_dict.items():
        for kw in kws:
            if kw in text:
                return label, kw
    return None, None


def classify_text(text):
    """
    三阶段分类：S3 > S2 > S1

    为何按此顺序？
    - S3 关键词最具体（自杀、杀人），误报率低
    - S2 次之（抑郁、焦虑），有心理学特异性
    - S1 关键词较通用（工作、学习），容易误报

    先匹配 S3 再 S2 再 S1，保证紧急类别优先。
    """
    label, kw = match_keywords(text, S3_KEYWORDS)
    if label:
        return label, 's3_kw', kw

    label, kw = match_keywords(text, S2_KEYWORDS)
    if label:
        return label, 's2_kw', kw

    label, kw = match_keywords(text, S1_KEYWORDS)
    if label:
        return label, 's1_kw', kw

    return '1.17', 'default', None  # 无匹配 → 默认


def label_file(input_path):
    """处理单个 JSON 文件，返回标注结果列表"""
    with open(input_path, encoding='utf-8') as f:
        items = json.load(f)

    results = []
    stats = Counter()

    for item in items:
        text = build_text(item)
        label, method, matched_kw = classify_text(text)

        item['labels'] = {'label': label}
        # 附上匹配信息（调试用，不干扰格式）
        item['_pseudo'] = {'method': method, 'keyword': matched_kw}
        results.append(item)
        stats[label] += 1

    return results, stats


# ============================================================
# 3. 主流程
# ============================================================

def main():
    print("=" * 60)
    print("伪标签生成器")
    print("=" * 60)

    # 扫描所有数据文件
    data_files = sorted(DATA_DIR.glob('No*.json'))
    total_stats = Counter()
    total_items = 0

    print(f"\n发现 {len(data_files)} 个数据文件\n")

    for fp in data_files:
        name = fp.stem
        results, stats = label_file(fp)

        # 统计
        n = len(results)
        total_items += n
        total_stats += stats

        # 打印该文件摘要
        s1 = sum(v for k, v in stats.items() if k.startswith('1.'))
        s2 = sum(v for k, v in stats.items() if k.startswith('2.'))
        s3 = sum(v for k, v in stats.items() if k.startswith('3.'))
        print(f"  {name}: {n}条 → S1={s1}({s1/n*100:.0f}%) "
              f"S2={s2}({s2/n*100:.0f}%) S3={s3}({s3/n*100:.0f}%)")

        # 保存到 data/人工标注/
        out_path = OUT_DIR / f'{name}_已标注.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # 汇总
    print(f"\n{'='*60}")
    print(f"总计: {total_items} 条对话已标注")
    print(f"类分布:")
    for label in sorted(total_stats.keys()):
        pct = total_stats[label] / total_items * 100
        print(f"  {label}: {total_stats[label]:>6}条 ({pct:5.1f}%)")

    print(f"\n输出目录: {OUT_DIR.resolve()}")

    # ===== 额外：合并成一份训练集 =====
    # 方便 nn 训练脚本直接读取
    all_labeled = []
    for fp in data_files:
        name = fp.stem
        out_path = OUT_DIR / f'{name}_已标注.json'
        with open(out_path, encoding='utf-8') as f:
            all_labeled.extend(json.load(f))

    seed_path = OUT_DIR / 'pseudo_labeled_all.json'
    with open(seed_path, 'w', encoding='utf-8') as f:
        json.dump(all_labeled, f, ensure_ascii=False, indent=2)
    print(f"合并训练集: {seed_path} ({len(all_labeled)}条)")

    # 类覆盖统计
    classes_covered = sorted(set(it['labels']['label'] for it in all_labeled))
    s3_count = sum(1 for c in classes_covered if c.startswith('3.'))
    s2_count = sum(1 for c in classes_covered if c.startswith('2.'))
    s1_count = sum(1 for c in classes_covered if c.startswith('1.'))
    print(f"类覆盖: {len(classes_covered)}/31 (S1:{s1_count} S2:{s2_count} S3:{s3_count})")


if __name__ == '__main__':
    main()
