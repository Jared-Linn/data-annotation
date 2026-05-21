#!/usr/bin/env python3
"""
思路A: 传统NLP + 机器学习分类
心理咨询对话三级标签 (S1/S2/S3) 自动标注
"""
import json, re, random, sys
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA = Path("/home/osboxes/Desktop/data-annotation/stu/student-01.json")
OUT  = Path("/home/osboxes/Desktop/data-annotation/stu/student-01_labeled.json")
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 60)
print("Step 1: 加载数据")
print("=" * 60)
with open(DATA) as f:
    raw = json.load(f)
print(f"  总条目: {len(raw)}")

# ============================================================
# 2. 预处理
# ============================================================
print("\n" + "=" * 60)
print("Step 2: 预处理")
print("=" * 60)

def clean(text):
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'[^\u4e00-\u9fff\w]', '', text)
    return text

def build_text(item):
    """合并 title + content + answer dialogs 成一条文本"""
    parts = [item.get('question_title', ''), item.get('question_content', '')]
    for ans in item.get('answers', []):
        for d in ans.get('dialogs', []):
            parts.append(d.get('content', ''))
    return clean(' '.join(parts))

texts = [build_text(item) for item in raw]

# 去重
seen = set()
dedup = []
dup_count = 0
for t, item in zip(texts, raw):
    if t not in seen:
        seen.add(t)
        dedup.append((t, item))
    else:
        dup_count += 1
print(f"  去重: 移除 {dup_count} 条重复")
texts = [t for t, _ in dedup]
items = [item for _, item in dedup]
print(f"  清洗后: {len(texts)} 条")

# ============================================================
# 3. 小样本人工标注 (用关键词启发式 + LLM辅助种子集)
# ============================================================
print("\n" + "=" * 60)
print("Step 3: 种子集标注 — 关键词启发式 + 人工规则")
print("=" * 60)

# --- S1 关键词映射 ---
S1_KEYWORDS = {
    # 1.1 学业烦恼
    '1.1': ['学业', '考研', '听课', '成绩', '考试', '毕业', '就业', '求职', '面试',
            '学习', '读书', '作业', '挂科', '补考', '论文', '答辩', '考研失败',
            '考不上', '成绩下滑', '专业', '选课', '课堂'],
    # 1.2 职场烦恼
    '1.2': ['工作', '同事', '老板', '加班', '绩效', '辞职', '职场', '实习', '转正',
            '工资', '薪水', '升职', '社团', '班级', '沟通', '岗位'],
    # 1.3 家庭矛盾
    '1.3': ['父母', '爸妈', '父亲', '母亲', '家庭', '家人', '离婚', '吵架', '亲子',
            '奶奶', '爷爷', '经济压力', '家庭经济', '观念分歧'],
    # 1.4 轻度消遣
    '1.4': ['喝酒', '吸烟', '抽烟', '棋牌', '偶尔喝酒', '小酌'],
    # 1.5 亲友离世
    '1.5': ['去世', '离世', '丧', '葬礼', '送别', '过世', '亲人离开', '悼念',
            '怀念', '哀悼', '吊唁'],
    # 1.6 短期失眠
    '1.6': ['失眠', '睡不着', '入睡', '熬夜', '睡眠', '夜醒', '早起', '醒得早',
            '难入睡', '多梦', '睡不好'],
    # 1.7 现实压力
    '1.7': ['压力', '焦虑', '紧张', '烦躁', '疲惫', '累', '紧绷', '心烦意乱',
            '提不起劲', '乏力', '没精神'],
    # 1.8 社交矛盾
    '1.8': ['社交', '朋友', '相处', '邻里', '同学', '人际关系', '不合群', '社恐',
            '内向', '不敢说话', '圈子', '陌生人', '聚会', '社交场合'],
    # 1.9 亲密关系
    '1.9': ['男朋友', '女朋友', '男友', '女友', '恋爱', '暗恋', '异地', '分手',
            '对象', '老公', '老婆', '夫妻', '婚姻', '结婚', '相亲', '挑明',
            '表白', '出轨', '暧昧'],
    # 1.10 离异后续
    '1.10': ['离异', '单亲', '抚养权', '再婚', '后爸', '后妈'],
    # 1.11 分手情绪
    '1.11': ['分手', '前任', '前男友', '前女友', '失恋', '走出来', '放不下',
            '复合', '挽回'],
    # 1.12 自我探索
    '1.12': ['性格', '兴趣', '爱好', '方向', '迷茫', '我是谁', '自我', '探索',
            '人生意义', '价值观'],
    # 1.13 低自尊
    '1.13': ['自卑', '自卑感', '低自尊', '敏感', '在意别人', '自我怀疑', '没自信',
            '不自信', '觉得自己差', '看不起'],
    # 1.14 青春期困扰
    '1.14': ['青春期', '发育', '身体', '发育焦虑', '青春期困惑', '青春'],
    # 1.15 性认知困惑
    '1.15': ['性', '性取向', '同性', '异性', '性困惑', '自慰', '手淫', '性行为',
            '性欲', '性冲动', '性心理'],
    # 1.16 亲子日常
    '1.16': ['和孩子', '儿子', '女儿', '教育', '管教', '叛逆', '代沟', '沟通',
            '说教', '孩子不听话'],
    # 1.17 其他
    '1.17': ['难受', '难过', '不开心', '郁闷', '烦', '无聊', '没意思'],
}

# --- S2 关键词映射 ---
S2_KEYWORDS = {
    # 2.1 抑郁
    '2.1': ['抑郁', '抑郁症', '想死', '活着没意思', '不想活了', '自杀', '轻生',
            '没意义', '绝望', '无助', '悲伤', '哭', '想哭', '情绪低落', '开心不起来',
            '自残', '割腕', '划伤', '伤害自己', '没价值', '废人', '累赘'],
    # 2.2 焦虑症
    '2.2': ['焦虑症', '惊恐', '心慌', '心悸', '手抖', '出汗', '恐惧', '害怕',
            '紧张过度', '广泛性焦虑', '莫名紧张', '坐立不安', '社交恐惧', '恐惧症',
            '心跳加速', '呼吸困难', '胸闷'],
    # 2.3 双向
    '2.3': ['躁郁', '双相', '情绪波动', '情绪极端', '亢奋', '精力旺盛', '不睡觉',
            '语速快', '思维跳跃', '冲动消费'],
    # 2.4 PTSD
    '2.4': ['创伤', 'PTSD', '阴影', '童年', '虐待', '性侵', '家暴', '霸凌',
            '回忆', '噩梦', '闪回', '应激'],
    # 2.5 恐慌
    '2.5': ['恐慌', '濒死', '窒息', '惊恐发作', 'panic', '急性焦虑', '突然心悸'],
    # 2.6 饮食障碍
    '2.6': ['厌食', '暴食', '催吐', '节食', '减肥', '体重', '进食障碍', '吃不下',
            '暴饮暴食', '瘦', '胖', '身材', '体重指数'],
    # 2.7 强迫
    '2.7': ['强迫', '强迫症', '反复', '洁癖', '检查', '数数', '仪式', '控制不住',
            '重复', '停不下来', '洗手'],
    # 2.8 物质成瘾
    '2.8': ['酗酒', '酒瘾', '吸毒', '成瘾', '药物', '赌博', '网瘾', '游戏成瘾',
            '戒不掉', '依赖'],
    # 2.9 其他
    '2.9': ['幻觉', '幻听', '妄想', '精神病', '精神分裂', '异常', '不说话',
            '呆滞', '自言自语'],
}

# --- S3 关键词 ---
S3_KEYWORDS = {
    '3.1': ['正在自杀', '跳楼', '上吊', '割腕', '服药', '在自杀'],
    '3.2': ['想自杀', '自杀计划', '准备死', '安排后事', '写遗书', '计划自杀'],
    '3.3': ['自残', '划手', '割手', '烫自己', '伤害身体', '自伤'],
    '3.4': ['打人', '杀人', '伤人', '持刀', '攻击', '暴力'],
    '3.5': ['报复', '报仇', '杀人计划', '干掉', '弄死'],
}


def heuristic_label(text):
    """关键词匹配 -> 优先 S3 > S2 > S1 """
    # 先查 S3
    for label, kws in S3_KEYWORDS.items():
        if any(kw in text for kw in kws):
            return label
    # 再查 S2
    for label, kws in S2_KEYWORDS.items():
        if any(kw in text for kw in kws):
            return label
    # 最后查 S1
    for label, kws in S1_KEYWORDS.items():
        if any(kw in text for kw in kws):
            return label
    return None

# 先打启发式标签
heuristic_labels = []
labeled_count = 0
for t in texts:
    lbl = heuristic_label(t)
    if lbl:
        heuristic_labels.append(lbl)
        labeled_count += 1
    else:
        heuristic_labels.append(None)

print(f"  启发式标注覆盖率: {labeled_count}/{len(texts)} ({labeled_count/len(texts)*100:.1f}%)")

# ============================================================
# 4. 特征工程 (Char n-gram TF-IDF)
# ============================================================
print("\n" + "=" * 60)
print("Step 4: 特征工程 — Char n-gram TF-IDF")
print("=" * 60)

vectorizer = TfidfVectorizer(
    analyzer='char',
    ngram_range=(2, 5),
    max_features=10000,
    min_df=3,
    max_df=0.8,
    sublinear_tf=True,
)
X = vectorizer.fit_transform(texts)
print(f"  TF-IDF 矩阵: {X.shape[0]} docs × {X.shape[1]} features")

# ============================================================
# 5. 训练分类器
# ============================================================
print("\n" + "=" * 60)
print("Step 5: 训练 — Logistic Regression")
print("=" * 60)

# 只用启发式有标签的样本训练
train_mask = [l is not None for l in heuristic_labels]
X_train = X[train_mask]
y_train = [heuristic_labels[i] for i, m in enumerate(train_mask) if m]

# 分层抽样
X_tr, X_te, y_tr, y_te = train_test_split(
    X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
)

clf = LogisticRegression(max_iter=1000, C=1.0,
                          random_state=SEED)
clf.fit(X_tr, y_tr)

y_pred = clf.predict(X_te)
acc = accuracy_score(y_te, y_pred)
print(f"  训练集: {X_tr.shape[0]} | 测试集: {X_te.shape[0]}")
print(f"  准确率: {acc:.4f}")
print(f"\n  分类报告:")
print(classification_report(y_te, y_pred))

# 混淆矩阵
cm = confusion_matrix(y_te, y_pred)
classes = sorted(set(y_te + list(y_pred)))
print(f"\n  混淆矩阵 ({len(classes)} classes):")
print("  " + " ".join(f"{c:>8}" for c in classes))
for i, row in enumerate(cm):
    print(f"  {classes[i]:4}" + "".join(f"{v:>8}" for v in row))

# ============================================================
# 6. 全量预测
# ============================================================
print("\n" + "=" * 60)
print("Step 6: 全量预测")
print("=" * 60)

all_pred = clf.predict(X)
# 对启发式未覆盖的, 用模型结果
final_labels = []
for i, (hl, pred) in enumerate(zip(heuristic_labels, all_pred)):
    if hl is None:
        final_labels.append(pred)
    else:
        # 模型 vs 启发式: 用模型结果统一
        final_labels.append(pred)

# 统计分布
from collections import Counter
dist = Counter(final_labels)
for k in sorted(dist.keys()):
    print(f"  {k}: {dist[k]:5d} ({dist[k]/len(final_labels)*100:.1f}%)")

# 层级分布
s1 = sum(1 for l in final_labels if l.startswith('1.'))
s2 = sum(1 for l in final_labels if l.startswith('2.'))
s3 = sum(1 for l in final_labels if l.startswith('3.'))
print(f"\n  层级分布: S1={s1} ({s1/len(final_labels)*100:.1f}%)  "
      f"S2={s2} ({s2/len(final_labels)*100:.1f}%)  "
      f"S3={s3} ({s3/len(final_labels)*100:.1f}%)")

# ============================================================
# 7. 输出标准 JSON
# ============================================================
print("\n" + "=" * 60)
print("Step 7: 输出标注结果")
print("=" * 60)

with open(DATA) as f:
    original = json.load(f)

# 重建 id -> label 映射 (原始顺序, 含重复)
id_to_label = {}
out_idx = 0
for i, item in enumerate(original):
    key = item['question_id']
    # 如果有重复, 取对应标签
    if key in id_to_label:
        # 找到对应去重后的 label
        pass
    else:
        if out_idx < len(final_labels):
            id_to_label[key] = final_labels[out_idx]
            out_idx += 1
        else:
            id_to_label[key] = '1.17'

# 写回
for item in original:
    item['labels'] = {'label': id_to_label.get(item['question_id'], '1.17')}

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(original, f, ensure_ascii=False, indent=2)

out_size = OUT.stat().st_size
print(f"  输出文件: {OUT}")
print(f"  文件大小: {out_size/1024/1024:.1f} MB")
print(f"  标注条数: {len(original)}")

# ============================================================
# 8. 样例展示
# ============================================================
print("\n" + "=" * 60)
print("标注样例 (随机5条):")
print("=" * 60)
samples = random.sample(list(enumerate(original)), 5)
for idx, item in samples:
    lbl = item['labels']['label']
    title = item['question_title'][:50]
    print(f"\n  [{idx}] {lbl} | {title}")

print("\n=== 完成 ===")
