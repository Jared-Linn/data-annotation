#!/usr/bin/env python3
"""
思路A: 传统NLP + 机器学习分类
心理咨询对话三级标签 (S1/S2/S3) 自动标注
批量处理 data/ 下所有 student-*.json (排除已标注的)
"""
import json, re, random, sys, glob, os, time
from pathlib import Path
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_DIR = Path("/home/osboxes/Desktop/data-annotation/data")
SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# ============================================================
# 关键词体系 (S3 > S2 > S1)
# ============================================================
S3_KEYWORDS = {
    '3.1': ['正在自杀', '跳楼', '上吊', '割腕', '服药', '在自杀'],
    '3.2': ['想自杀', '自杀计划', '准备死', '安排后事', '写遗书', '计划自杀'],
    '3.3': ['自残', '划手', '割手', '烫自己', '伤害身体', '自伤'],
    '3.4': ['打人', '杀人', '伤人', '持刀', '攻击', '暴力'],
    '3.5': ['报复', '报仇', '杀人计划', '干掉', '弄死'],
}
S2_KEYWORDS = {
    '2.1': ['抑郁', '抑郁症', '想死', '活着没意思', '不想活了', '自杀', '轻生',
            '没意义', '绝望', '无助', '悲伤', '哭', '想哭', '情绪低落', '开心不起来',
            '自残', '割腕', '划伤', '伤害自己', '没价值', '废人', '累赘'],
    '2.2': ['焦虑症', '惊恐', '心慌', '心悸', '手抖', '出汗', '恐惧', '害怕',
            '紧张过度', '广泛性焦虑', '莫名紧张', '坐立不安', '社交恐惧', '恐惧症',
            '心跳加速', '呼吸困难', '胸闷'],
    '2.3': ['躁郁', '双相', '情绪波动', '情绪极端', '亢奋', '精力旺盛', '不睡觉',
            '语速快', '思维跳跃', '冲动消费'],
    '2.4': ['创伤', 'PTSD', '阴影', '童年', '虐待', '性侵', '家暴', '霸凌',
            '回忆', '噩梦', '闪回', '应激'],
    '2.5': ['恐慌', '濒死', '窒息', '惊恐发作', 'panic', '急性焦虑', '突然心悸'],
    '2.6': ['厌食', '暴食', '催吐', '节食', '减肥', '体重', '进食障碍', '吃不下',
            '暴饮暴食', '瘦', '胖', '身材', '体重指数'],
    '2.7': ['强迫', '强迫症', '反复', '洁癖', '检查', '数数', '仪式', '控制不住',
            '重复', '停不下来', '洗手'],
    '2.8': ['酗酒', '酒瘾', '吸毒', '成瘾', '药物', '赌博', '网瘾', '游戏成瘾',
            '戒不掉', '依赖'],
    '2.9': ['幻觉', '幻听', '妄想', '精神病', '精神分裂', '异常', '不说话',
            '呆滞', '自言自语'],
}
S1_KEYWORDS = {
    '1.1': ['学业', '考研', '听课', '成绩', '考试', '毕业', '就业', '求职', '面试',
            '学习', '读书', '作业', '挂科', '补考', '论文', '答辩', '考研失败',
            '考不上', '成绩下滑', '专业', '选课', '课堂'],
    '1.2': ['工作', '同事', '老板', '加班', '绩效', '辞职', '职场', '实习', '转正',
            '工资', '薪水', '升职', '社团', '班级', '沟通', '岗位'],
    '1.3': ['父母', '爸妈', '父亲', '母亲', '家庭', '家人', '离婚', '吵架', '亲子',
            '奶奶', '爷爷', '经济压力', '家庭经济', '观念分歧'],
    '1.4': ['喝酒', '吸烟', '抽烟', '棋牌', '偶尔喝酒', '小酌'],
    '1.5': ['去世', '离世', '丧', '葬礼', '送别', '过世', '亲人离开', '悼念',
            '怀念', '哀悼', '吊唁'],
    '1.6': ['失眠', '睡不着', '入睡', '熬夜', '睡眠', '夜醒', '早起', '醒得早',
            '难入睡', '多梦', '睡不好'],
    '1.7': ['压力', '焦虑', '紧张', '烦躁', '疲惫', '累', '紧绷', '心烦意乱',
            '提不起劲', '乏力', '没精神'],
    '1.8': ['社交', '朋友', '相处', '邻里', '同学', '人际关系', '不合群', '社恐',
            '内向', '不敢说话', '圈子', '陌生人', '聚会', '社交场合'],
    '1.9': ['男朋友', '女朋友', '男友', '女友', '恋爱', '暗恋', '异地', '分手',
            '对象', '老公', '老婆', '夫妻', '婚姻', '结婚', '相亲', '挑明',
            '表白', '出轨', '暧昧'],
    '1.10': ['离异', '单亲', '抚养权', '再婚', '后爸', '后妈'],
    '1.11': ['分手', '前任', '前男友', '前女友', '失恋', '走出来', '放不下',
             '复合', '挽回'],
    '1.12': ['性格', '兴趣', '爱好', '方向', '迷茫', '我是谁', '自我', '探索',
             '人生意义', '价值观'],
    '1.13': ['自卑', '自卑感', '低自尊', '敏感', '在意别人', '自我怀疑', '没自信',
             '不自信', '觉得自己差', '看不起'],
    '1.14': ['青春期', '发育', '身体', '发育焦虑', '青春期困惑', '青春'],
    '1.15': ['性', '性取向', '同性', '异性', '性困惑', '自慰', '手淫', '性行为',
             '性欲', '性冲动', '性心理'],
    '1.16': ['和孩子', '儿子', '女儿', '教育', '管教', '叛逆', '代沟', '沟通',
             '说教', '孩子不听话'],
    '1.17': ['难受', '难过', '不开心', '郁闷', '烦', '无聊', '没意思'],
}
ALL_KW = {**S3_KEYWORDS, **S2_KEYWORDS, **S1_KEYWORDS}  # 有序: S3->S2->S1


def clean(text):
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'[^\u4e00-\u9fff\w]', '', text)
    return text


def segment(text):
    return ' '.join(jieba.cut(text))


def build_text(item):
    parts = [item.get('question_title', ''), item.get('question_content', '')]
    for ans in item.get('answers', []):
        for d in ans.get('dialogs', []):
            parts.append(d.get('content', ''))
    return segment(clean(' '.join(parts)))


def heuristic_label(text):
    for label, kws in ALL_KW.items():
        if any(kw in text for kw in kws):
            return label
    return None


def process_one(in_path, out_path):
    """处理单个文件, 返回 (文件大小MB, 条目数, 准确率)"""
    print(f"\n{'=' * 60}")
    print(f"处理: {in_path.name}")
    print(f"{'=' * 60}")

    # 1. 加载
    with open(in_path) as f:
        raw = json.load(f)
    n_total = len(raw)
    print(f"  总条目: {n_total}")

    # 2. 预处理
    texts = [build_text(item) for item in raw]

    seen = set()
    dedup_texts = []
    dedup_items = []
    dup_count = 0
    for t, item in zip(texts, raw):
        if t not in seen:
            seen.add(t)
            dedup_texts.append(t)
            dedup_items.append(item)
        else:
            dup_count += 1
    print(f"  去重: 移除 {dup_count} 条重复")
    print(f"  清洗后: {len(dedup_texts)} 条")

    # 3. 启发式标注
    heuristic_labels = []
    labeled_count = 0
    for t in dedup_texts:
        lbl = heuristic_label(t)
        heuristic_labels.append(lbl)
        if lbl:
            labeled_count += 1
    print(f"  启发式标注覆盖率: {labeled_count}/{len(dedup_texts)} "
          f"({labeled_count / len(dedup_texts) * 100:.1f}%)")

    # 4. TF-IDF
    vectorizer = TfidfVectorizer(
        analyzer='word', token_pattern=r'(?u)\b\w+\b',
        ngram_range=(1, 3), max_features=10000,
        min_df=3, max_df=0.8, sublinear_tf=True,
    )
    X = vectorizer.fit_transform(dedup_texts)
    print(f"  TF-IDF 矩阵: {X.shape[0]} docs × {X.shape[1]} features")

    # 5. 训练
    train_mask = [l is not None for l in heuristic_labels]
    X_train = X[train_mask]
    y_train = [heuristic_labels[i] for i, m in enumerate(train_mask) if m]

    if len(set(y_train)) < 2:
        # 标签太少, 直接回退启发式
        print("  警告: 有效标签类太少 (<2), 跳过模型训练, 直接使用启发式结果")
        final_labels = [l if l else '1.17' for l in heuristic_labels]
        acc = 0.0
    else:
        # 检查每个类的样本数, 不足2的类不能用 stratified split
        from collections import Counter
        y_counter = Counter(y_train)
        min_class_size = min(y_counter.values())
        if min_class_size < 2:
            print(f"  警告: 存在稀有类(最小样本数={min_class_size}), 使用普通随机分割")
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_train, y_train, test_size=0.2, random_state=SEED
            )
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
            )
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        print(f"  训练集: {X_tr.shape[0]} | 测试集: {X_te.shape[0]}")
        print(f"  准确率: {acc:.4f}")

        # 全量预测
        all_pred = clf.predict(X)
        # 模型结果覆盖所有
        final_labels = list(all_pred)

    # 6. 统计
    from collections import Counter
    dist = Counter(final_labels)
    s1 = sum(c for l, c in dist.items() if l.startswith('1.'))
    s2 = sum(c for l, c in dist.items() if l.startswith('2.'))
    s3 = sum(c for l, c in dist.items() if l.startswith('3.'))
    nl = len(final_labels)
    print(f"  层级分布: S1={s1} ({s1 / nl * 100:.1f}%)  "
          f"S2={s2} ({s2 / nl * 100:.1f}%)  "
          f"S3={s3} ({s3 / nl * 100:.1f}%)")

    # 7. 写回原始顺序
    with open(in_path) as f:
        original = json.load(f)

    id_to_label = {}
    out_idx = 0
    for i, item in enumerate(original):
        key = item['question_id']
        if key in id_to_label:
            pass  # 重复的 skip
        else:
            if out_idx < len(final_labels):
                id_to_label[key] = final_labels[out_idx]
                out_idx += 1
            else:
                id_to_label[key] = '1.17'

    for item in original:
        item['labels'] = {'label': id_to_label.get(item['question_id'], '1.17')}

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(original, f, ensure_ascii=False, indent=2)

    mb = out_path.stat().st_size / 1024 / 1024
    print(f"  输出: {out_path.name} ({mb:.1f} MB, {len(original)} 条)")
    return mb, n_total, acc


# ============================================================
# 主流程: 批量处理
# ============================================================
def main():
    # 找到所有 student-*.json (排除 _labeled 的)
    pattern = str(DATA_DIR / "student-*.json")
    all_files = sorted(glob.glob(pattern))
    to_process = [p for p in all_files if '_labeled' not in p]

    if not to_process:
        print("未找到需要处理的 student-*.json 文件")
        sys.exit(1)

    print(f"找到 {len(to_process)} 个文件待处理:")
    for p in to_process:
        fsize = os.path.getsize(p) / 1024 / 1024
        print(f"  {os.path.basename(p)} ({fsize:.0f} MB)")
    print()

    results = []
    t0 = time.time()

    for in_path in to_process:
        in_path = Path(in_path)
        stem = in_path.stem  # e.g. student-01
        out_path = DATA_DIR / f"{stem}_labeled_a2.json"

        # 跳过已存在的 (可选)
        if out_path.exists():
            print(f"  跳过 {out_path.name} (已存在)")
            continue

        try:
            mb, n, acc = process_one(in_path, out_path)
            results.append((in_path.name, mb, n, acc))
        except Exception as e:
            print(f"  !! 处理失败: {e}")
            results.append((in_path.name, 0, 0, 0))

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print("批量处理完成")
    print(f"{'=' * 60}")
    print(f"耗时: {elapsed / 60:.1f} 分钟")
    print(f"文件 | 大小 | 条目 | 准确率")
    print("-" * 50)
    for name, mb, n, acc in results:
        print(f"  {name:20s} | {mb:4.0f} MB | {n:5d} | {acc:.2%}")
    print(f"\n总计: {len(results)} 个文件")


if __name__ == '__main__':
    main()
