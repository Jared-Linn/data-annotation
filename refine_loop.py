#!/usr/bin/env python3
"""
refine_loop.py — 两阶段自动修正循环

阶段1: refine_loop.py --phase prepare --file student-01
  → 训练模型 → 生成 review CSV → 生成 correction_tasks.json

阶段2: (等子代理修正完成后)
  refine_loop.py --phase apply --file student-01 --corrections data/student-01_corrections.json
  → 合并修正 → 加权重训 → 输出 refined JSON

用法:
  python3 refine_loop.py --phase prepare --file student-01
  python3 refine_loop.py --phase apply --file student-01 --corrections data/student-01_corrections.json
  python3 refine_loop.py --phase full --file student-01 --corrections ... --target 0.95
"""
import json, re, os, sys, time, math, csv, argparse, random
from pathlib import Path
from collections import Counter
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_DIR = Path("/home/osboxes/Desktop/data-annotation/data")
SEED = 42

# === 标签体系 ===
S3_KW = {'3.1':['正在自杀','跳楼','上吊','割腕','服药','在自杀'],'3.2':['想自杀','自杀计划','准备死','安排后事','写遗书','计划自杀'],'3.3':['自残','划手','割手','烫自己','伤害身体','自伤'],'3.4':['打人','杀人','伤人','持刀','攻击','暴力'],'3.5':['报复','报仇','杀人计划','干掉','弄死']}
S2_KW = {'2.1':['抑郁','抑郁症','想死','活着没意思','不想活了','轻生','没意义','绝望','无助','悲伤','哭','想哭','情绪低落','开心不起来','伤害自己','没价值','废人','累赘'],'2.2':['焦虑症','惊恐','心慌','心悸','手抖','出汗','恐惧','害怕','紧张过度','莫名紧张','坐立不安','社交恐惧','恐惧症','心跳加速','呼吸困难','胸闷'],'2.3':['躁郁','双相','情绪波动','情绪极端','亢奋','精力旺盛','不睡觉','语速快','思维跳跃','冲动消费'],'2.4':['创伤','PTSD','阴影','童年','虐待','性侵','家暴','霸凌','噩梦','闪回','应激'],'2.5':['恐慌','濒死','窒息','惊恐发作','panic','急性焦虑','突然心悸'],'2.6':['厌食','暴食','催吐','节食','减肥','体重','进食障碍','吃不下','暴饮暴食'],'2.7':['强迫','强迫症','反复','洁癖','检查','控制不住','重复','停不下来','洗手'],'2.8':['酗酒','酒瘾','吸毒','成瘾','药物','赌博','网瘾','游戏成瘾','戒不掉','依赖'],'2.9':['幻觉','幻听','妄想','精神病','精神分裂','异常','呆滞','自言自语']}
S1_KW = {'1.1':['学业','考研','听课','成绩','考试','毕业','就业','求职','面试','学习','读书','作业','挂科','补考','论文','答辩','考研失败','考不上','成绩下滑','专业','选课','课堂'],'1.2':['工作','同事','老板','加班','绩效','辞职','职场','实习','转正','工资','薪水','升职','社团','班级','沟通','岗位'],'1.3':['父母','爸妈','父亲','母亲','家庭','家人','离婚','吵架','亲子','奶奶','爷爷','经济压力','家庭经济','观念分歧'],'1.4':['喝酒','吸烟','抽烟','棋牌','偶尔喝酒','小酌'],'1.5':['去世','离世','丧','葬礼','送别','过世','亲人离开','悼念','怀念'],'1.6':['失眠','睡不着','入睡','熬夜','睡眠','夜醒','早起','醒得早','难入睡','多梦','睡不好'],'1.7':['压力','焦虑','紧张','烦躁','疲惫','累','紧绷','心烦意乱','提不起劲','乏力','没精神'],'1.8':['社交','朋友','相处','邻里','同学','人际关系','不合群','社恐','内向','不敢说话','圈子','陌生人','聚会','社交场合'],'1.9':['男朋友','女朋友','男友','女友','恋爱','暗恋','异地','分手','对象','老公','老婆','夫妻','婚姻','结婚','相亲','挑明','表白','出轨','暧昧'],'1.10':['离异','单亲','抚养权','再婚','后爸','后妈'],'1.11':['分手','前任','前男友','前女友','失恋','走出来','放不下','复合','挽回'],'1.12':['性格','兴趣','爱好','方向','迷茫','我是谁','自我','探索','人生意义','价值观'],'1.13':['自卑','自卑感','低自尊','敏感','在意别人','自我怀疑','没自信','不自信','觉得自己差','看不起'],'1.14':['青春期','发育','身体','发育焦虑','青春期困惑','青春'],'1.15':['性','性取向','同性','异性','性困惑','自慰','手淫','性行为','性欲','性冲动','性心理'],'1.16':['和孩子','儿子','女儿','教育','管教','叛逆','代沟','沟通','说教','孩子不听话'],'1.17':['难受','难过','不开心','郁闷','烦','无聊','没意思']}
ALL_KW = {**S3_KW, **S2_KW, **S1_KW}

TAXONOMY = """
S1 (日常困扰): 1.1学业 1.2职场 1.3家庭 1.4消遣 1.5离世 1.6失眠 1.7压力 1.8社交 1.9亲密关系 1.10离异 1.11分手 1.12自我探索 1.13低自尊 1.14青春期 1.15性认知 1.16亲子 1.17其他
S2 (中度障碍): 2.1抑郁 2.2焦虑 2.3双相 2.4PTSD 2.5恐慌 2.6饮食障碍 2.7强迫 2.8成瘾 2.9其他
S3 (紧急危机): 3.1正在自杀 3.2自杀计划 3.3自残 3.4伤害他人 3.5报复
规则: S3>S2>S1, 选最匹配子类
"""


def clean(text):
    return re.sub(r'\s+', '', re.sub(r'[^\u4e00-\u9fff\w]', '', text))


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
    return '1.17'


def load_student(student_id):
    """加载数据, 返回 (raw, texts, labels)"""
    stem = f"student-{student_id}"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    in_path = DATA_DIR / f"{stem}.json"
    if not in_path.exists():
        print(f"文件不存在: {in_path}")
        return None

    with open(in_path) as f:
        raw = json.load(f)
    texts = [build_text(item) for item in raw]
    labels = []
    for item in raw:
        t = clean(item.get('question_title', '') + item.get('question_content', ''))
        labels.append(heuristic_label(t))
    return raw, texts, labels, stem


def train_model(texts, labels, sample_weights=None):
    """TF-IDF + LogisticRegression, 返回 (vec, clf, X, acc)"""
    vec = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b',
                          ngram_range=(1, 3), max_features=10000,
                          min_df=3, max_df=0.8, sublinear_tf=True)
    X = vec.fit_transform(texts)

    # 训练
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)

    # 拆分评估
    try:
        counter = Counter(labels)
        use_stratify = min(counter.values()) >= 2
        split_kw = {'test_size': 0.2, 'random_state': SEED}
        if use_stratify:
            split_kw['stratify'] = labels

        if sample_weights is not None:
            X_tr, X_te, y_tr, y_te, sw_tr, sw_te = train_test_split(
                X, labels, sample_weights, **split_kw)
            clf.fit(X_tr, y_tr, sample_weight=sw_tr)
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(X, labels, **split_kw)
            clf.fit(X_tr, y_tr)
    except Exception:
        # fallback
        X_tr, X_te, y_tr, y_te = train_test_split(X, labels, test_size=0.2, random_state=SEED)
        clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    return vec, clf, X, acc


# ============================================================
# 阶段1: 准备修正任务
# ============================================================
def phase_prepare(student_id, n_samples=500, labels_file=None):
    np.random.seed(SEED)

    loaded = load_student(student_id)
    if not loaded:
        return
    raw, texts, labels, stem = loaded
    n_total = len(raw)
    print(f"{stem}: {n_total} 条")

    if labels_file:
        with open(labels_file) as f:
            labeled = json.load(f)
        labels = [item['labels']['label'] for item in labeled]
        print(f"使用已有标注结果: {labels_file} ({len(labels)} 条)")
    else:
        # 初始: 启发式标签
        labels = []
        for item in raw:
            t = clean(item.get('question_title', '') + item.get('question_content', ''))
            labels.append(heuristic_label(t))

    # 初始训练
    vec, clf, X, acc = train_model(texts, labels)
    print(f"初始准确率: {acc:.4f}")
    probs = clf.predict_proba(X)
    max_probs = probs.max(axis=1)
    all_pred = clf.predict(X)

    # 选不确定样本
    uncertain_idx = np.argsort(max_probs)[:n_samples * 2]
    class_buckets = {}
    for i in uncertain_idx:
        class_buckets.setdefault(labels[i], []).append(i)

    selected = []
    per_class = max(1, n_samples // max(1, len(class_buckets)))
    for lbl, indices in class_buckets.items():
        selected.extend(indices[:per_class])
    selected = list(set(selected))[:n_samples]
    random.shuffle(selected)

    # 生成 review CSV
    review_csv = DATA_DIR / f"{stem}_review.csv"
    with open(review_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['idx', 'question_id', 'heuristic_label', 'model_label',
                         'confidence', 'correct_label', 'title', 'content_preview'])
        for idx in selected:
            item = raw[idx]
            writer.writerow([
                idx, item['question_id'], labels[idx], all_pred[idx],
                f"{max_probs[idx]:.3f}", '',
                item.get('question_title', '')[:60],
                item.get('question_content', '')[:80],
            ])
    print(f"Review CSV: {review_csv} ({len(selected)} 条)")

    # 生成修正任务 JSON
    tasks = []
    for idx in selected:
        item = raw[idx]
        dialogs = []
        for ans in item.get('answers', [])[:2]:
            for d in ans.get('dialogs', [])[:3]:
                dialogs.append(d.get('content', '')[:150])
        tasks.append({
            'idx': int(idx),
            'question_id': item['question_id'],
            'title': item.get('question_title', '')[:100],
            'content': item.get('question_content', '')[:300],
            'dialogs': dialogs,
            'old_label': str(all_pred[idx]),
            'confidence': round(float(max_probs[idx]), 4),
        })

    task_file = DATA_DIR / f"{stem}_correction_tasks.json"
    with open(task_file, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    print(f"修正任务: {task_file} ({len(tasks)} 条)")
    print(f"\n下一步: 用子代理修正标签, 输出到 {DATA_DIR}/{stem}_correction_results.json")
    print(f"然后运行: python3 refine_loop.py --phase apply --file {student_id}")

    return task_file


# ============================================================
# 阶段2: 应用修正 + 加权重训
# ============================================================
def phase_apply(student_id, corrections_path, output_suffix='refined', weight=10.0, c_param=1.0):
    loaded = load_student(student_id)
    if not loaded:
        return
    raw, texts, labels, stem = loaded

    # 加载修正
    corrections = {}
    corr_path = Path(corrections_path)
    if corr_path.exists():
        with open(corr_path) as f:
            items = json.load(f)
        for item in items:
            corrections[int(item['idx'])] = item['label']
        print(f"修正加载: {len(corrections)} 条")

    # 应用修正
    refined_labels = list(labels)
    for idx, label in corrections.items():
        if 0 <= idx < len(refined_labels):
            refined_labels[idx] = label

    # 统计修正改变了多少
    changes = sum(1 for i in corrections if refined_labels[i] != labels[i])
    print(f"标签变化: {changes}/{len(corrections)}")

    # 权重
    sw = np.ones(len(refined_labels))
    for i in corrections:
        sw[i] = weight

    # 训练
    vec = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b',
                          ngram_range=(1, 3), max_features=10000,
                          min_df=3, max_df=0.8, sublinear_tf=True)
    X = vec.fit_transform(texts)

    clf = LogisticRegression(max_iter=1000, C=c_param, random_state=SEED)
    X_tr, X_te, y_tr, y_te = train_test_split(X, refined_labels, test_size=0.2, random_state=SEED)
    sw_tr, sw_te = train_test_split(sw, test_size=0.2, random_state=SEED)
    clf.fit(X_tr, y_tr, sample_weight=sw_tr)

    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"准确率 (测试集 {len(y_te)}条): {acc:.4f}")

    # 修正子集评估
    corrected_in_test = [i for i, _ in enumerate(y_te) if sw_te[i] >= weight]
    if corrected_in_test:
        h_pred = [y_pred[i] for i in corrected_in_test]
        h_true = [y_te[i] for i in corrected_in_test]
        h_acc = accuracy_score(h_true, h_pred)
        print(f"准确率 (修正标注子集 {len(corrected_in_test)}条): {h_acc:.4f}")

    # 全量预测并输出
    all_pred = clf.predict(X)
    for i, item in enumerate(raw):
        item['labels'] = {'label': all_pred[i]}

    out_path = DATA_DIR / f"{stem}_labeled_{output_suffix}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    dist = Counter(all_pred)
    s1 = sum(c for l, c in dist.items() if l.startswith('1.'))
    s2 = sum(c for l, c in dist.items() if l.startswith('2.'))
    s3 = sum(c for l, c in dist.items() if l.startswith('3.'))
    print(f"层级: S1={s1} ({s1/len(raw)*100:.1f}%)  S2={s2} ({s2/len(raw)*100:.1f}%)  S3={s3} ({s3/len(raw)*100:.1f}%)")
    print(f"输出: {out_path}")
    print(f"修正样本: {len(corrections)}/{len(raw)}")
    return acc


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare', 'apply', 'full'], required=True)
    parser.add_argument('--file', required=True, help='文件编号, 如 student-01 或 01')
    parser.add_argument('--corrections', help='修正结果 JSON (apply/full 需要)')
    parser.add_argument('--n', type=int, default=500, help='每轮采样数')
    parser.add_argument('--target', type=float, default=0.95, help='目标准确率')
    parser.add_argument('--max-cycles', type=int, default=5, help='最大循环数')
    parser.add_argument('--weight', type=float, default=10.0, help='修正样本权重')
    parser.add_argument('--labels', help='已有标注 JSON (用于 prepare 阶段初始模型)')
    args = parser.parse_args()

    student_id = args.file.replace('student-', '')

    random.seed(SEED)

    if args.phase == 'prepare':
        phase_prepare(student_id, args.n, args.labels)

    elif args.phase == 'apply':
        if not args.corrections:
            print("--phase apply 需要 --corrections 参数")
            return
        phase_apply(student_id, args.corrections, weight=args.weight)

    elif args.phase == 'full':
        # 全自动: prepare → 提示用户用子代理 → apply
        # 但这里不适合全自动因为子代理需要手动触发
        task_file = phase_prepare(student_id, args.n)
        if task_file:
            n_tasks = len(json.load(open(task_file)))
            print(f"\n{'=' * 60}")
            print(f"需要你手动用 delegate_task 修正 {n_tasks} 条标签")
            print(f"修正文件: {task_file}")
            print(f"然后运行: python3 refine_loop.py --phase apply --file {student_id} --corrections data/student-{student_id}_correction_results.json")


if __name__ == '__main__':
    main()
