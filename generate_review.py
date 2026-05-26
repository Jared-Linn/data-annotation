#!/usr/bin/env python3
"""
生成审核 CSV: 找出模型最不确定的样本供人工/LLM修正标签
用法: python3 generate_review.py --data data/student-01.json --n 500
"""
import json, csv, random, re, argparse
from pathlib import Path
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

SEED = 42

# --- 关键词标签 ---
S1_KW = {
    '1.1': ['学业','考研','听课','成绩','考试','毕业','就业','求职','面试','学习','读书','作业','挂科','补考','论文','答辩','考研失败','考不上','成绩下滑','专业','选课','课堂'],
    '1.2': ['工作','同事','老板','加班','绩效','辞职','职场','实习','转正','工资','薪水','升职','社团','班级','沟通','岗位'],
    '1.3': ['父母','爸妈','父亲','母亲','家庭','家人','离婚','吵架','亲子','奶奶','爷爷','经济压力','家庭经济','观念分歧'],
    '1.4': ['喝酒','吸烟','抽烟','棋牌','偶尔喝酒','小酌'],
    '1.5': ['去世','离世','丧','葬礼','送别','过世','亲人离开','悼念','怀念','哀悼','吊唁'],
    '1.6': ['失眠','睡不着','入睡','熬夜','睡眠','夜醒','早起','醒得早','难入睡','多梦','睡不好'],
    '1.7': ['压力','焦虑','紧张','烦躁','疲惫','累','紧绷','心烦意乱','提不起劲','乏力','没精神'],
    '1.8': ['社交','朋友','相处','邻里','同学','人际关系','不合群','社恐','内向','不敢说话','圈子','陌生人','聚会','社交场合'],
    '1.9': ['男朋友','女朋友','男友','女友','恋爱','暗恋','异地','分手','对象','老公','老婆','夫妻','婚姻','结婚','相亲','挑明','表白','出轨','暧昧'],
    '1.10': ['离异','单亲','抚养权','再婚','后爸','后妈'],
    '1.11': ['分手','前任','前男友','前女友','失恋','走出来','放不下','复合','挽回'],
    '1.12': ['性格','兴趣','爱好','方向','迷茫','我是谁','自我','探索','人生意义','价值观'],
    '1.13': ['自卑','自卑感','低自尊','敏感','在意别人','自我怀疑','没自信','不自信','觉得自己差','看不起'],
    '1.14': ['青春期','发育','身体','发育焦虑','青春期困惑','青春'],
    '1.15': ['性','性取向','同性','异性','性困惑','自慰','手淫','性行为','性欲','性冲动','性心理'],
    '1.16': ['和孩子','儿子','女儿','教育','管教','叛逆','代沟','沟通','说教','孩子不听话'],
    '1.17': ['难受','难过','不开心','郁闷','烦','无聊','没意思'],
}
S2_KW = {
    '2.1': ['抑郁','抑郁症','想死','活着没意思','不想活了','自杀','轻生','没意义','绝望','无助','悲伤','哭','想哭','情绪低落','开心不起来','自残','割腕','划伤','伤害自己','没价值','废人','累赘'],
    '2.2': ['焦虑症','惊恐','心慌','心悸','手抖','出汗','恐惧','害怕','紧张过度','广泛性焦虑','莫名紧张','坐立不安','社交恐惧','恐惧症','心跳加速','呼吸困难','胸闷'],
    '2.3': ['躁郁','双相','情绪波动','情绪极端','亢奋','精力旺盛','不睡觉','语速快','思维跳跃','冲动消费'],
    '2.4': ['创伤','PTSD','阴影','童年','虐待','性侵','家暴','霸凌','回忆','噩梦','闪回','应激'],
    '2.5': ['恐慌','濒死','窒息','惊恐发作','panic','急性焦虑','突然心悸'],
    '2.6': ['厌食','暴食','催吐','节食','减肥','体重','进食障碍','吃不下','暴饮暴食','瘦','胖','身材','体重指数'],
    '2.7': ['强迫','强迫症','反复','洁癖','检查','数数','仪式','控制不住','重复','停不下来','洗手'],
    '2.8': ['酗酒','酒瘾','吸毒','成瘾','药物','赌博','网瘾','游戏成瘾','戒不掉','依赖'],
    '2.9': ['幻觉','幻听','妄想','精神病','精神分裂','异常','不说话','呆滞','自言自语'],
}
S3_KW = {
    '3.1': ['正在自杀','跳楼','上吊','割腕','服药','在自杀'],
    '3.2': ['想自杀','自杀计划','准备死','安排后事','写遗书','计划自杀'],
    '3.3': ['自残','划手','割手','烫自己','伤害身体','自伤'],
    '3.4': ['打人','杀人','伤人','持刀','攻击','暴力'],
    '3.5': ['报复','报仇','杀人计划','干掉','弄死'],
}
ALL_KW = {**S3_KW, **S2_KW, **S1_KW}


def heuristic_label(text):
    for label, kws in ALL_KW.items():
        if any(kw in text for kw in kws):
            return label
    return None


def clean(text):
    return re.sub(r'\s+', '', re.sub(r'[^\u4e00-\u9fff\w]', '', text))


def build_text(item):
    parts = [item.get('question_title',''), item.get('question_content','')]
    for ans in item.get('answers',[]):
        for d in ans.get('dialogs',[]):
            parts.append(d.get('content',''))
    return ' '.join(jieba.cut(clean(' '.join(parts))))


def main():
    parser = argparse.ArgumentParser(description='Generate review CSV with uncertain samples')
    parser.add_argument('--data', required=True, help='原始 student JSON 路径')
    parser.add_argument('--output', default=None, help='输出 CSV 路径')
    parser.add_argument('--n', type=int, default=500, help='采样数量')
    args = parser.parse_args()

    np.random.seed(SEED)
    random.seed(SEED)

    DATA = Path(args.data)
    if args.output:
        REVIEW_CSV = Path(args.output)
    else:
        REVIEW_CSV = DATA.parent / f"{DATA.stem}_review.csv"

    # 加载
    print("加载数据...")
    with open(DATA) as f:
        raw = json.load(f)

    texts = [build_text(item) for item in raw]
    labels = []
    for item in raw:
        t = clean(item.get('question_title','') + item.get('question_content',''))
        lbl = heuristic_label(t)
        labels.append(lbl if lbl else '1.17')

    print(f"TF-IDF 向量化...")
    vectorizer = TfidfVectorizer(
        analyzer='word', token_pattern=r'(?u)\b\w+\b',
        ngram_range=(1,3), max_features=10000, min_df=3, max_df=0.8, sublinear_tf=True,
    )
    X = vectorizer.fit_transform(texts)

    # 训练 + 概率
    print("训练 Logistic Regression + 概率预测...")
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
    clf.fit(X, labels)
    probs = clf.predict_proba(X)
    max_probs = probs.max(axis=1)
    preds = clf.predict(X)

    # 采样: 低置信度 + 各类均衡
    N = args.n
    uncertain_idx = np.argsort(max_probs)[:min(N*2, len(max_probs))]

    class_sample = {}
    for i, lbl in enumerate(labels):
        if i in uncertain_idx:
            continue
        class_sample.setdefault(lbl, []).append(i)

    extra = []
    for lbl, indices in class_sample.items():
        take = min(len(indices), max(1, int(N * 0.7 / max(1, len(class_sample)))))
        extra.extend(random.sample(indices, take))

    all_idx = list(set(list(uncertain_idx) + extra))[:N]
    random.shuffle(all_idx)

    print(f"生成审核文件: {len(all_idx)} 条...")
    with open(REVIEW_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['idx', 'question_id', 'heuristic_label', 'model_label',
                         'confidence', 'correct_label', 'title', 'content_preview'])
        for idx in all_idx:
            item = raw[idx]
            title = item['question_title'][:60]
            content = item['question_content'][:80]
            conf = max_probs[idx]
            writer.writerow([
                idx, item['question_id'], labels[idx], preds[idx],
                f"{conf:.3f}", '', title, content
            ])

    print(f"\n输出: {REVIEW_CSV} ({len(all_idx)} 条)")
    print(f"置信度分布: < 0.3: {int((max_probs < 0.3).sum())}  < 0.5: {int((max_probs < 0.5).sum())}")


if __name__ == '__main__':
    main()
