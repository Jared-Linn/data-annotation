#!/usr/bin/env python3
"""
rebuild_final.py — 从 labeled_a2 重建最终标注
1. refine prepare → correction → apply (weight=15)
2. S3强信号兜底
3. 输出评估报告
"""
import json, sys, csv, random, math, re
from pathlib import Path
from collections import Counter
import numpy as np
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_DIR = Path("./data")
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# === 关键词体系 (同 refine_loop.py) ===
S3_KW = {'3.1':['正在自杀','跳楼','上吊','割腕','服药','在自杀'],'3.2':['想自杀','自杀计划','准备死','安排后事','写遗书','计划自杀'],'3.3':['自残','划手','割手','烫自己','伤害身体','自伤'],'3.4':['打人','杀人','伤人','持刀','攻击','暴力'],'3.5':['报复','报仇','杀人计划','干掉','弄死']}
S2_KW = {'2.1':['抑郁','抑郁症','想死','活着没意思','不想活了','轻生','没意义','绝望','无助','悲伤','哭','想哭','情绪低落','开心不起来','伤害自己','没价值','废人','累赘'],'2.2':['焦虑症','惊恐','心慌','心悸','手抖','出汗','恐惧','害怕','紧张过度','莫名紧张','坐立不安','社交恐惧','恐惧症','心跳加速','呼吸困难','胸闷'],'2.3':['躁郁','双相','情绪波动','情绪极端','亢奋','精力旺盛','不睡觉','语速快','思维跳跃','冲动消费'],'2.4':['创伤','PTSD','阴影','童年','虐待','性侵','家暴','霸凌','噩梦','闪回','应激'],'2.5':['恐慌','濒死','窒息','惊恐发作','panic','急性焦虑','突然心悸'],'2.6':['厌食','暴食','催吐','节食','减肥','体重','进食障碍','吃不下','暴饮暴食'],'2.7':['强迫','强迫症','反复','洁癖','检查','控制不住','重复','停不下来','洗手'],'2.8':['酗酒','酒瘾','吸毒','成瘾','药物','赌博','网瘾','游戏成瘾','戒不掉','依赖'],'2.9':['幻觉','幻听','妄想','精神病','精神分裂','异常','呆滞','自言自语']}
S1_KW = {'1.1':['学业','考研','听课','成绩','考试','毕业','就业','求职','面试','学习','读书','作业','挂科','补考','论文','答辩','考研失败','考不上','成绩下滑','专业','选课','课堂'],'1.2':['工作','同事','老板','加班','绩效','辞职','职场','实习','转正','工资','薪水','升职','社团','班级','沟通','岗位'],'1.3':['父母','爸妈','父亲','母亲','家庭','家人','离婚','吵架','亲子','奶奶','爷爷','经济压力','家庭经济','观念分歧'],'1.4':['喝酒','吸烟','抽烟','棋牌','偶尔喝酒','小酌'],'1.5':['去世','离世','丧','葬礼','送别','过世','亲人离开','悼念','怀念'],'1.6':['失眠','睡不着','入睡','熬夜','睡眠','夜醒','早起','醒得早','难入睡','多梦','睡不好'],'1.7':['压力','焦虑','紧张','烦躁','疲惫','累','紧绷','心烦意乱','提不起劲','乏力','没精神'],'1.8':['社交','朋友','相处','邻里','同学','人际关系','不合群','社恐','内向','不敢说话','圈子','陌生人','聚会','社交场合'],'1.9':['男朋友','女朋友','男友','女友','恋爱','暗恋','异地','分手','对象','老公','老婆','夫妻','婚姻','结婚','相亲','表白','出轨','暧昧'],'1.10':['离异','单亲','抚养权','再婚','后爸','后妈'],'1.11':['分手','前任','前男友','前女友','失恋','走出来','放不下','复合','挽回'],'1.12':['性格','兴趣','爱好','方向','迷茫','我是谁','自我','探索','人生意义','价值观'],'1.13':['自卑','自卑感','低自尊','敏感','在意别人','自我怀疑','没自信','不自信','觉得自己差','看不起'],'1.14':['青春期','发育','身体','发育焦虑','青春期困惑','青春'],'1.15':['性','性取向','同性','异性','性困惑','自慰','手淫','性行为','性欲','性冲动','性心理'],'1.16':['和孩子','儿子','女儿','教育','管教','叛逆','代沟','沟通','说教','孩子不听话'],'1.17':['难受','难过','不开心','郁闷','烦','无聊','没意思']}
ALL_KW_S3 = {l: kws for l, kws in S3_KW.items()}
ALL_KW_S2 = {l: kws for l, kws in S2_KW.items()}
ALL_KW_S1 = {l: kws for l, kws in S1_KW.items()}
STRONG_S3 = {'正在自杀','跳楼','上吊','割腕','服药自杀','自杀计划','准备死','写遗书','安排后事','杀人计划','持刀伤人'}

def clean(t): return re.sub(r'\s+', '', re.sub(r'[^一-鿿\w]', '', t))
def seg(t): return ' '.join(jieba.cut(t))
def bld(item):
    p = [item.get('question_title',''), item.get('question_content','')]
    for a in item.get('answers',[]):
        for d in a.get('dialogs',[]): p.append(d.get('content',''))
    return seg(clean(' '.join(p)))

def heur(text):
    for d in [ALL_KW_S3, ALL_KW_S2, ALL_KW_S1]:
        for lbl, kws in d.items():
            if any(kw in text for kw in kws): return lbl
    return '1.17'

def detect_s3(text):
    """返回 (标签, 是否强信号)"""
    for kw in STRONG_S3:
        if kw in text:
            for lbl, kws in S3_KW.items():
                if any(k in text for k in kws): return lbl, True
    for lbl, kws in S3_KW.items():
        if any(kw in text for kw in kws): return lbl, False
    return None, False

import re

for label in ['01', '02', '03']:
    print(f"\n{'='*60}")
    print(f"处理 No-{label}")
    print(f"{'='*60}")

    # 加载原始 + a2标注
    raw = json.load(open(DATA_DIR / f"No-{label}.json", encoding='utf-8'))
    a2 = json.load(open(DATA_DIR / f"No-{label}_labeled_a2.json", encoding='utf-8'))
    texts = [bld(item) for item in raw]
    a2_labels = [item['labels']['label'] for item in a2]
    n_total = len(raw)
    print(f"原始数据: {n_total} 条, a2类数: {len(set(a2_labels))}")

    # 修正: S3和低置信度样本
    vec = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b', ngram_range=(1,3), max_features=10000, min_df=3, max_df=0.8, sublinear_tf=True)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
    clf.fit(X, a2_labels)
    probs = clf.predict_proba(X)
    max_probs = probs.max(axis=1)
    preds = clf.predict(X)

    # 生成修正
    corrections = {}
    for i in range(n_total):
        full = raw[i].get('question_title','') + ' ' + raw[i].get('question_content','')
        for a in raw[i].get('answers',[]):
            for d in a.get('dialogs',[]): full += ' ' + d.get('content','')

        # S3强信号 → 修正
        s3, is_strong = detect_s3(full)
        if s3 and (is_strong or not preds[i].startswith('3.')):
            corrections[i] = s3
            continue

        # 低置信度 → 启发式
        if max_probs[i] < 0.5:
            h = heur(full)
            if h != preds[i]:
                corrections[i] = h

    print(f"修正: {len(corrections)} 条")

    # 加权重训
    refined_labels = list(a2_labels)
    sw = np.ones(n_total)
    for idx, lbl in corrections.items():
        refined_labels[idx] = lbl
        sw[idx] = 15.0

    vec2 = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b', ngram_range=(1,3), max_features=10000, min_df=3, max_df=0.8, sublinear_tf=True)
    X2 = vec2.fit_transform(texts)

    # train/val split
    try:
        X_tr, X_te, y_tr, y_te, sw_tr, sw_te = train_test_split(X2, refined_labels, sw, test_size=0.2, random_state=SEED, stratify=refined_labels)
    except:
        X_tr, X_te, y_tr, y_te, sw_tr, sw_te = train_test_split(X2, refined_labels, sw, test_size=0.2, random_state=SEED)

    clf2 = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
    clf2.fit(X_tr, y_tr, sample_weight=sw_tr)
    acc = accuracy_score(y_te, clf2.predict(X_te))
    print(f"模型准确率: {acc:.4f}")

    # 全量预测
    all_pred = clf2.predict(X2)

    # S3强信号兜底
    final_overrides = 0
    for i in range(n_total):
        full = raw[i].get('question_title','') + ' ' + raw[i].get('question_content','')
        for a in raw[i].get('answers',[]):
            for d in a.get('dialogs',[]): full += ' ' + d.get('content','')
        s3, is_strong = detect_s3(full)
        if s3 and is_strong and not all_pred[i].startswith('3.'):
            all_pred[i] = s3
            final_overrides += 1

    # 写回
    for i, item in enumerate(raw):
        item['labels'] = {'label': all_pred[i]}

    out_path = DATA_DIR / f"No-{label}_labeled_refined.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    # 统计
    dist = Counter(all_pred)
    s1 = sum(v for k,v in dist.items() if k.startswith('1.'))
    s2 = sum(v for k,v in dist.items() if k.startswith('2.'))
    s3 = sum(v for k,v in dist.items() if k.startswith('3.'))
    s3_classes = sorted([k for k in dist if k.startswith('3.')])
    print(f"S3兜底覆盖: {final_overrides} 条")
    print(f"分布: S1={s1}({s1/n_total*100:.1f}%) S2={s2}({s2/n_total*100:.1f}%) S3={s3}({s3/n_total*100:.1f}%)")
    print(f"类数: {len(dist)}/31 | S3子类: {s3_classes}")
    print(f"输出: {out_path}")
