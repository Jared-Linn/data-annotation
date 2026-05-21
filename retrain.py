#!/usr/bin/env python3
"""解析人工审核 + retrain (全量+加权) + 输出"""
import json, re
from pathlib import Path
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA = Path("/home/osboxes/Desktop/data-annotation/data/student-01.json")
REVIEW = Path("/home/osboxes/Desktop/data-annotation/data/review_500_out.csv")
OUT = Path("/home/osboxes/Desktop/data-annotation/data/student-01_labeled_a3.json")
SEED = 42
np.random.seed(SEED)

# 1) 解析人工标注
corrected = {}
with open(REVIEW) as f:
    for line in f:
        m = re.match(r'^(\d+)\s*[→➡]\s*([\d.]+)', line)
        if not m:
            continue
        idx = int(m.group(1))
        fl = m.group(2)
        override = fl
        for p in [r'标([\d.]+)', r'改([\d.]+)', r'更像([\d.]+)']:
            cm = re.search(p, line)
            if cm:
                override = cm.group(1)
        corrected[idx] = override
print(f"人工标注: {len(corrected)} 条")

# 2) 加载
with open(DATA) as f:
    raw = json.load(f)

def cln(t):
    return re.sub(r'\s+', '', re.sub(r'[^\u4e00-\u9fff\w]', '', t))

def bld(item):
    parts = [item.get('question_title',''), item.get('question_content','')]
    for a in item.get('answers',[]):
        for d in a.get('dialogs',[]):
            parts.append(d.get('content',''))
    return ' '.join(jieba.cut(cln(' '.join(parts))))

texts = [bld(item) for item in raw]

# 3) 启发式标签
S = {
    '3.1':['正在自杀','跳楼','上吊','割腕','服药','在自杀'],
    '3.2':['想自杀','自杀计划','准备死','安排后事','写遗书','计划自杀'],
    '3.3':['自残','划手','割手','烫自己','伤害身体','自伤'],
    '3.4':['打人','杀人','伤人','持刀','攻击','暴力'],
    '3.5':['报复','报仇','杀人计划','干掉','弄死'],
    '2.1':['抑郁','抑郁症','想死','活着没意思','不想活了','轻生','没意义','绝望','无助','悲伤','哭','想哭','情绪低落','开心不起来','伤害自己','没价值','废人','累赘'],
    '2.2':['焦虑症','惊恐','心慌','心悸','手抖','出汗','恐惧','害怕','紧张过度','莫名紧张','坐立不安','社交恐惧','恐惧症','心跳加速','呼吸困难','胸闷'],
    '2.3':['躁郁','双相','情绪波动','情绪极端','亢奋','精力旺盛','不睡觉','语速快','思维跳跃','冲动消费'],
    '2.4':['创伤','PTSD','阴影','童年','虐待','性侵','家暴','霸凌','噩梦','闪回','应激'],
    '2.5':['恐慌','濒死','窒息','惊恐发作','panic','急性焦虑','突然心悸'],
    '2.6':['厌食','暴食','催吐','节食','减肥','体重','进食障碍','吃不下','暴饮暴食'],
    '2.7':['强迫','强迫症','反复','洁癖','检查','控制不住','重复','停不下来','洗手'],
    '2.8':['酗酒','酒瘾','吸毒','成瘾','药物','赌博','网瘾','游戏成瘾','戒不掉','依赖'],
    '2.9':['幻觉','幻听','妄想','精神病','精神分裂','异常','呆滞','自言自语'],
    '1.1':['学业','考研','听课','成绩','考试','毕业','就业','求职','面试','学习','读书','作业','挂科','补考','论文','答辩','考研失败','考不上','成绩下滑','专业','选课','课堂'],
    '1.2':['工作','同事','老板','加班','绩效','辞职','职场','实习','转正','工资','薪水','升职','社团','班级','沟通','岗位'],
    '1.3':['父母','爸妈','父亲','母亲','家庭','家人','离婚','吵架','亲子','奶奶','爷爷','经济压力','家庭经济','观念分歧'],
    '1.4':['喝酒','吸烟','抽烟','棋牌','偶尔喝酒','小酌'],
    '1.5':['去世','离世','丧','葬礼','送别','过世','亲人离开','悼念','怀念'],
    '1.6':['失眠','睡不着','入睡','熬夜','睡眠','夜醒','早起','醒得早','难入睡','多梦','睡不好'],
    '1.7':['压力','焦虑','紧张','烦躁','疲惫','累','紧绷','心烦意乱','提不起劲','乏力','没精神'],
    '1.8':['社交','朋友','相处','邻里','同学','人际关系','不合群','社恐','内向','不敢说话','圈子','陌生人','聚会','社交场合'],
    '1.9':['男朋友','女朋友','男友','女友','恋爱','暗恋','异地','对象','老公','老婆','夫妻','婚姻','结婚','相亲','挑明','表白','出轨','暧昧'],
    '1.10':['离异','单亲','抚养权','再婚','后爸','后妈'],
    '1.11':['前任','前男友','前女友','失恋','走出来','放不下','复合','挽回'],
    '1.12':['性格','兴趣','爱好','方向','迷茫','我是谁','自我','探索','人生意义','价值观'],
    '1.13':['自卑','自卑感','低自尊','敏感','在意别人','自我怀疑','没自信','不自信','觉得自己差','看不起'],
    '1.14':['青春期','发育','身体','发育焦虑','青春期困惑','青春'],
    '1.15':['性取向','同性','异性','性困惑','自慰','手淫','性行为','性欲','性冲动','性心理','恋物'],
    '1.16':['教育','管教','叛逆','代沟','说教','孩子不听话'],
    '1.17':['难受','难过','不开心','郁闷','烦','无聊','没意思'],
}

def heur(t):
    for lbl, kws in S.items():
        if any(kw in t for kw in kws):
            return lbl
    return '1.17'

labels = []
for i, item in enumerate(raw):
    if i in corrected:
        labels.append(corrected[i])
    else:
        labels.append(heur(cln(item.get('question_title','') + item.get('question_content',''))))

n_human = sum(1 for i in range(len(raw)) if i in corrected)
print(f"标签: {n_human} 人工 + {len(raw)-n_human} 启发式")

# 4) 全量训练 + 人工加权
vec = TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b',
                       ngram_range=(1,3), max_features=10000, min_df=3, max_df=0.8, sublinear_tf=True)
X = vec.fit_transform(texts)

# 样本权重: 人工 x10
sw = np.ones(len(labels))
for i in corrected:
    sw[i] = 10.0

X_tr, X_te, y_tr, y_te = train_test_split(X, labels, test_size=0.2, random_state=SEED, stratify=labels)
sw_tr, sw_te = train_test_split(sw, test_size=0.2, random_state=SEED, stratify=labels)

clf = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
clf.fit(X_tr, y_tr, sample_weight=sw_tr)
y_pred = clf.predict(X_te)
acc = accuracy_score(y_te, y_pred)
print(f"\n准确率 (测试集 {len(y_te)}条): {acc:.4f} ({len(set(labels))} classes)")

# 人工标注子集评估
tr_idx, te_idx = train_test_split(list(range(len(raw))), test_size=0.2, random_state=SEED, stratify=labels)
corrected_in_test = [i for i in te_idx if i in corrected]
if corrected_in_test:
    h_pred = clf.predict(vec.transform([texts[i] for i in corrected_in_test]))
    h_true = [labels[i] for i in corrected_in_test]
    h_acc = accuracy_score(h_true, h_pred)
    print(f"准确率 (人工标注测试集 {len(corrected_in_test)}条): {h_acc:.4f}")

# 5) 全量预测
all_pred = clf.predict(X)
for i, item in enumerate(raw):
    item['labels'] = {'label': all_pred[i]}

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(raw, f, ensure_ascii=False, indent=2)

from collections import Counter
dist = Counter(all_pred)
s1 = sum(c for l,c in dist.items() if l.startswith('1.'))
s2 = sum(c for l,c in dist.items() if l.startswith('2.'))
s3 = sum(c for l,c in dist.items() if l.startswith('3.'))
print(f"\n层级: S1={s1} ({s1/len(raw)*100:.1f}%)  S2={s2} ({s2/len(raw)*100:.1f}%)  S3={s3} ({s3/len(raw)*100:.1f}%)")
for k in sorted(dist):
    print(f"  {k}: {dist[k]}")
print(f"\n输出: {OUT} ({OUT.stat().st_size/1024/1024:.1f}MB)")
