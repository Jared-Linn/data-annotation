#!/usr/bin/env python3
"""
auto_correction.py — 启发式增强自动修正
读取 correction_tasks.json, 输出 correction_results.json

策略:
  1. S3关键词覆盖: 任何含S3关键词的样本强制标S3
  2. 启发式共识: 当模型置信度<0.5, 改用启发式标签
  3. 修正+apply合并运行
"""
import json, sys, re, os
from pathlib import Path

DATA_DIR = Path("./data")

S3_KW = {
    '3.1':['正在自杀','跳楼','上吊','割腕','服药','在自杀'],
    '3.2':['想自杀','自杀计划','准备死','安排后事','写遗书','计划自杀'],
    '3.3':['自残','划手','割手','烫自己','伤害身体','自伤'],
    '3.4':['打人','杀人','伤人','持刀','攻击','暴力'],
    '3.5':['报复','报仇','杀人计划','干掉','弄死'],
}
ALL_S3_KW = {kw for kws in S3_KW.values() for kw in kws}

S2_KW = {
    '2.1':['抑郁','抑郁症','想死','活着没意思','不想活了','轻生','没意义','绝望','无助','悲伤','哭','想哭','情绪低落','开心不起来','伤害自己','没价值','废人','累赘'],
    '2.2':['焦虑症','惊恐','心慌','心悸','手抖','出汗','恐惧','害怕','紧张过度','莫名紧张','坐立不安','社交恐惧','恐惧症','心跳加速','呼吸困难','胸闷'],
    '2.3':['躁郁','双相','情绪波动','情绪极端','亢奋','精力旺盛','不睡觉','语速快','思维跳跃','冲动消费'],
    '2.4':['创伤','PTSD','阴影','童年','虐待','性侵','家暴','霸凌','噩梦','闪回','应激'],
    '2.5':['恐慌','濒死','窒息','惊恐发作','panic','急性焦虑','突然心悸'],
    '2.6':['厌食','暴食','催吐','节食','减肥','体重','进食障碍','吃不下','暴饮暴食'],
    '2.7':['强迫','强迫症','反复','洁癖','检查','控制不住','重复','停不下来','洗手'],
    '2.8':['酗酒','酒瘾','吸毒','成瘾','药物','赌博','网瘾','游戏成瘾','戒不掉','依赖'],
    '2.9':['幻觉','幻听','妄想','精神病','精神分裂','异常','呆滞','自言自语'],
}
ALL_S2_KW = {kw for kws in S2_KW.values() for kw in kws}

S1_KW = {
    '1.1':['学业','考研','听课','成绩','考试','毕业','就业','求职','面试','学习','读书','作业','挂科','补考','论文','答辩','考研失败','考不上','成绩下滑','专业','选课','课堂'],
    '1.2':['工作','同事','老板','加班','绩效','辞职','职场','实习','转正','工资','薪水','升职','岗位'],
    '1.3':['父母','爸妈','父亲','母亲','家庭','家人','离婚','吵架','亲子','奶奶','爷爷','经济压力','家庭经济','观念分歧'],
    '1.4':['喝酒','吸烟','抽烟','棋牌','偶尔喝酒','小酌'],
    '1.5':['去世','离世','丧','葬礼','送别','过世','亲人离开','悼念','怀念','哀悼','吊唁'],
    '1.6':['失眠','睡不着','入睡','熬夜','睡眠','夜醒','早起','醒得早','难入睡','多梦','睡不好'],
    '1.7':['压力','焦虑','紧张','烦躁','疲惫','累','紧绷','心烦意乱','提不起劲','乏力','没精神'],
    '1.8':['社交','朋友','相处','邻里','同学','人际关系','不合群','社恐','内向','不敢说话','圈子','陌生人','聚会','社交场合'],
    '1.9':['男朋友','女朋友','男友','女友','恋爱','暗恋','异地','分手','对象','老公','老婆','夫妻','婚姻','结婚','相亲','表白','出轨','暧昧'],
    '1.10':['离异','单亲','抚养权','再婚','后爸','后妈'],
    '1.11':['分手','前任','前男友','前女友','失恋','走出来','放不下','复合','挽回'],
    '1.12':['性格','兴趣','爱好','方向','迷茫','我是谁','自我','探索','人生意义','价值观'],
    '1.13':['自卑','自卑感','低自尊','敏感','在意别人','自我怀疑','没自信','不自信','觉得自己差','看不起'],
    '1.14':['青春期','发育','身体','发育焦虑','青春期困惑','青春'],
    '1.15':['性','性取向','同性','异性','性困惑','自慰','手淫','性行为','性欲','性冲动','性心理'],
    '1.16':['和孩子','儿子','女儿','教育','管教','叛逆','代沟','沟通','说教','孩子不听话'],
    '1.17':['难受','难过','不开心','郁闷','烦','无聊','没意思'],
}
# 带父级层次的完整关键词
ALL_KW_S3 = {l: kws for l, kws in S3_KW.items()}
ALL_KW_S2 = {l: kws for l, kws in S2_KW.items()}
ALL_KW_S1 = {l: kws for l, kws in S1_KW.items()}

def heuristic_label(text):
    """S3 > S2 > S1 优先级"""
    for d in [ALL_KW_S3, ALL_KW_S2, ALL_KW_S1]:
        for label, kws in d.items():
            if any(kw in text for kw in kws):
                return label
    return None

def smart_correct(task):
    """为单个修正任务生成更好标签"""
    title = task.get('title', '')
    content = task.get('content', '')
    dialogs = ' '.join(task.get('dialogs', []))
    full_text = title + ' ' + content + ' ' + dialogs
    old_label = task.get('old_label', '')
    confidence = task.get('confidence', 1.0)

    # 策略1: S3关键词强覆盖
    for label, kws in S3_KW.items():
        if any(kw in full_text for kw in kws):
            if label != old_label:
                return label, f"s3_kw_override"
            return None  # 无需修正

    # 策略2: 低置信度 → 用启发式
    if confidence < 0.5:
        h = heuristic_label(full_text)
        if h and h != old_label:
            return h, f"heuristic_replace_conf={confidence:.2f}"
        if h is None and old_label != '1.17':
            return '1.17', "heuristic_fallback"

    # 策略3: 启发式 vs 模型标签不一致, 高置信度启发式胜出
    h = heuristic_label(full_text)
    if h and h != old_label and confidence < 0.7:
        return h, f"heuristic_over_model_conf={confidence:.2f}"

    return None  # 无需修正

def process_file(student_id):
    task_file = DATA_DIR / f"No-{student_id}_correction_tasks.json"
    result_file = DATA_DIR / f"No-{student_id}_correction_results.json"

    if not task_file.exists():
        print(f"跳过 No-{student_id}: 无修正任务文件")
        return

    with open(task_file, encoding='utf-8') as f:
        tasks = json.load(f)

    corrections = []
    stats = {'total': len(tasks), 'corrected': 0, 'reasons': {}}

    for task in tasks:
        result = smart_correct(task)
        if result:
            new_label, reason = result
            corrections.append({'idx': task['idx'], 'label': new_label})
            stats['corrected'] += 1
            stats['reasons'][reason] = stats['reasons'].get(reason, 0) + 1

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(corrections, f, ensure_ascii=False, indent=2)

    print(f"No-{student_id}: {stats['corrected']}/{stats['total']} 条修正")
    for reason, count in sorted(stats['reasons'].items()):
        print(f"  {reason}: {count}")

if __name__ == '__main__':
    ids = sys.argv[1:] if len(sys.argv) > 1 else ['01', '02', '03']
    for sid in ids:
        process_file(sid)
        print()
