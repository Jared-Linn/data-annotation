#!/usr/bin/env python3
"""混合 pipeline: CharCNN Stage1 + ML Stage2 + S3兜底"""
import json, re, glob, sys
from pathlib import Path
from collections import Counter
import numpy as np, jieba, torch, torch.nn as nn, joblib

MODEL_DIR = Path('nn/models')
ML_MODEL_DIR = Path('ml/models')
ML_OUT = Path('ml/output')
OUT_DIR = Path('nn/output')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 字符表（必须与 train_save.py 一致）
CHARS = sorted(set('abcdefghijklmnopqrstuvwxyz0123456789' +
    '的一是不了人在我有他这那中心大小上到说会走时自家为以看好起学过如生动作发后出没开面'
    '心理情绪压力焦虑抑郁恐惧强迫悲伤愤怒痛苦绝望伤害死亡自杀攻击暴力报复学业考试工作'
    '家庭关系婚姻恋爱男女朋友父母孩子教育成绩毕业考研就业睡梦哭吃喝玩钱想知道看见听见'))
C2I = {c:i+1 for i,c in enumerate(CHARS)}
VOCAB = len(C2I) + 1

def seq(t, m=300):
    s = [C2I.get(c,0) for c in t[:m]]
    return (s+[0]*m)[:m]

class CharCNN(nn.Module):
    def __init__(self, vocab, ncls):
        super().__init__()
        self.emb = nn.Embedding(vocab, 128, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(128, 64, k, padding=k//2), nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveMaxPool1d(1))
            for k in [3,5,7]
        ])
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(64*3, ncls))
    def forward(self, x):
        x = self.emb(x).permute(0,2,1)
        x = torch.cat([c(x).squeeze(-1) for c in self.convs], dim=1)
        return self.fc(x)

# S3 关键词
S3_KW = {
    '3.1': ['正在自杀','跳楼','上吊','割腕','服药自杀','在自杀'],
    '3.2': ['想自杀','自杀计划','准备死','写遗书','计划自杀'],
    '3.3': ['自残','划手','割手','烫自己','伤害身体','自伤'],
    '3.4': ['打人','杀人','伤人','持刀','攻击','暴力','持械'],
    '3.5': ['报复','报仇','杀人计划','干掉','弄死','同归于尽'],
}

# 停用词
with open('data/stopwords.txt', encoding='utf-8') as f:
    SW = set(l.strip() for l in f if l.strip())

def cut(t): return ' '.join(w for w in jieba.cut(t) if w.strip() and w not in SW)
def cln(t): return re.sub(r'\s+','',re.sub(r'[^一-鿿\w]','',t))
def bld(it):
    p = [it.get('question_title',''), it.get('question_content','')]
    for a in it.get('answers',[]):
        for d in a.get('dialogs',[]): p.append(d.get('content',''))
    return cut(cln(' '.join(p)))
def get_full(it):
    t = it.get('question_title','') + ' ' + it.get('question_content','')
    for a in it.get('answers',[]):
        for d in a.get('dialogs',[]): t += ' ' + d.get('content','')
    return t

def detect_s3(text):
    for lbl, kws in S3_KW.items():
        if any(kw in text for kw in kws):
            return lbl
    return None

print("=" * 60)
print("混合 Pipeline: CharCNN Stage1 + ML Stage2 + S3兜底")
print("=" * 60)

# 1. 加载 CharCNN Stage1
print("\n加载 CharCNN Stage1...")
c1 = CharCNN(VOCAB, 3)
c1.load_state_dict(torch.load(MODEL_DIR / 'char_cnn_stage1.pt', map_location='cpu'))
c1.eval()
print("  CharCNN Stage1: OK")

# 2. 加载 ML Stage2 模型
print("加载 ML Stage2 模型...")
stage2 = {}
vecs = {}
for lvl in ['1','2','3']:
    mp = ML_MODEL_DIR / f'final_stage2_{lvl}.pkl'
    vp = ML_MODEL_DIR / f'final_stage2_{lvl}_vec.pkl'
    if mp.exists() and vp.exists():
        stage2[lvl] = joblib.load(mp)
        vecs[lvl] = joblib.load(vp)
        names = ['S1','S2','S3'][int(lvl)-1]
        print(f"  Stage2 {names}: OK")
    else:
        stage2[lvl] = vecs[lvl] = None

# 3. 全量预测
for tgt in ['No-01','No-02','No-03']:
    fp = f'data/{tgt}.json'
    if not Path(fp).exists():
        print(f"  跳过 {tgt}（文件不存在）")
        continue

    with open(fp, encoding='utf-8') as f:
        items = json.load(f)

    print(f"\n{tgt}: {len(items)}条")

    # Stage1: CharCNN 预测 S层级
    texts = [bld(it) for it in items]
    X_c = np.array([seq(t) for t in texts])
    with torch.no_grad():
        s_pred = c1(torch.tensor(X_c)).argmax(dim=1).numpy()
    s_map = {0:'1', 1:'2', 2:'3'}
    s_levels = [s_map[s] for s in s_pred]

    # Stage2 + S3兜底
    final = []
    for i, it in enumerate(items):
        level = s_levels[i]
        full = get_full(it)

        # S3 关键词优先
        s3 = detect_s3(full)
        if s3:
            final.append(s3)
            continue

        # Stage2
        if stage2.get(level) and vecs.get(level):
            x = vecs[level].transform([bld(it)])
            sub = stage2[level].predict(x)[0]
            final.append(sub)
        else:
            final.append(level + '.17' if level == '1' else level + '.9')

    # 统计
    d = Counter(final)
    s1 = sum(v for k,v in d.items() if k.startswith('1.'))
    s2 = sum(v for k,v in d.items() if k.startswith('2.'))
    s3 = sum(v for k,v in d.items() if k.startswith('3.'))
    s3c = sorted([k for k in d if k.startswith('3.')])
    n = len(items)

    print(f"  S1={s1}({s1/n*100:.1f}%) S2={s2}({s2/n*100:.1f}%) S3={s3}({s3/n*100:.1f}%)")
    print(f"  类: {len(d)}/31 | S3子类: {len(s3c)}/5 {s3c}")

    # 与 ml/output 对比
    ml_fp = ML_OUT / f'{tgt}_最终版.json'
    if ml_fp.exists():
        with open(ml_fp, encoding='utf-8') as f:
            ml_items = json.load(f)
        ml_labels = [it['labels']['label'] for it in ml_items]
        changes = sum(1 for i in range(n) if ml_labels[i] != final[i])
        print(f"  vs ML pipeline: {changes}条不同")

    for i, it in enumerate(items):
        it['labels'] = {'label': final[i]}

    out_path = OUT_DIR / f'{tgt}_hybrid.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"  输出: {out_path}")

print(f"\n完成! 结果在 {OUT_DIR}/")
