#!/usr/bin/env python3
"""
auto_correct.py — 用 LLM (DeepSeek) 自动修正不确定样本的标签
输入: review_500.csv (由 generate_review.py 生成)
       student-XX.json (原始数据, 用于获取完整上下文)
输出: review_500_out.csv (格式兼容 retrain.py)
"""
import json, csv, os, sys, re, time, math
from pathlib import Path
from openai import OpenAI  # DeepSeek 兼容 OpenAI 协议

DATA_DIR = Path("/home/osboxes/Desktop/data-annotation/data")

# 完整的三级标签体系说明 (给 LLM 的 taxonomy)
TAXONOMY = """
心理咨询对话三级标签体系 (必须从中选一个最合适的标签输出):

=== S1 — 轻度心理不适 (日常生活困扰) ===
1.1 学业烦恼: 学业、考研、听课、成绩、考试、毕业、就业、求职、面试、学习、作业、挂科、补考、论文、答辩
1.2 职场烦恼: 工作、同事、老板、加班、绩效、辞职、实习、转正、工资、升职、岗位
1.3 家庭矛盾: 父母、爸妈、家庭、离婚、吵架、亲子、经济压力、观念分歧
1.4 轻度消遣: 喝酒、吸烟、抽烟、棋牌、小酌
1.5 亲友离世: 去世、离世、葬礼、送别、过世、悼念、怀念
1.6 短期失眠: 失眠、睡不着、熬夜、睡眠、夜醒、多梦、睡不好
1.7 现实压力: 压力、焦虑、紧张、烦躁、疲惫、累、没精神
1.8 社交矛盾: 社交、朋友、相处、人际关系、不合群、社恐、内向
1.9 亲密关系: 男朋友、女朋友、恋爱、暗恋、异地、分手、对象、夫妻、婚姻、结婚、相亲、表白、出轨、暧昧
1.10 离异后续: 离异、单亲、抚养权、再婚、后爸、后妈
1.11 分手情绪: 分手、前任、失恋、放不下、复合、挽回
1.12 自我探索: 性格、兴趣、方向、迷茫、自我、人生意义、价值观
1.13 低自尊: 自卑、敏感、在意别人、自我怀疑、不自信
1.14 青春期困扰: 青春期、发育、身体、发育焦虑
1.15 性认知困惑: 性取向、同性、性困惑、性行为
1.16 亲子日常: 教育、管教、叛逆、代沟、孩子不听话
1.17 其他: 难受、难过、不开心、郁闷、烦、无聊

=== S2 — 中度心理障碍 (需要专业干预) ===
2.1 抑郁: 抑郁、想死、活着没意思、绝望、无助、悲伤、哭、情绪低落、自残、没价值
2.2 焦虑症: 焦虑症、惊恐、心慌、心悸、手抖、恐惧、社交恐惧、呼吸困难、胸闷
2.3 双相: 躁郁、双相、情绪波动、亢奋、不睡觉、思维跳跃、冲动消费
2.4 PTSD/创伤: 创伤、阴影、童年、虐待、性侵、家暴、霸凌、噩梦、闪回
2.5 恐慌: 恐慌、濒死、窒息、惊恐发作、急性焦虑
2.6 饮食障碍: 厌食、暴食、催吐、节食、减肥、体重、进食障碍
2.7 强迫: 强迫症、反复、洁癖、检查、控制不住、重复、洗手
2.8 物质成瘾: 酗酒、酒瘾、吸毒、成瘾、网瘾、戒不掉、依赖
2.9 其他精神异常: 幻觉、幻听、妄想、精神分裂、异常、自言自语

=== S3 — 紧急危机 (需要立即干预) ===
3.1 正在自杀: 正在自杀、跳楼、上吊、割腕、服药
3.2 自杀计划: 想自杀、自杀计划、准备死、写遗书
3.3 自残: 自残、划手、割手、烫自己、伤害身体
3.4 伤害他人: 打人、杀人、伤人、持刀、攻击、暴力
3.5 报复: 报复、报仇、杀人计划、干掉、弄死

规则:
- 优先 S3 > S2 > S1
- S3 仅在有明确生命危险信号时使用
- S2 需要出现该障碍的典型症状描述
- S1 用于一般的日常生活困扰
- 选择最具体、最匹配的子类标签
"""


def load_api_key():
    """从环境变量获取 DeepSeek API key"""
    key = os.environ.get('DEEPSEEK_API_KEY')
    if not key:
        print("错误: 未设置 DEEPSEEK_API_KEY 环境变量")
        sys.exit(1)
    return key


def build_prompt_batch(batch):
    """为一批样本构建 LLM 提示词"""
    lines = []
    for item in batch:
        idx = item['idx']
        title = item['title'][:100]
        content = item['content'][:300]
        dialogs = item['dialogs'][:500]
        lines.append(
            f"[{idx}] 标题: {title}\n"
            f"    内容: {content}\n"
            f"    对话: {dialogs}\n"
        )
    prompt = "请为以下每条心理咨询对话选择最合适的分类标签。\n"
    prompt += "请严格按照格式输出, 每行一个: idx → 标签 (如 4673 → 1.9)\n"
    prompt += "只输出标签, 不需要解释。\n\n"
    prompt += "\n---\n".join(lines)
    return prompt


def parse_llm_response(response_text, batch_indices):
    """解析 LLM 返回结果, 返回 {idx: label}"""
    corrections = {}
    for line in response_text.strip().split('\n'):
        line = line.strip()
        # 匹配格式: idx → 1.9  或 idx -> 1.9  或 idx ➡ 1.9
        m = re.match(r'^(\d+)\s*[→➡->]+\s*(\d+\.\d+)', line)
        if m:
            idx = int(m.group(1))
            label = m.group(2)
            if idx in batch_indices:
                corrections[idx] = label
    return corrections


def get_full_content(data, idx):
    """从原始数据中获取完整的 question + dialogs"""
    item = data[idx]
    title = item.get('question_title', '')
    content = item.get('question_content', '')
    dialogs_text = ''
    for ans in item.get('answers', [])[:2]:  # 取前2个回答
        for d in ans.get('dialogs', [])[:3]:  # 每个回答取前3轮对话
            dialogs_text += d.get('content', '') + ' '
    return title, content, dialogs_text.strip()


def correct_batch(client, batch, data):
    """修正一批样本"""
    batch_indices = {item['idx'] for item in batch}
    
    prompt = build_prompt_batch(batch)
    
    try:
        resp = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": f"你是一个心理咨询对话三级标签分类器。请严格按照标签体系输出。\n\n{TAXONOMY}"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2000,
        )
        result = resp.choices[0].message.content
        corrections = parse_llm_response(result, batch_indices)
        return corrections
    except Exception as e:
        print(f"  LLM 调用失败: {e}")
        return {}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='LLM auto-correction of uncertain labels')
    parser.add_argument('--input', required=True, help='review CSV path (from generate_review.py)')
    parser.add_argument('--data', required=True, help='original student JSON path')
    parser.add_argument('--output', required=True, help='output corrected file path')
    parser.add_argument('--batch-size', type=int, default=25, help='samples per LLM call')
    parser.add_argument('--limit', type=int, default=500, help='max samples to correct')
    args = parser.parse_args()

    api_key = load_api_key()
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    # 加载原始数据 (用于完整上下文)
    with open(args.data) as f:
        data = json.load(f)
    print(f"原始数据: {len(data)} 条")

    # 加载 review CSV
    samples = []
    with open(args.input, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row['idx'])
            title, content, dialogs = get_full_content(data, idx)
            samples.append({
                'idx': idx,
                'question_id': row['question_id'],
                'title': title,
                'content': content,
                'dialogs': dialogs,
                'model_label': row.get('model_label', ''),
                'confidence': float(row.get('confidence', 0)),
            })

    # 按置信度升序排序 (最不确定的先修正)
    samples.sort(key=lambda x: x['confidence'])
    samples = samples[:args.limit]
    print(f"待修正: {len(samples)} 条 (batch_size={args.batch_size})")

    # 分批修正
    all_corrections = {}
    total_batches = math.ceil(len(samples) / args.batch_size)
    t0 = time.time()

    for i in range(0, len(samples), args.batch_size):
        batch = samples[i:i + args.batch_size]
        batch_num = i // args.batch_size + 1
        print(f"  批次 {batch_num}/{total_batches} ({len(batch)} 条)...", end=' ', flush=True)

        corrections = correct_batch(client, batch, data)
        all_corrections.update(corrections)
        print(f"→ 成功 {len(corrections)}/{len(batch)} 条")

        # 速率限制
        if batch_num < total_batches:
            time.sleep(1)

    elapsed = time.time() - t0
    print(f"\n修正完成: {len(all_corrections)}/{len(samples)} 条, 耗时 {elapsed:.0f}s")

    # 输出 review_500_out.csv (兼容 retrain.py 格式)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("根据标签体系自动标注结果:\n\n")
        for s in samples:
            idx = s['idx']
            if idx in all_corrections:
                new_label = all_corrections[idx]
                title_short = s['title'][:30]
                f.write(f"{idx} → {new_label}（{title_short}）\n")
            else:
                f.write(f"{idx} → {s['model_label']}（未修正，保留原标签）\n")

    print(f"输出: {args.output}")


if __name__ == '__main__':
    main()
