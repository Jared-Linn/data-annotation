#!/usr/bin/env python3
"""
prepare_correction_tasks.py — 从 review CSV 生成修正任务文件
输出: data/{stem}_correction_tasks.json
      每个任务包含 idx, title, content, dialogs 供 LLM 分类
"""
import json, csv, argparse
from pathlib import Path

DATA_DIR = Path("/home/osboxes/Desktop/data-annotation/data")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='原始 student JSON')
    parser.add_argument('--review', required=True, help='review CSV')
    parser.add_argument('--output', help='输出任务文件 (默认 auto)')
    parser.add_argument('--limit', type=int, default=500)
    args = parser.parse_args()

    # 加载原始数据
    with open(args.data) as f:
        raw = json.load(f)

    # 加载 review CSV
    samples = []
    with open(args.review, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row['idx'])
            item = raw[idx]
            samples.append({
                'idx': idx,
                'question_id': item['question_id'],
                'title': item.get('question_title', '')[:100],
                'content': item.get('question_content', '')[:300],
                'dialogs': [],
                'old_label': row.get('model_label', ''),
                'confidence': float(row.get('confidence', 0)),
            })
            # 收集对话
            for ans in item.get('answers', [])[:2]:
                for d in ans.get('dialogs', [])[:3]:
                    samples[-1]['dialogs'].append(d.get('content', '')[:150])

    # 按置信度排序
    samples.sort(key=lambda x: x['confidence'])
    samples = samples[:args.limit]

    # 输出任务文件
    output = args.output or args.data.replace('.json', '_correction_tasks.json')
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    
    print(f"任务文件: {output}")
    print(f"样本数: {len(samples)}")
    print(f"\n运行方式: 用子代理读取此文件, 对每条输出 idx → label")


if __name__ == '__main__':
    main()
