#!/usr/bin/env python3
"""
merge_corrections.py — 合并多个子代理的修正结果
用法: python3 merge_corrections.py --output data/student-01_corrections.json data/corrections_*.json
"""
import json, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', help='修正结果 JSON 文件')
    parser.add_argument('--output', required=True, help='合并输出')
    args = parser.parse_args()

    all_corrections = {}
    for fpath in args.files:
        with open(fpath) as f:
            data = json.load(f)
        for item in data:
            idx = int(item['idx'])
            label = item['label']
            if idx in all_corrections and all_corrections[idx] != label:
                print(f"  冲突 idx {idx}: {all_corrections[idx]} vs {label}, 取前者")
            else:
                all_corrections[idx] = label

    result = [{'idx': k, 'label': v} for k, v in sorted(all_corrections.items())]
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"合并完成: {len(result)} 条 → {args.output}")


if __name__ == '__main__':
    main()
