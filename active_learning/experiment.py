#!/usr/bin/env python3
"""主动学习 - 实验运行器"""
import argparse, json, time
from pathlib import Path
from collections import Counter
import numpy as np
from active_learning.sampler import select_samples
from active_learning.train_loop import ActiveLearningLoop

DATA = Path('data')
OUT = Path('data/人工标注')


def run_experiment(strategy, n_rounds=5, n_per_round=300, conf=0.7,
                   weight=5.0, auto=True, label=None):
    """运行一轮完整实验"""
    print(f"\n{'='*60}")
    print(f"实验: strategy={strategy} rounds={n_rounds} n={n_per_round} conf={conf}")
    print(f"{'='*60}")

    loop = ActiveLearningLoop(weight=weight, strategy=strategy)
    loop.initialize()

    results = []
    for rnd in range(n_rounds):
        if len(loop.pool_raw) < n_per_round:
            print(f"池不足，停止 (剩余{len(loop.pool_raw)}条)")
            break
        info = loop.run_round(
            n_select=min(n_per_round, len(loop.pool_raw)),
            conf_threshold=conf,
            auto_label=auto,
        )
        results.append(info)
        if info.get('pseudo_dist'):
            print(f"  伪标签分布: {info['pseudo_dist']}")

    final = results[-1] if results else {}
    n_human = sum(1 for s in loop.source if s == 'human')
    n_pseudo = sum(1 for s in loop.source if s == 'pseudo')
    print(f"\n--- 实验结果: {strategy} ---")
    print(f"最终训练集: {len(loop.train_labels)}条 (人工{n_human} + 伪标签{n_pseudo})")
    print(f"池剩余: {len(loop.pool_raw)}条")
    print(f"最终准确率: {final.get('accuracy', 0):.4f}")
    print(f"类别数: {len(set(loop.train_labels))}/31")

    exp_label = label or strategy
    out_path = OUT / f'active_round{len(results)}_{exp_label}.json'
    print(f"实验保存: {out_path}")

    return {
        'strategy': strategy,
        'n_rounds': len(results),
        'final_train_size': len(loop.train_labels),
        'n_pseudo': n_pseudo,
        'final_accuracy': final.get('accuracy', 0),
        'n_classes': len(set(loop.train_labels)),
        'pool_remaining': len(loop.pool_raw),
        'history': results,
        'loop': loop,
        'out_path': str(out_path),
    }


def compare_strategies(strategies=None, n_rounds=5, n_per_round=300):
    """对比多种策略的效果"""
    if strategies is None:
        strategies = ['least_confidence', 'margin', 'entropy', 'random']

    print(f"\n{'='*60}")
    print(f"策略对比实验 ({n_rounds}轮, 每轮{n_per_round}条)")
    print(f"{'='*60}")

    all_results = []
    for strategy in strategies:
        result = run_experiment(strategy, n_rounds=n_rounds, n_per_round=n_per_round)
        all_results.append(result)

    print(f"\n{'='*60}")
    print(f"策略对比总结")
    print(f"{'='*60}")
    print(f"{'策略':<20} {'训练集':>8} {'伪标签':>8} {'准确率':>8} {'类别':>5} {'池剩余':>8}")
    print("-" * 60)
    for r in sorted(all_results, key=lambda x: -x['final_accuracy']):
        print(f"{r['strategy']:<20} {r['final_train_size']:>8} {r['n_pseudo']:>8} "
              f"{r['final_accuracy']:.4f}  {r['n_classes']:>3}/31 {r['pool_remaining']:>8}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description='主动学习实验')
    parser.add_argument('--mode', default='run', choices=['run', 'compare', 'inspect'])
    parser.add_argument('--strategy', default='least_confidence',
                        choices=['least_confidence', 'margin', 'entropy', 'random'])
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--n', type=int, default=300)
    parser.add_argument('--conf', type=float, default=0.7)
    parser.add_argument('--weight', type=float, default=5.0)

    args = parser.parse_args()

    if args.mode == 'run':
        run_experiment(args.strategy, args.rounds, args.n, args.conf, args.weight)
    elif args.mode == 'compare':
        compare_strategies(n_rounds=args.rounds, n_per_round=args.n)
    elif args.mode == 'inspect':
        loop = ActiveLearningLoop()
        loop.initialize()
        print(f"\n池分析:")
        print(f"  总未标注: {len(loop.pool_raw)}条")
        if loop.pool_raw:
            print(f"  来源: No-01剩余 + No-02 + No-03")
            for i in range(min(3, len(loop.pool_raw))):
                item = loop.pool_raw[i]
                print(f"    [{i}] {item['question_id']}: {item.get('question_title','')[:60]}")


if __name__ == '__main__':
    main()
