#!/usr/bin/env python3
"""
主动学习运行入口

用法:
    # dry-run: 看看选哪些
    python -m research.active_learning.run --dry-run --n 200

    # 跑3轮entropy策略
    python -m research.active_learning.run --rounds 3 --n 500 --strategy entropy

    # 全自动5轮
    python -m research.active_learning.run --rounds 5 --n 300 --auto
"""
import argparse
from active_learning.train_loop import ActiveLearningLoop


def main():
    parser = argparse.ArgumentParser(description='主动学习循环')
    parser.add_argument('--rounds', type=int, default=3, help='迭代轮数')
    parser.add_argument('--n', type=int, default=300, help='每轮选择样本数')
    parser.add_argument('--strategy', default='least_confidence',
                        choices=['least_confidence', 'margin', 'entropy', 'random'],
                        help='采样策略')
    parser.add_argument('--conf', type=float, default=0.7, help='自动标注置信阈值')
    parser.add_argument('--weight', type=float, default=5.0, help='伪标签权重')
    parser.add_argument('--dry-run', action='store_true', help='仅展示不添加')
    parser.add_argument('--auto', action='store_true', help='全自动模式')

    args = parser.parse_args()

    loop = ActiveLearningLoop(weight=args.weight, strategy=args.strategy)
    loop.initialize()

    for rnd in range(args.rounds):
        if len(loop.pool_raw) < args.n:
            print(f"\n池剩余{len(loop.pool_raw)}条不足{args.n}条，停止")
            break

        info = loop.run_round(
            n_select=min(args.n, len(loop.pool_raw)),
            conf_threshold=args.conf if not args.dry_run else 1.0,
            auto_label=args.auto or not args.dry_run,
        )

    loop.summary()


if __name__ == '__main__':
    main()
