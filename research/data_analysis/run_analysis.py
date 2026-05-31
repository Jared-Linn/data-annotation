#!/usr/bin/env python3
"""数据分析统一入口"""
import argparse, sys

from .class_profile import analyze_keywords, compare_similar_classes
from .confusion_analyzer import analyze_confusion
from .feature_importance import analyze_feature_importance, analyze_s3_keywords
from .distribution_viz import plot_all_distributions


def main():
    parser = argparse.ArgumentParser(description='数据分析')
    parser.add_argument('--mode', default='all',
                        choices=['all', 'keywords', 'confusion', 'features', 's3', 'viz', 'compare'])
    args = parser.parse_args()

    if args.mode in ('all', 'keywords'):
        analyze_keywords()
    if args.mode in ('all', 'compare'):
        compare_similar_classes()
    if args.mode in ('all', 'confusion'):
        analyze_confusion()
    if args.mode in ('all', 'features'):
        analyze_feature_importance()
    if args.mode in ('all', 's3'):
        analyze_s3_keywords()
    if args.mode in ('all', 'viz'):
        plot_all_distributions()

    print("\n分析完成!")


if __name__ == '__main__':
    main()
