"""
tune_pyramid_safety.py - 综合金字塔任务安全参数自动调优脚本

目标：
1. 批量搜索 CBF 安全参数 d_safe / d_activate / gamma
2. 在保持误差与通信开销可接受的前提下，提高最小间距
3. 保存调参结果，便于后续复现实验

用法：
    python src/tune_pyramid_safety.py
    python src/tune_pyramid_safety.py --quick
    python src/tune_pyramid_safety.py --d-safe 0.16,0.17,0.18 --d-activate 0.85,0.9,0.95 --gamma 6,7,8
"""

import argparse
import csv
import itertools
import json
import os
from datetime import datetime

from main_sim import (
    PYRAMID_CONFIG_PATH,
    run_pyramid_integrated_mission,
    save_pyramid_mission_config,
)


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs', 'tuning')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _parse_float_list(value):
    return [float(item.strip()) for item in value.split(',') if item.strip()]


def _build_parser():
    parser = argparse.ArgumentParser(description='综合金字塔任务安全参数调优')
    parser.add_argument('--d-safe', default='0.16,0.17,0.18,0.19',
                        help='候选 d_safe，逗号分隔')
    parser.add_argument('--d-activate', default='0.85,0.9,0.95,1.0',
                        help='候选 d_activate，逗号分隔')
    parser.add_argument('--gamma', default='5.0,6.0,7.0,8.0',
                        help='候选 CBF gamma，逗号分隔')
    parser.add_argument('--max-final-error', type=float, default=0.42,
                        help='允许的最终误差上限')
    parser.add_argument('--max-comm-rate', type=float, default=15.0,
                        help='允许的平均通信率上限(%%)')
    parser.add_argument('--top-k', type=int, default=10,
                        help='输出前 k 个候选')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式，缩小搜索网格用于冒烟验证')
    parser.add_argument('--no-write', action='store_true',
                        help='仅搜索，不自动写回默认配置文件')
    return parser


def _score_result(summary, baseline_min_distance):
    min_distance = summary['min_distance']
    final_error = summary['final_error']
    comm_rate = summary['mean_comm_rate_pct']
    cbf_active = summary['cbf_active_ratio_pct']
    est_error = summary['final_estimation_error']

    improvement = min_distance - baseline_min_distance
    return (
        200.0 * improvement
        + 40.0 * min_distance
        - 12.0 * final_error
        - 0.3 * comm_rate
        - 0.03 * cbf_active
        - 2.0 * est_error
    )


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.quick:
        d_safe_values = [0.16, 0.17, 0.18]
        d_activate_values = [0.9, 0.95]
        gamma_values = [6.0, 7.0]
    else:
        d_safe_values = _parse_float_list(args.d_safe)
        d_activate_values = _parse_float_list(args.d_activate)
        gamma_values = _parse_float_list(args.gamma)

    print('=' * 68)
    print('综合金字塔任务安全参数自动调优')
    print('=' * 68)
    print(f'候选规模: d_safe={len(d_safe_values)}, d_activate={len(d_activate_values)}, gamma={len(gamma_values)}')

    baseline = run_pyramid_integrated_mission(render_outputs=False, verbose=False)
    baseline_summary = baseline['summary']
    baseline_config = baseline['config']
    print('基线参数:')
    print(f"  d_safe={baseline_config['d_safe']:.3f}, d_activate={baseline_config['d_activate']:.3f}, gamma={baseline_config['cbf_gamma']:.3f}")
    print(f"  min_distance={baseline_summary['min_distance']:.4f} m, final_error={baseline_summary['final_error']:.6f}, comm_rate={baseline_summary['mean_comm_rate_pct']:.2f}%")

    results = []
    candidates = list(itertools.product(d_safe_values, d_activate_values, gamma_values))
    total = len(candidates)
    for idx, (d_safe, d_activate, gamma) in enumerate(candidates, start=1):
        config = {
            'd_safe': d_safe,
            'd_activate': d_activate,
            'cbf_gamma': gamma,
        }
        mission = run_pyramid_integrated_mission(
            render_outputs=False,
            verbose=False,
            config=config,
        )
        summary = mission['summary']
        feasible = (
            summary['final_error'] <= args.max_final_error
            and summary['mean_comm_rate_pct'] <= args.max_comm_rate
        )
        score = _score_result(summary, baseline_summary['min_distance'])
        result = {
            'rank': 0,
            'candidate_id': idx,
            'd_safe': d_safe,
            'd_activate': d_activate,
            'cbf_gamma': gamma,
            'min_distance': summary['min_distance'],
            'min_distance_gain': summary['min_distance'] - baseline_summary['min_distance'],
            'final_error': summary['final_error'],
            'mean_comm_rate_pct': summary['mean_comm_rate_pct'],
            'comm_saving_pct': summary['comm_saving_pct'],
            'cbf_active_ratio_pct': summary['cbf_active_ratio_pct'],
            'final_estimation_error': summary['final_estimation_error'],
            'n_switches': summary['n_switches'],
            'feasible': feasible,
            'score': score,
        }
        results.append(result)
        print(f"[{idx:02d}/{total:02d}] d_safe={d_safe:.3f}, d_activate={d_activate:.3f}, gamma={gamma:.3f} -> d_min={summary['min_distance']:.4f} m, err={summary['final_error']:.4f}, comm={summary['mean_comm_rate_pct']:.2f}%, feasible={'Y' if feasible else 'N'}")

    results.sort(key=lambda item: (item['feasible'], item['score'], item['min_distance']), reverse=True)
    for rank, item in enumerate(results, start=1):
        item['rank'] = rank

    best = results[0]
    feasible_results = [item for item in results if item['feasible']]
    chosen = feasible_results[0] if feasible_results else best

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(OUTPUT_DIR, f'pyramid_safety_tuning_{timestamp}.csv')
    json_path = os.path.join(OUTPUT_DIR, f'pyramid_safety_tuning_{timestamp}.json')

    fieldnames = [
        'rank', 'candidate_id', 'd_safe', 'd_activate', 'cbf_gamma',
        'min_distance', 'min_distance_gain', 'final_error', 'mean_comm_rate_pct',
        'comm_saving_pct', 'cbf_active_ratio_pct', 'final_estimation_error',
        'n_switches', 'feasible', 'score',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'baseline': {
                'config': baseline_config,
                'summary': baseline_summary,
            },
            'constraints': {
                'max_final_error': args.max_final_error,
                'max_comm_rate': args.max_comm_rate,
            },
            'recommended': chosen,
            'results': results,
        }, f, indent=2, ensure_ascii=False)

    written_config_path = None
    if not args.no_write:
        tuned_config = baseline_config.copy()
        tuned_config.update({
            'd_safe': chosen['d_safe'],
            'd_activate': chosen['d_activate'],
            'cbf_gamma': chosen['cbf_gamma'],
        })
        written_config_path = save_pyramid_mission_config(tuned_config)

    print('\n推荐参数:')
    print(f"  d_safe={chosen['d_safe']:.3f}, d_activate={chosen['d_activate']:.3f}, gamma={chosen['cbf_gamma']:.3f}")
    print(f"  min_distance={chosen['min_distance']:.4f} m (提升 {chosen['min_distance_gain']:+.4f} m)")
    print(f"  final_error={chosen['final_error']:.6f}, comm_rate={chosen['mean_comm_rate_pct']:.2f}%, feasible={'Y' if chosen['feasible'] else 'N'}")

    print(f'\n已保存 CSV: {csv_path}')
    print(f'已保存 JSON: {json_path}')
    if written_config_path is not None:
        print(f'已自动写回默认参数: {written_config_path}')
    else:
        print(f'未写回默认参数，当前配置文件路径: {PYRAMID_CONFIG_PATH}')

    print('\nTop candidates:')
    for item in results[:max(1, args.top_k)]:
        print(
            f"  #{item['rank']:02d} d_safe={item['d_safe']:.3f}, d_activate={item['d_activate']:.3f}, "
            f"gamma={item['cbf_gamma']:.3f}, d_min={item['min_distance']:.4f} m, "
            f"err={item['final_error']:.4f}, comm={item['mean_comm_rate_pct']:.2f}%, "
            f"score={item['score']:.3f}, feasible={'Y' if item['feasible'] else 'N'}"
        )


if __name__ == '__main__':
    main()