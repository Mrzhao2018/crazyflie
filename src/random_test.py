"""
random_test.py - 随机仿射压力测试与模块集成测试入口

功能：
1. 随机初始状态 + 多阶段复杂仿射变换 Monte Carlo 压力测试
2. 统一调用主仿真中的场景构造函数，覆盖 AFC / CBF / ESO / ET / RHF / Mission
3. 输出单次轨迹图、Monte Carlo 统计图、模块集成对比图

用法：
    python src/random_test.py
    python src/random_test.py --mode affine
    python src/random_test.py --mode modules
    python src/random_test.py --trials 10 --no-show
"""

import argparse
import os

import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from formation import (
    CRAZYFLIE_COMM,
    affine_transform,
    create_leader_trajectory,
    double_pentagon,
    rotation_matrix_axis,
    rotation_matrix_z,
    scale_matrix,
    shear_matrix_3d,
)
from main_sim import (
    build_base_animation_setup,
    build_baseline_animation_scenario,
    build_cbf_animation_scenario,
    build_eso_animation_scenario,
    build_et_animation_scenario,
    build_rhf_animation_scenario,
    run_pyramid_integrated_mission,
    simulate_first_order,
    simulate_second_order_et,
)


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
for _dir in (OUTPUT_DIR, FIGURE_DIR):
    os.makedirs(_dir, exist_ok=True)


def figure_path(name):
    return os.path.join(FIGURE_DIR, name)


def reflection_matrix_3d(normal):
    """反射矩阵：关于法向量 normal 所定义的平面反射。"""
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    return np.eye(3) - 2.0 * np.outer(normal, normal)


def general_affine_matrix(scale=(1.0, 1.0, 1.0), rotate_axis=None,
                          rotate_angle=0.0, shear=(0.0, 0.0, 0.0)):
    """构造一般仿射矩阵 A = Shear @ Rotate @ Scale。"""
    scale_matrix_local = np.diag(scale)
    if rotate_axis is not None:
        rotate_matrix = rotation_matrix_axis(rotate_axis, rotate_angle)
    else:
        rotate_matrix = np.eye(3)
    shear_matrix_local = shear_matrix_3d(*shear)
    return shear_matrix_local @ rotate_matrix @ scale_matrix_local


TRANSFORM_LIBRARY = {
    'scale_uniform': {
        'name': '均匀缩放 2x',
        'A': lambda: scale_matrix(3, 2.0),
    },
    'scale_nonuniform': {
        'name': '非均匀缩放 (1.5, 0.8, 1.2)',
        'A': lambda: np.diag([1.5, 0.8, 1.2]),
    },
    'rotate_z45': {
        'name': '绕 z 轴旋转 45°',
        'A': lambda: rotation_matrix_z(np.pi / 4),
    },
    'rotate_oblique': {
        'name': '绕 (1,1,1) 轴旋转 60°',
        'A': lambda: rotation_matrix_axis([1, 1, 1], np.pi / 3),
    },
    'shear_xy': {
        'name': '剪切 sxy=0.4, sxz=0.2',
        'A': lambda: shear_matrix_3d(sxy=0.4, sxz=0.2),
    },
    'reflect_xy': {
        'name': '关于 xy 平面反射 (z 翻转)',
        'A': lambda: reflection_matrix_3d([0, 0, 1]),
    },
    'general': {
        'name': '一般仿射: 缩放+旋转+剪切',
        'A': lambda: general_affine_matrix(
            scale=(1.3, 0.9, 1.1),
            rotate_axis=[0, 1, 0],
            rotate_angle=np.pi / 6,
            shear=(0.2, 0.0, 0.15),
        ),
    },
}


def _compute_min_pair_metrics(pos_hist):
    n_steps, n_agents, _ = pos_hist.shape
    pair_indices = np.zeros((n_steps, 2), dtype=int)
    pair_distances = np.zeros(n_steps)
    for step_idx in range(n_steps):
        pos = pos_hist[step_idx]
        best_dist = float('inf')
        best_pair = (0, 1)
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                dist = float(np.linalg.norm(pos[i] - pos[j]))
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (i, j)
        pair_indices[step_idx] = best_pair
        pair_distances[step_idx] = best_dist
    return pair_indices, pair_distances


def random_initial_positions(nominal_pos, sigma=2.0, rng=None):
    """生成完全随机的初始位置。"""
    rng = rng or np.random.default_rng()
    center = nominal_pos.mean(axis=0)
    return center + rng.normal(0.0, sigma, size=nominal_pos.shape)


def run_random_affine_trial(transform_keys, init_sigma=2.0, seed=None,
                            t_settle=12.0, t_trans=5.0, t_hold=6.0,
                            dt=0.02, use_second_order=True):
    """运行一次随机仿射压力测试。"""
    setup = build_base_animation_setup()
    controller = setup['controller']
    nominal_pos = setup['nominal_pos']
    leader_indices = setup['leader_indices']
    rng = np.random.default_rng(seed)

    init_pos = random_initial_positions(nominal_pos, sigma=init_sigma, rng=rng)
    init_vel = np.zeros_like(init_pos)

    nominal_leaders = nominal_pos[leader_indices]
    phases = [{
        'start_positions': nominal_leaders,
        't_start': 0.0,
        't_end': 0.1,
        'positions': nominal_leaders.copy(),
    }]

    transform_info = []
    cumulative_a = np.eye(nominal_pos.shape[1])
    t_cursor = t_settle
    for key in transform_keys:
        transform_entry = TRANSFORM_LIBRARY[key]
        local_a = transform_entry['A']()
        cumulative_a = local_a @ cumulative_a
        target = affine_transform(nominal_leaders, A=cumulative_a)
        phases.append({
            't_start': t_cursor,
            't_end': t_cursor + t_trans,
            'positions': target,
        })
        transform_info.append({
            'key': key,
            'name': transform_entry['name'],
            'A': local_a.copy(),
            'A_cumul': cumulative_a.copy(),
            't_start': t_cursor,
            't_end': t_cursor + t_trans,
        })
        t_cursor += t_trans + t_hold

    leader_traj = create_leader_trajectory(phases)
    total_time = t_cursor

    if use_second_order:
        times, pos_hist, errors, ctrl_inputs, _ = simulate_second_order_et(
            controller,
            init_pos,
            init_vel,
            leader_traj,
            (0.0, total_time),
            dt=dt,
            et_manager=None,
        )
    else:
        times, pos_hist, errors, ctrl_inputs = simulate_first_order(
            controller,
            init_pos,
            leader_traj,
            (0.0, total_time),
            dt=dt,
        )

    final_leaders = leader_traj(total_time)
    follower_target = controller.steady_state(final_leaders)
    follower_final = pos_hist[-1][controller.follower_indices]
    convergence_error = float(np.linalg.norm(follower_final - follower_target))

    transformed_nominal = affine_transform(nominal_pos, A=cumulative_a)
    theoretical_error = float(np.linalg.norm(
        follower_target - transformed_nominal[controller.follower_indices]
    ))
    _, min_distance_hist = _compute_min_pair_metrics(pos_hist)

    return {
        'times': times,
        'positions': pos_hist,
        'errors': errors,
        'control_inputs': ctrl_inputs,
        'init_pos': init_pos,
        'leader_indices': leader_indices,
        'adj': setup['adj'],
        'transforms_info': transform_info,
        'A_cumul': cumulative_a,
        'final_error': float(errors[-1]),
        'convergence_error': convergence_error,
        'affine_invariance_error': theoretical_error,
        'min_distance': float(min_distance_hist.min()),
        'T_total': total_time,
        'seed': seed,
        'use_second_order': use_second_order,
    }


def monte_carlo_affine_test(transform_keys, n_trials=20, init_sigma=2.5,
                            base_seed=100, **kwargs):
    """执行 Monte Carlo 随机仿射测试。"""
    results = []
    for trial_idx in range(n_trials):
        seed = base_seed + trial_idx
        print(f"  Trial {trial_idx + 1}/{n_trials} (seed={seed})...", end='')
        result = run_random_affine_trial(
            transform_keys,
            init_sigma=init_sigma,
            seed=seed,
            **kwargs,
        )
        print(
            f" final_err={result['final_error']:.4f}"
            f" affine_inv={result['affine_invariance_error']:.2e}"
            f" d_min={result['min_distance']:.4f}"
        )
        results.append(result)

    final_errors = np.array([item['final_error'] for item in results])
    affine_errors = np.array([item['affine_invariance_error'] for item in results])
    min_distances = np.array([item['min_distance'] for item in results])
    summary = {
        'n_trials': n_trials,
        'transform_keys': list(transform_keys),
        'final_error_mean': float(final_errors.mean()),
        'final_error_std': float(final_errors.std()),
        'final_error_max': float(final_errors.max()),
        'affine_inv_mean': float(affine_errors.mean()),
        'affine_inv_max': float(affine_errors.max()),
        'min_distance_mean': float(min_distances.mean()),
        'min_distance_min': float(min_distances.min()),
        'all_converged': bool(np.all(final_errors < 2.0)),
    }
    return results, summary


def plot_single_affine_result(result, save_prefix='fig_rt'):
    """绘制单次随机仿射测试结果。"""
    times = result['times']
    pos_hist = result['positions']
    errors = result['errors']
    transforms_info = result['transforms_info']
    leader_indices = result['leader_indices']
    adj = result['adj']
    n_agents = pos_hist.shape[1]
    follower_indices = sorted(set(range(n_agents)) - set(leader_indices))

    fig = plt.figure(figsize=(15, 5))
    ax_err = fig.add_subplot(121)
    ax_3d = fig.add_subplot(122, projection='3d')

    ax_err.plot(times, errors + 1e-12, color='tab:blue', linewidth=1.5)
    ax_err.set_yscale('log')
    ax_err.set_xlabel('时间 (s)')
    ax_err.set_ylabel('编队误差')
    ax_err.set_title('随机仿射测试误差收敛')
    ax_err.grid(True, alpha=0.3)
    phase_colors = plt.colormaps['Set2'](np.linspace(0, 1, max(len(transforms_info), 1)))
    for idx, info in enumerate(transforms_info):
        ax_err.axvspan(info['t_start'], info['t_end'], color=phase_colors[idx], alpha=0.18)
        ax_err.text(
            (info['t_start'] + info['t_end']) * 0.5,
            ax_err.get_ylim()[1] * 0.55,
            info['name'],
            ha='center',
            va='center',
            fontsize=7,
            alpha=0.8,
        )

    init_pos = result['init_pos']
    final_pos = pos_hist[-1]
    ax_3d.scatter(*init_pos[leader_indices].T, c='salmon', s=70, marker='^', alpha=0.35, label='Leader 初始')
    ax_3d.scatter(*init_pos[follower_indices].T, c='lightskyblue', s=35, marker='o', alpha=0.35, label='Follower 初始')
    ax_3d.scatter(*final_pos[leader_indices].T, c='red', s=110, marker='^', edgecolors='black', label='Leader 最终')
    ax_3d.scatter(*final_pos[follower_indices].T, c='dodgerblue', s=55, marker='o', edgecolors='black', label='Follower 最终')
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            if adj[i, j] != 0:
                ax_3d.plot(*zip(final_pos[i], final_pos[j]), color='gray', alpha=0.25, linewidth=0.6)
    ax_3d.set_title('初始(透明) vs 最终编队')
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.legend(fontsize=7, loc='upper left')

    fig.suptitle(
        f'随机仿射单次测试 (seed={result["seed"]}, final_err={result["final_error"]:.4f}, '
        f'd_min={result["min_distance"]:.4f})',
        fontsize=12,
    )
    fig.tight_layout()
    out_path = figure_path(f'{save_prefix}_single.png')
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f'  已保存: {out_path}')
    return fig


def plot_affine_monte_carlo(results, summary, save_prefix='fig_rt'):
    """绘制 Monte Carlo 随机仿射统计结果。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for result in results:
        axes[0].plot(result['times'], result['errors'] + 1e-12, alpha=0.3, linewidth=0.8)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('时间 (s)')
    axes[0].set_ylabel('编队误差')
    axes[0].set_title(f'所有试验误差曲线 (n={summary["n_trials"]})')
    axes[0].grid(True, alpha=0.3)

    final_errors = [item['final_error'] for item in results]
    axes[1].hist(final_errors, bins=min(15, len(results)), edgecolor='black', alpha=0.75)
    axes[1].axvline(np.mean(final_errors), color='red', linestyle='--', label=f'均值={np.mean(final_errors):.4f}')
    axes[1].set_xlabel('最终误差')
    axes[1].set_ylabel('次数')
    axes[1].set_title('最终误差分布')
    axes[1].legend()

    min_distances = [item['min_distance'] for item in results]
    axes[2].bar(range(len(results)), min_distances, color='teal', alpha=0.75)
    axes[2].set_xlabel('试验编号')
    axes[2].set_ylabel('最小间距 (m)')
    axes[2].set_title('随机试验最小间距')
    axes[2].grid(True, alpha=0.3, axis='y')

    fig.suptitle(
        f'Monte Carlo 随机仿射测试统计 | err={summary["final_error_mean"]:.4f}±{summary["final_error_std"]:.4f}',
        fontsize=13,
    )
    fig.tight_layout()
    out_path = figure_path(f'{save_prefix}_monte_carlo.png')
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f'  已保存: {out_path}')
    return fig


def _scenario_summary(name, scenario):
    times = scenario['times']
    pos_hist = scenario['pos_hist']
    errors = scenario['errors']
    _, min_distance_hist = _compute_min_pair_metrics(pos_hist)
    extra = scenario.get('extra', {})

    summary = {
        'name': name,
        'kind': scenario['kind'],
        'n_steps': len(times),
        'final_error': float(errors[-1]),
        'min_distance': float(min_distance_hist.min()),
        'mean_comm_rate_pct': float('nan'),
        'cbf_active_ratio_pct': float('nan'),
        'final_estimation_error': float('nan'),
        'n_switches': float('nan'),
    }

    if scenario['kind'] == 'cbf':
        summary['min_distance'] = float(np.min(extra['min_dist_yes']))
        summary['cbf_active_ratio_pct'] = float(100.0 * np.mean(np.asarray(extra['n_active']) > 0))
    elif scenario['kind'] == 'eso':
        summary['final_estimation_error'] = float(extra['estimation_errors'][-1])
    elif scenario['kind'] == 'et':
        summary['mean_comm_rate_pct'] = float(extra['comm_rates']['mean'])
    elif scenario['kind'] == 'rhf':
        summary['n_switches'] = float(len(extra['switch_times']))
    elif scenario['kind'] == 'mission':
        summary['min_distance'] = float(np.min(extra['min_distances']))
        summary['mean_comm_rate_pct'] = float(extra['comm_rates']['mean'])
        summary['cbf_active_ratio_pct'] = float(100.0 * np.mean(np.asarray(extra['n_active']) > 0))
        summary['final_estimation_error'] = float(extra['estimation_error'][-1])
        summary['n_switches'] = float(len(extra['switch_times']))

    return summary


def run_module_integration_suite():
    """统一执行场景 1-6，形成模块集成测试摘要。"""
    scenarios = [
        ('baseline', build_baseline_animation_scenario()),
        ('cbf', build_cbf_animation_scenario()),
        ('eso', build_eso_animation_scenario()),
        ('et', build_et_animation_scenario()),
        ('rhf', build_rhf_animation_scenario()),
    ]

    mission = run_pyramid_integrated_mission(render_outputs=False, verbose=False)
    mission_scenario = {
        'kind': 'mission',
        'title': '场景6：地面起飞到空中金字塔',
        'output': 'afc_scene6_pyramid_mission',
        'times': mission['times'],
        'pos_hist': mission['positions'],
        'errors': mission['errors'],
        'adj': mission['schedule'][0]['adj'],
        'leader_indices': mission['schedule'][0]['leader_indices'],
        'follower_indices': sorted(set(range(mission['positions'].shape[1])) - set(mission['schedule'][0]['leader_indices'])),
        'phase_specs': [],
        'extra': {
            'schedule': mission['schedule'],
            'switch_times': [item['t_switch'] for item in mission['schedule']],
            'min_distances': mission['data']['min_distances'],
            'n_active': mission['data']['n_active_constraints'],
            'comm_rates': mission['data']['comm_rates'],
            'estimation_error': mission['data']['estimation_error'],
        },
    }
    scenarios.append(('mission', mission_scenario))

    summaries = []
    for name, scenario in scenarios:
        summary = _scenario_summary(name, scenario)
        summaries.append(summary)
        print(
            f"  {name:8s} | final_err={summary['final_error']:.4f}"
            f" | d_min={summary['min_distance']:.4f}"
            f" | comm={summary['mean_comm_rate_pct'] if not np.isnan(summary['mean_comm_rate_pct']) else float('nan'):.2f}"
        )
    return summaries


def plot_module_integration_summary(summaries, save_prefix='fig_rt_modules'):
    """绘制模块集成测试结果摘要。"""
    names = [item['name'] for item in summaries]
    final_errors = [item['final_error'] for item in summaries]
    min_distances = [item['min_distance'] for item in summaries]
    comm_rates = [0.0 if np.isnan(item['mean_comm_rate_pct']) else item['mean_comm_rate_pct'] for item in summaries]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].bar(names, final_errors, color='#4c78a8')
    axes[0].set_title('各场景最终误差')
    axes[0].set_ylabel('最终误差')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(names, min_distances, color='#f58518')
    axes[1].set_title('各场景最小间距')
    axes[1].set_ylabel('最小间距 (m)')
    axes[1].grid(True, alpha=0.3, axis='y')

    axes[2].bar(names, comm_rates, color='#54a24b')
    axes[2].set_title('各场景平均通信率')
    axes[2].set_ylabel('通信率 (%)')
    axes[2].grid(True, alpha=0.3, axis='y')

    fig.suptitle('随机测试模块集成摘要', fontsize=13)
    fig.tight_layout()
    out_path = figure_path(f'{save_prefix}.png')
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f'  已保存: {out_path}')
    return fig


def _build_parser():
    parser = argparse.ArgumentParser(description='随机仿射压力测试与模块集成测试')
    parser.add_argument(
        '--mode',
        choices=['all', 'affine', 'modules'],
        default='all',
        help='all: 全部执行；affine: 仅随机仿射测试；modules: 仅模块集成测试',
    )
    parser.add_argument('--trials', type=int, default=20, help='Monte Carlo 试验次数')
    parser.add_argument('--sigma', type=float, default=2.5, help='随机初始位置标准差')
    parser.add_argument('--no-show', action='store_true', help='生成图片但不弹出 matplotlib 窗口')
    return parser


def main():
    args = _build_parser().parse_args()

    figures = []
    print('=' * 60)
    print('随机仿射压力测试 + 模块集成测试')
    print('=' * 60)

    if args.mode in ('all', 'affine'):
        print('\n[1] 随机仿射单次压力测试...')
        transform_keys = ['scale_nonuniform', 'rotate_oblique', 'shear_xy', 'general']
        print(f"  变换序列: {[TRANSFORM_LIBRARY[key]['name'] for key in transform_keys]}")
        single_result = run_random_affine_trial(
            transform_keys,
            init_sigma=2.0,
            seed=7,
            t_settle=12.0,
            t_trans=5.0,
            t_hold=6.0,
            use_second_order=True,
        )
        print(f"  初始误差: {single_result['errors'][0]:.4f}")
        print(f"  最终误差: {single_result['final_error']:.4f}")
        print(f"  最小间距: {single_result['min_distance']:.4f} m")
        print(f"  仿射不变性误差: {single_result['affine_invariance_error']:.2e}")
        figures.append(plot_single_affine_result(single_result, save_prefix='fig_rt'))

        print(f'\n[2] Monte Carlo 随机仿射测试 ({args.trials} 次)...')
        mc_keys = ['scale_nonuniform', 'rotate_oblique', 'shear_xy']
        mc_results, mc_summary = monte_carlo_affine_test(
            mc_keys,
            n_trials=args.trials,
            init_sigma=args.sigma,
            base_seed=100,
            t_settle=15.0,
            use_second_order=True,
        )
        print(f"  最终误差: {mc_summary['final_error_mean']:.4f} ± {mc_summary['final_error_std']:.4f}")
        print(f"  最大最终误差: {mc_summary['final_error_max']:.4f}")
        print(f"  仿射不变性误差均值: {mc_summary['affine_inv_mean']:.2e}")
        print(f"  平均最小间距: {mc_summary['min_distance_mean']:.4f} m")
        print(f"  全部收敛: {'✓' if mc_summary['all_converged'] else '✗'}")
        figures.append(plot_affine_monte_carlo(mc_results, mc_summary, save_prefix='fig_rt'))

    if args.mode in ('all', 'modules'):
        print('\n[3] 模块集成测试套件...')
        summaries = run_module_integration_suite()
        figures.append(plot_module_integration_summary(summaries, save_prefix='fig_rt_modules'))

    print('\n测试完成！')
    if not args.no_show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)


if __name__ == '__main__':
    main()