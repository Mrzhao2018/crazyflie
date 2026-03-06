"""
random_test.py - 随机初始状态 + 复杂仿射变换独立测试模块

测试内容：
1. 完全随机初始位置（不依赖标称构型）
2. 多种仿射变换：缩放、旋转、剪切、反射、一般仿射组合
3. 多次 Monte Carlo 随机试验
4. 自动验证仿射不变性

用法：
    在仓库根目录运行：
    python src/random_test.py

    或者在 src 目录运行：
    python random_test.py
"""

import os

import numpy as np
import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from stress_matrix import compute_stress_matrix, compute_sparse_stress_matrix
from formation import (
    double_pentagon, affine_transform, scale_matrix,
    rotation_matrix_z, rotation_matrix_axis, shear_matrix_3d,
    create_leader_trajectory, CRAZYFLIE_COMM,
)
from afc_controller import AFCController

# 中文显示
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


# ============================================================
# 复杂仿射变换构建
# ============================================================

def reflection_matrix_3d(normal):
    """反射矩阵：关于法向量 normal 所定义的平面反射。"""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    return np.eye(3) - 2.0 * np.outer(n, n)


def general_affine_matrix(scale=(1, 1, 1), rotate_axis=None, rotate_angle=0.0,
                          shear=(0, 0, 0)):
    """
    组合一般仿射变换矩阵 A = Shear @ Rotate @ Scale。

    Parameters
    ----------
    scale : tuple (sx, sy, sz)
    rotate_axis : ndarray (3,) or None
    rotate_angle : float (rad)
    shear : tuple (sxy, sxz, syz)

    Returns
    -------
    A : ndarray (3, 3)
    """
    S = np.diag(scale)
    if rotate_axis is not None:
        axis = np.asarray(rotate_axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(rotate_angle) * K + (1 - np.cos(rotate_angle)) * (K @ K)
    else:
        R = np.eye(3)
    H = shear_matrix_3d(*shear)
    return H @ R @ S


# ============================================================
# 预定义的复杂变换阶段
# ============================================================

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
            rotate_axis=[0, 1, 0], rotate_angle=np.pi / 6,
            shear=(0.2, 0, 0.15),
        ),
    },
}


# ============================================================
# 随机初始状态生成
# ============================================================

def random_initial_positions(nominal_pos, sigma=2.0, rng=None):
    """
    生成完全随机的初始位置。

    Parameters
    ----------
    nominal_pos : ndarray (n, d)  标称位置（仅用于确定形状）
    sigma : float  随机扰动标准差
    rng : np.random.Generator or None

    Returns
    -------
    init_pos : ndarray (n, d)
    """
    if rng is None:
        rng = np.random.default_rng()
    n, d = nominal_pos.shape
    # 以标称中心为基准，加大范围高斯随机
    center = nominal_pos.mean(axis=0)
    return center + rng.normal(0, sigma, size=(n, d))


# ============================================================
# 单次仿射变换测试仿真
# ============================================================

def run_single_test(controller, nominal_pos, leader_indices,
                    transform_keys, init_sigma=2.0,
                    seed=None, T_settle=12.0, T_trans=5.0, T_hold=6.0,
                    dt=0.02, use_second_order=True):
    """
    运行一次随机初始状态 + 多阶段复杂仿射变换测试。

    Parameters
    ----------
    controller : AFCController
    nominal_pos : ndarray (n, d)
    leader_indices : list
    transform_keys : list of str  TRANSFORM_LIBRARY 中的 key 序列
    init_sigma : float  初始位置随机程度
    seed : int or None
    T_settle : float  初始收敛时间
    T_trans : float  每个变换过渡时间
    T_hold : float  每个变换保持时间
    dt : float
    use_second_order : bool  使用二阶积分器

    Returns
    -------
    result : dict
        times, positions, errors, transforms_info, affine_invariance_error
    """
    rng = np.random.default_rng(seed)
    n, d = nominal_pos.shape
    n_l = len(leader_indices)

    # 随机初始位置
    init_pos = random_initial_positions(nominal_pos, sigma=init_sigma, rng=rng)

    # 构建变换序列
    nominal_leaders = nominal_pos[leader_indices]
    transforms_info = []
    A_cumul = np.eye(d)  # 累积变换矩阵

    phases = [{
        'start_positions': nominal_leaders,
        't_start': 0.0, 't_end': 0.1,
        'positions': nominal_leaders.copy(),
    }]

    t_cursor = T_settle
    for key in transform_keys:
        entry = TRANSFORM_LIBRARY[key]
        A_new = entry['A']()
        A_cumul = A_new @ A_cumul
        target = affine_transform(nominal_leaders, A=A_cumul)

        phases.append({
            't_start': t_cursor,
            't_end': t_cursor + T_trans,
            'positions': target,
        })
        transforms_info.append({
            'key': key, 'name': entry['name'],
            'A': A_new.copy(), 'A_cumul': A_cumul.copy(),
            't_start': t_cursor, 't_end': t_cursor + T_trans,
        })
        t_cursor += T_trans + T_hold

    leader_traj = create_leader_trajectory(phases)
    T_total = t_cursor

    # 仿真
    from main_sim import simulate_second_order_et, simulate_first_order
    if use_second_order:
        init_vel = np.zeros_like(init_pos)
        times, pos_hist, errors, ctrl, _ = simulate_second_order_et(
            controller, init_pos, init_vel, leader_traj,
            (0, T_total), dt=dt, et_manager=None,
        )
    else:
        times, pos_hist, errors, ctrl = simulate_first_order(
            controller, init_pos, leader_traj, (0, T_total), dt=dt,
        )

    # 仿射不变性验证
    p_l_final = leader_traj(T_total)
    p_f_star = controller.steady_state(p_l_final)
    p_f_final = pos_hist[-1][controller.follower_indices]
    affine_err = np.linalg.norm(p_f_final - p_f_star)

    # 理论目标位置验证（Ar+b）
    p_all_target = affine_transform(nominal_pos, A=A_cumul)
    p_target_leaders = p_all_target[leader_indices]
    p_target_followers = p_all_target[controller.follower_indices]
    theory_err = np.linalg.norm(
        p_f_star - p_target_followers
    )

    return {
        'times': times,
        'positions': pos_hist,
        'errors': errors,
        'control_inputs': ctrl,
        'init_pos': init_pos,
        'transforms_info': transforms_info,
        'A_cumul': A_cumul,
        'final_error': errors[-1],
        'affine_invariance_error': theory_err,
        'convergence_error': affine_err,
        'T_total': T_total,
        'seed': seed,
    }


# ============================================================
# Monte Carlo 批量测试
# ============================================================

def monte_carlo_test(controller, nominal_pos, leader_indices,
                     transform_keys, n_trials=20, init_sigma=2.0,
                     base_seed=0, **kwargs):
    """
    多次随机初始状态 Monte Carlo 测试。

    Returns
    -------
    results : list of dict
    summary : dict  统计摘要
    """
    results = []
    for i in range(n_trials):
        seed = base_seed + i
        print(f"  Trial {i+1}/{n_trials} (seed={seed})...", end='')
        r = run_single_test(
            controller, nominal_pos, leader_indices,
            transform_keys, init_sigma=init_sigma, seed=seed,
            **kwargs,
        )
        print(f"  final_err={r['final_error']:.4f}  "
              f"affine_inv={r['affine_invariance_error']:.2e}")
        results.append(r)

    final_errs = [r['final_error'] for r in results]
    inv_errs = [r['affine_invariance_error'] for r in results]
    summary = {
        'n_trials': n_trials,
        'final_error_mean': np.mean(final_errs),
        'final_error_std': np.std(final_errs),
        'final_error_max': np.max(final_errs),
        'affine_inv_mean': np.mean(inv_errs),
        'affine_inv_max': np.max(inv_errs),
        'all_converged': all(e < 2.0 for e in final_errs),
        'transform_keys': transform_keys,
    }
    return results, summary


# ============================================================
# 可视化
# ============================================================

def plot_single_result(result, leader_indices, adj, save_prefix=''):
    """绘制单次测试结果：3D 轨迹 + 误差曲线 + 变换标注。"""
    times = result['times']
    pos_hist = result['positions']
    errors = result['errors']
    info_list = result['transforms_info']
    n = pos_hist.shape[1]
    f_idx = sorted(set(range(n)) - set(leader_indices))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：误差收敛曲线
    ax = axes[0]
    ax.plot(times, errors, 'b-', linewidth=1.5)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('编队误差 ||p_f - p_f*||')
    ax.set_title('编队误差收敛')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    # 标注变换阶段
    colors = plt.colormaps['Set2'](np.linspace(0, 1, max(len(info_list), 1)))
    for k, info in enumerate(info_list):
        ax.axvspan(info['t_start'], info['t_end'],
                   alpha=0.2, color=colors[k], label=info['name'])
    if info_list:
        ax.legend(fontsize=7, loc='upper right')

    # 右图: 初始 vs 最终编队 3D
    ax3 = fig.add_subplot(122, projection='3d')
    axes[1].set_visible(False)  # 隐藏 2D 占位

    init = result['init_pos']
    final = pos_hist[-1]
    ax3.scatter(*init[leader_indices].T, c='red', s=80,
                marker='^', label='Leader 初始', alpha=0.4)
    ax3.scatter(*init[f_idx].T, c='blue', s=40,
                marker='o', label='Follower 初始', alpha=0.4)
    ax3.scatter(*final[leader_indices].T, c='red', s=120,
                marker='^', label='Leader 最终', edgecolors='k')
    ax3.scatter(*final[f_idx].T, c='blue', s=60,
                marker='o', label='Follower 最终', edgecolors='k')
    # 连线
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                ax3.plot(*zip(final[i], final[j]), 'gray', alpha=0.3, lw=0.5)
    ax3.set_title('初始(透明) vs 最终编队')
    ax3.legend(fontsize=7, loc='upper left')

    fig.suptitle(f'随机测试 (seed={result["seed"]}, '
                 f'final_err={result["final_error"]:.4f})', fontsize=12)
    fig.tight_layout()

    if save_prefix:
        path = figure_path(f'{save_prefix}_single.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  已保存: {path}")
    return fig


def plot_monte_carlo(results, summary, save_prefix=''):
    """绘制 Monte Carlo 统计结果。"""
    n_trials = len(results)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1) 所有试验的误差曲线叠加
    ax = axes[0]
    for r in results:
        ax.plot(r['times'], r['errors'], alpha=0.3, linewidth=0.8)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('编队误差')
    ax.set_title(f'所有试验误差曲线 (n={n_trials})')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # 2) 最终误差分布
    ax = axes[1]
    final_errs = [r['final_error'] for r in results]
    ax.hist(final_errs, bins=min(15, n_trials), edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(final_errs), color='red', linestyle='--',
               label=f'均值={np.mean(final_errs):.4f}')
    ax.set_xlabel('最终编队误差')
    ax.set_ylabel('次数')
    ax.set_title('最终误差分布')
    ax.legend()

    # 3) 仿射不变性误差
    ax = axes[2]
    inv_errs = [r['affine_invariance_error'] for r in results]
    ax.bar(range(n_trials), inv_errs, color='green', alpha=0.7)
    ax.set_xlabel('试验编号')
    ax.set_ylabel('仿射不变性误差')
    ax.set_title('仿射不变性验证 (理论应≈0)')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))

    fig.suptitle(f'Monte Carlo 随机测试统计 (n={n_trials})', fontsize=13)
    fig.tight_layout()

    if save_prefix:
        path = figure_path(f'{save_prefix}_monte_carlo.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  已保存: {path}")
    return fig


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print("随机初始状态 + 复杂仿射变换 独立测试")
    print("=" * 60)

    # --- 编队与应力矩阵 ---
    print("\n[1] 构建编队与应力矩阵...")
    nominal_pos, leader_indices, adj_full = double_pentagon()
    n, d = nominal_pos.shape

    Omega, sparse_info = compute_sparse_stress_matrix(
        nominal_pos, leader_indices,
        comm_range=CRAZYFLIE_COMM['p2p_range'],
        max_degree=CRAZYFLIE_COMM['max_neighbors'],
    )
    adj = sparse_info['sparse_adj']
    print(f"  边数: {sparse_info['n_edges_sparse']}, "
          f"lambda_min={sparse_info['min_eig_ff']:.6f}")

    gain = 5.0
    K_d = 2.0
    controller = AFCController(Omega, leader_indices, gain=gain,
                                damping=K_d, u_max=1.0,
                                saturation_type='smooth')

    # --- 测试1: 单次复杂变换序列 ---
    print("\n[2] 单次复杂变换序列测试...")
    complex_keys = ['scale_nonuniform', 'rotate_oblique',
                    'shear_xy', 'general']
    print(f"  变换序列: {[TRANSFORM_LIBRARY[k]['name'] for k in complex_keys]}")

    result = run_single_test(
        controller, nominal_pos, leader_indices,
        complex_keys, init_sigma=2.0, seed=7,
        T_settle=12.0, T_trans=5.0, T_hold=6.0,
        use_second_order=True,
    )

    print(f"\n  === 单次结果 ===")
    print(f"  初始误差: {result['errors'][0]:.4f}")
    print(f"  最终误差: {result['final_error']:.4f}")
    print(f"  仿射不变性误差: {result['affine_invariance_error']:.2e}")
    print(f"  累积变换矩阵 A:")
    print(np.array2string(result['A_cumul'], precision=4, suppress_small=True))

    fig1 = plot_single_result(result, leader_indices, adj,
                              save_prefix='fig_rt')

    # --- 测试2: Monte Carlo 随机测试 ---
    print("\n[3] Monte Carlo 随机测试 (20 次)...")
    mc_keys = ['scale_nonuniform', 'rotate_oblique', 'shear_xy']
    results, summary = monte_carlo_test(
        controller, nominal_pos, leader_indices,
        mc_keys, n_trials=20, init_sigma=2.5, base_seed=100,
        T_settle=15.0, use_second_order=True,
    )

    print(f"\n  === Monte Carlo 统计 ===")
    print(f"  试验次数: {summary['n_trials']}")
    print(f"  最终误差: {summary['final_error_mean']:.4f} "
          f"± {summary['final_error_std']:.4f}")
    print(f"  最终误差最大: {summary['final_error_max']:.4f}")
    print(f"  仿射不变性误差均值: {summary['affine_inv_mean']:.2e}")
    print(f"  全部收敛: {'✓' if summary['all_converged'] else '✗'}")

    fig2 = plot_monte_carlo(results, summary, save_prefix='fig_rt')

    # --- 测试3: 变换类型逐个对比 ---
    print("\n[4] 各变换类型单独测试...")
    for key, entry in TRANSFORM_LIBRARY.items():
        r = run_single_test(
            controller, nominal_pos, leader_indices,
            [key], init_sigma=1.5, seed=42,
            T_settle=12.0, use_second_order=True,
        )
        det_A = np.linalg.det(entry['A']())
        print(f"  {entry['name']:30s}  "
              f"final_err={r['final_error']:.4f}  "
              f"det(A)={det_A:+.3f}  "
              f"affine_inv={r['affine_invariance_error']:.2e}")

    print("\n测试完成！")
    plt.show()


if __name__ == '__main__':
    main()
