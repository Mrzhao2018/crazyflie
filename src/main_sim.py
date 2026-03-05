"""
main_sim.py - 仿射编队控制 (AFC) 完整仿真与可视化

本脚本演示仿射编队控制的完整流程：
1. 定义 10 无人机双层正五边形编队
2. 计算应力矩阵 Ω
3. 验证应力矩阵性质
4. 仿真三个阶段：
   Phase 1: 编队建立（从随机位置收敛到标称编队）
   Phase 2: 缩放变换（编队整体放大 1.5 倍）
   Phase 3: 旋转变换（编队绕 z 轴旋转 45°）
5. 生成可视化图表用于毕设论文

用法：
    cd e:/crazyflie/src
    python main_sim.py
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

from stress_matrix import (compute_stress_matrix, validate_stress_matrix,
                           print_validation, compute_sparse_stress_matrix)
from formation import (
    double_pentagon, affine_transform, scale_matrix,
    rotation_matrix_z, create_leader_trajectory, graph_info, smoothstep,
    CRAZYFLIE_COMM,
)
from afc_controller import AFCController

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120


# ============================================================
# 仿真引擎
# ============================================================

def simulate_first_order(controller, initial_positions, leader_traj_func,
                         t_span, dt=0.01):
    """
    一阶积分器模型仿真。

    动力学：ṗ_f = -K_p (Ω_ff p_f + Ω_fl p_l(t))
    Leader 按 leader_traj_func(t) 运动。

    Parameters
    ----------
    controller : AFCController
    initial_positions : ndarray (n, d)
    leader_traj_func : callable
        输入时间 t，返回 leader 位置 ndarray (n_l, d)
    t_span : (float, float)
    dt : float

    Returns
    -------
    times : ndarray (n_steps,)
    positions_history : ndarray (n_steps, n, d)
    errors : ndarray (n_steps,)
    control_inputs : ndarray (n_steps, n_f, d)
        各时刻跟随者控制输入
    """
    n = controller.n
    d = initial_positions.shape[1]
    f_idx = controller.follower_indices
    l_idx = controller.leader_indices
    n_f = controller.n_f

    Omega_ff = controller.Omega_ff
    Omega_fl = controller.Omega_fl

    def dynamics(t, state):
        p_f = state.reshape(n_f, d)
        p_l = leader_traj_func(t)
        u_raw = -controller.gain * (Omega_ff @ p_f + Omega_fl @ p_l)
        u_sat = controller.saturate(u_raw)
        return u_sat.flatten()

    p_f_0 = initial_positions[f_idx].flatten()
    t_eval = np.arange(t_span[0], t_span[1], dt)
    if t_eval[-1] < t_span[1]:
        t_eval = np.append(t_eval, t_span[1])

    sol = solve_ivp(dynamics, t_span, p_f_0, t_eval=t_eval, method='RK45',
                    max_step=dt, rtol=1e-8, atol=1e-10)

    # 重建完整轨迹
    n_steps = len(sol.t)
    positions_history = np.zeros((n_steps, n, d))
    errors = np.zeros(n_steps)
    control_inputs = np.zeros((n_steps, n_f, d))

    for idx, t in enumerate(sol.t):
        p_l = leader_traj_func(t)
        positions_history[idx, l_idx] = p_l
        positions_history[idx, f_idx] = sol.y[:, idx].reshape(n_f, d)

        # 计算编队误差
        p_f_star = controller.steady_state(p_l)
        p_f_cur = positions_history[idx, f_idx]
        errors[idx] = np.linalg.norm(p_f_cur - p_f_star)

        # 记录控制输入
        u_raw = -controller.gain * (Omega_ff @ p_f_cur + Omega_fl @ p_l)
        control_inputs[idx] = controller.saturate(u_raw)

    return sol.t, positions_history, errors, control_inputs


def simulate_second_order(controller, initial_positions, initial_velocities,
                          leader_traj_func, t_span, dt=0.01):
    """
    二阶积分器模型仿真。

    动力学：p̈_f = -K_p (Ω_ff p_f + Ω_fl p_l(t)) - K_d ṗ_f
    """
    n = controller.n
    d = initial_positions.shape[1]
    f_idx = controller.follower_indices
    l_idx = controller.leader_indices
    n_f = controller.n_f

    Omega_ff = controller.Omega_ff
    Omega_fl = controller.Omega_fl

    def dynamics(t, state):
        p_f = state[:n_f * d].reshape(n_f, d)
        v_f = state[n_f * d:].reshape(n_f, d)
        p_l = leader_traj_func(t)

        acc_raw = -controller.gain * (Omega_ff @ p_f + Omega_fl @ p_l) \
              - controller.damping * v_f
        acc = controller.saturate(acc_raw)
        return np.concatenate([v_f.flatten(), acc.flatten()])

    p_f_0 = initial_positions[f_idx].flatten()
    v_f_0 = initial_velocities[f_idx].flatten() if initial_velocities is not None \
            else np.zeros(n_f * d)
    state_0 = np.concatenate([p_f_0, v_f_0])

    t_eval = np.arange(t_span[0], t_span[1], dt)
    if t_eval[-1] < t_span[1]:
        t_eval = np.append(t_eval, t_span[1])

    sol = solve_ivp(dynamics, t_span, state_0, t_eval=t_eval, method='RK45',
                    max_step=dt, rtol=1e-8, atol=1e-10)

    n_steps = len(sol.t)
    positions_history = np.zeros((n_steps, n, d))
    errors = np.zeros(n_steps)
    control_inputs = np.zeros((n_steps, n_f, d))

    for idx, t in enumerate(sol.t):
        p_l = leader_traj_func(t)
        positions_history[idx, l_idx] = p_l
        positions_history[idx, f_idx] = sol.y[:n_f * d, idx].reshape(n_f, d)

        p_f_star = controller.steady_state(p_l)
        p_f_cur = positions_history[idx, f_idx]
        errors[idx] = np.linalg.norm(p_f_cur - p_f_star)

        v_f_cur = sol.y[n_f * d:, idx].reshape(n_f, d)
        u_raw = -controller.gain * (Omega_ff @ p_f_cur + Omega_fl @ p_l) \
                - controller.damping * v_f_cur
        control_inputs[idx] = controller.saturate(u_raw)

    return sol.t, positions_history, errors, control_inputs


# ============================================================
# 可视化工具
# ============================================================

def plot_formation_3d(ax, positions, leader_indices, adj_matrix=None,
                      title="", alpha=1.0, show_labels=True):
    """在给定 Axes3D 上绘制编队快照。"""
    n = positions.shape[0]
    follower_indices = sorted(set(range(n)) - set(leader_indices))

    # 绘制通信边
    if adj_matrix is not None:
        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j] > 0:
                    ax.plot3D(
                        [positions[i, 0], positions[j, 0]],
                        [positions[i, 1], positions[j, 1]],
                        [positions[i, 2], positions[j, 2]],
                        'gray', alpha=0.15, linewidth=0.5
                    )

    # Leader（红色三角形）
    l_pos = positions[leader_indices]
    ax.scatter3D(l_pos[:, 0], l_pos[:, 1], l_pos[:, 2],
                 c='red', s=100, marker='^', label='Leader',
                 alpha=alpha, edgecolors='darkred', linewidths=0.5)

    # Follower（蓝色圆点）
    f_pos = positions[follower_indices]
    ax.scatter3D(f_pos[:, 0], f_pos[:, 1], f_pos[:, 2],
                 c='dodgerblue', s=60, marker='o', label='Follower',
                 alpha=alpha, edgecolors='navy', linewidths=0.5)

    # 标注编号
    if show_labels:
        for i in range(n):
            ax.text(positions[i, 0], positions[i, 1], positions[i, 2],
                    f'  {i}', fontsize=7, alpha=0.8)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    if title:
        ax.set_title(title, fontsize=11)


def plot_trajectories_3d(ax, times, positions_history, leader_indices):
    """绘制所有智能体的 3D 轨迹。"""
    n = positions_history.shape[1]
    follower_indices = sorted(set(range(n)) - set(leader_indices))

    for i in leader_indices:
        ax.plot3D(
            positions_history[:, i, 0],
            positions_history[:, i, 1],
            positions_history[:, i, 2],
            'r-', linewidth=1.2, alpha=0.7
        )

    for i in follower_indices:
        ax.plot3D(
            positions_history[:, i, 0],
            positions_history[:, i, 1],
            positions_history[:, i, 2],
            'b-', linewidth=0.8, alpha=0.5
        )

    # 标注起点和终点
    for i in range(n):
        c = 'red' if i in leader_indices else 'dodgerblue'
        m = '^' if i in leader_indices else 'o'
        ax.scatter3D(*positions_history[0, i], c='gray', s=30, marker='x', alpha=0.5)
        ax.scatter3D(*positions_history[-1, i], c=c, s=60, marker=m, alpha=0.9)


def plot_error_convergence(ax, times, errors, title="编队误差收敛"):
    """绘制编队误差随时间的收敛曲线。"""
    ax.semilogy(times, errors + 1e-16, 'b-', linewidth=1.5)
    ax.set_xlabel('时间 (s)')
    ax.set_ylabel('编队误差 ||p_f - p_f*||')
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([times[0], times[-1]])


# ============================================================
# 主仿真流程
# ============================================================

def main():
    print("=" * 60)
    print("仿射编队控制 (AFC) 仿真 - 10 无人机双层正五边形编队")
    print("=" * 60)

    # ----------------------------------------------------------
    # Step 1: 定义编队构型
    # ----------------------------------------------------------
    print("\n[Step 1] 定义编队构型...")
    nominal_pos, leader_indices, adj_complete = double_pentagon(radius=1.0, height=1.0)
    follower_indices = sorted(set(range(10)) - set(leader_indices))
    d = 3  # 三维空间

    print("  标称编队: 双层正五边形 (10 agents, 3D)")
    print(f"  Leaders: {leader_indices}")
    print(f"  Followers: {follower_indices}")
    print("  完全图基准:")
    graph_info(adj_complete)

    # ----------------------------------------------------------
    # Step 2: 稀疏应力矩阵设计（Crazyflie P2P 通信约束）
    # ----------------------------------------------------------
    print("\n[Step 2] 稀疏应力矩阵设计 (Crazyflie P2P 约束)...")
    print(f"  Crazyflie P2P 通信范围: {CRAZYFLIE_COMM['p2p_range']}m")
    print(f"  单机最大邻居数: {CRAZYFLIE_COMM['max_neighbors']}")
    print(f"  无线芯片: {CRAZYFLIE_COMM['radio_chip']}")
    print(f"  控制频率: {CRAZYFLIE_COMM['control_freq_hz']}Hz")

    Omega, info = compute_sparse_stress_matrix(
        nominal_pos, leader_indices,
        comm_range=CRAZYFLIE_COMM['p2p_range'],
        max_degree=CRAZYFLIE_COMM['max_neighbors'],
        convergence_ratio=0.5,
    )
    adj = info['sparse_adj']

    print(f"\n  === 稀疏设计结果 ===")
    print(f"  求解方法: {info['method']}")
    print(f"  零空间维数: {info['null_dim']}")
    print(f"  稀疏 λ_min(Ω_ff): {info['min_eig_ff']:.6f}")
    print(f"  密集基准 λ_min: {info['t_max_dense']:.6f}")
    print(f"  边数: {info['n_edges_sparse']}/{info['n_edges_complete']} "
          f"(削减 {info['edge_reduction_pct']:.1f}%)")
    print(f"  度数: min={info['degree_min']}, max={info['degree_max']}, "
          f"mean={info['degree_mean']:.1f}")
    print("  稀疏图拓扑:")
    graph_info(adj)

    # ----------------------------------------------------------
    # Step 3: 验证应力矩阵
    # ----------------------------------------------------------
    print("\n[Step 3] 验证应力矩阵...")
    results = validate_stress_matrix(Omega, nominal_pos, leader_indices)
    print_validation(results)

    if not results['全部通过']:
        print("\n[ERROR] 应力矩阵验证未通过，请检查编队定义和通信图。")
        return

    # ----------------------------------------------------------
    # Step 4: 创建控制器
    # ----------------------------------------------------------
    print("\n[Step 4] 创建 AFC 控制器...")
    gain = 5.0
    u_max = CRAZYFLIE_COMM['max_velocity']
    controller = AFCController(Omega, leader_indices, gain=gain,
                               u_max=u_max, saturation_type='smooth')

    rate, tau = controller.convergence_rate_bound()
    print(f"  比例增益 K_p = {gain}")
    print(f"  输入饱和上限 u_max = {u_max} m/s (smooth tanh)")
    print(f"  理论收敛速率 (无饱和): {rate:.4f}")
    print(f"  时间常数 τ (无饱和): {tau:.4f} s")

    # ----------------------------------------------------------
    # Step 5: 设计 Leader 轨迹（三阶段仿射变换）
    # ----------------------------------------------------------
    print("\n[Step 5] 设计 Leader 轨迹...")
    nominal_leaders = nominal_pos[leader_indices]

    # Phase 1: 编队建立（Leader 从标称位置起飞到高度 1m）
    # Phase 2: 缩放 1.5 倍
    # Phase 3: 绕 z 轴旋转 45°

    A_scale = scale_matrix(3, 1.5)
    A_rotate = rotation_matrix_z(np.pi / 4)

    pos_phase1 = nominal_leaders.copy()
    pos_phase2 = affine_transform(nominal_leaders, A=A_scale)
    pos_phase3 = affine_transform(nominal_leaders, A=A_rotate @ A_scale)

    t_settle = 8.0    # 初始收敛时间
    t_trans = 5.0     # 每次变换的过渡时间
    t_hold = 8.0      # 每次变换后的保持时间

    phases = [
        {
            'start_positions': nominal_leaders,
            't_start': 0.0,
            't_end': 0.1,              # Leader 已经在标称位置
            'positions': pos_phase1,
        },
        {
            't_start': t_settle,
            't_end': t_settle + t_trans,
            'positions': pos_phase2,     # 缩放
        },
        {
            't_start': t_settle + t_trans + t_hold,
            't_end': t_settle + 2 * t_trans + t_hold,
            'positions': pos_phase3,     # 旋转（在缩放基础上）
        },
    ]

    leader_traj = create_leader_trajectory(phases)
    T_total = t_settle + 2 * t_trans + 2 * t_hold

    print(f"  Phase 1: 编队建立 (0 ~ {t_settle}s)")
    print(f"  Phase 2: 缩放 1.5x ({t_settle} ~ {t_settle + t_trans}s)")
    print(f"  Phase 3: 旋转 45° ({t_settle + t_trans + t_hold} ~ "
          f"{t_settle + 2 * t_trans + t_hold}s)")
    print(f"  总仿真时长: {T_total}s")

    # ----------------------------------------------------------
    # Step 6: 运行仿真
    # ----------------------------------------------------------
    print("\n[Step 6] 运行仿真...")

    # 初始位置：标称位置 + 随机扰动（模拟起飞后的初始误差）
    np.random.seed(42)
    init_pos = nominal_pos.copy()
    init_pos[follower_indices] += np.random.randn(len(follower_indices), d) * 0.5

    dt = 0.02
    times, pos_hist, errors, ctrl_inputs = simulate_first_order(
        controller, init_pos, leader_traj, (0, T_total), dt=dt
    )
    print(f"  仿真完成 (饱和): {len(times)} 步")
    print(f"  初始编队误差: {errors[0]:.4f}")
    print(f"  最终编队误差: {errors[-1]:.6f}")

    # 无饱和对照仿真
    controller_nosat = AFCController(Omega, leader_indices, gain=gain,
                                     u_max=None)
    times_ns, pos_hist_ns, errors_ns, ctrl_inputs_ns = simulate_first_order(
        controller_nosat, init_pos, leader_traj, (0, T_total), dt=dt
    )
    print(f"  仿真完成 (无饱和): {len(times_ns)} 步")
    print(f"  最终编队误差 (无饱和): {errors_ns[-1]:.6f}")

    # 控制输入统计
    ctrl_norms = np.linalg.norm(ctrl_inputs, axis=2)         # (n_steps, n_f)
    ctrl_norms_ns = np.linalg.norm(ctrl_inputs_ns, axis=2)   # (n_steps, n_f)
    print(f"\n  控制输入统计 (饱和):")
    print(f"    最大 ||u||: {ctrl_norms.max():.4f} m/s")
    print(f"    饱和上限: {u_max} m/s")
    print(f"  控制输入统计 (无饱和):")
    print(f"    最大 ||u||: {ctrl_norms_ns.max():.4f} m/s")

    # ----------------------------------------------------------
    # Step 7: 可视化
    # ----------------------------------------------------------
    print("\n[Step 7] 生成可视化图表...")

    # ==== 图1: 编队快照对比 ====
    fig1 = plt.figure(figsize=(18, 5))

    # 初始状态
    ax1 = fig1.add_subplot(141, projection='3d')
    plot_formation_3d(ax1, pos_hist[0], leader_indices, adj,
                      title="(a) 初始状态 t=0s")

    # 编队建立后
    t_idx_settle = int(t_settle / dt)
    ax2 = fig1.add_subplot(142, projection='3d')
    plot_formation_3d(ax2, pos_hist[min(t_idx_settle, len(pos_hist) - 1)],
                      leader_indices, adj,
                      title=f"(b) 编队建立 t={t_settle}s")

    # 缩放后
    t_idx_scale = int((t_settle + t_trans + 1.0) / dt)
    ax3 = fig1.add_subplot(143, projection='3d')
    plot_formation_3d(ax3, pos_hist[min(t_idx_scale, len(pos_hist) - 1)],
                      leader_indices, adj,
                      title=f"(c) 缩放后 t={t_settle + t_trans + 1.0:.0f}s")

    # 旋转后
    ax4 = fig1.add_subplot(144, projection='3d')
    plot_formation_3d(ax4, pos_hist[-1], leader_indices, adj,
                      title=f"(d) 旋转后 t={T_total:.0f}s")

    fig1.suptitle('仿射编队控制 - 编队形态变换过程', fontsize=14, y=1.02)
    fig1.tight_layout()
    fig1.savefig('e:/crazyflie/src/fig1_formation_snapshots.png',
                 dpi=200, bbox_inches='tight')
    print("  已保存: fig1_formation_snapshots.png")

    # ==== 图2: 3D 轨迹 ====
    fig2 = plt.figure(figsize=(10, 8))
    ax_traj = fig2.add_subplot(111, projection='3d')
    plot_trajectories_3d(ax_traj, times, pos_hist, leader_indices)
    ax_traj.set_title('智能体 3D 轨迹', fontsize=13)
    ax_traj.legend(['Leader', 'Follower'], loc='upper left')
    fig2.tight_layout()
    fig2.savefig('e:/crazyflie/src/fig2_trajectories_3d.png',
                 dpi=200, bbox_inches='tight')
    print("  已保存: fig2_trajectories_3d.png")

    # ==== 图3: 误差收敛 ====
    fig3, axes3 = plt.subplots(2, 1, figsize=(12, 7))

    # 总误差
    plot_error_convergence(axes3[0], times, errors,
                           title="编队总误差 ||p_f - p_f*||")

    # 添加阶段标注
    phase_times = [0, t_settle, t_settle + t_trans,
                   t_settle + t_trans + t_hold,
                   t_settle + 2 * t_trans + t_hold, T_total]
    phase_labels = ['编队建立', '缩放变换', '缩放保持', '旋转变换', '旋转保持']
    colors = ['#FFCCCC', '#CCFFCC', '#CCCCFF', '#FFFFCC', '#FFCCFF']
    for i in range(len(phase_labels)):
        if i < len(phase_times) - 1:
            axes3[0].axvspan(phase_times[i], phase_times[i + 1],
                             alpha=0.15, color=colors[i % len(colors)])
            mid_t = (phase_times[i] + phase_times[i + 1]) / 2
            axes3[0].text(mid_t, axes3[0].get_ylim()[1] * 0.5,
                          phase_labels[i], ha='center', fontsize=8, alpha=0.7)

    # 各坐标分量轨迹
    for i in follower_indices:
        axes3[1].plot(times, pos_hist[:, i, 0], 'b-', alpha=0.3, linewidth=0.8)
        axes3[1].plot(times, pos_hist[:, i, 1], 'g-', alpha=0.3, linewidth=0.8)
        axes3[1].plot(times, pos_hist[:, i, 2], 'r-', alpha=0.3, linewidth=0.8)
    axes3[1].set_xlabel('时间 (s)')
    axes3[1].set_ylabel('位置 (m)')
    axes3[1].set_title('Follower 各坐标分量轨迹 (蓝:X, 绿:Y, 红:Z)', fontsize=11)
    axes3[1].grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig('e:/crazyflie/src/fig3_convergence.png',
                 dpi=200, bbox_inches='tight')
    print("  已保存: fig3_convergence.png")

    # ==== 图4: 应力矩阵热力图 ====
    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 4))

    # 完整 Ω
    im0 = axes4[0].imshow(Omega, cmap='RdBu_r', aspect='equal')
    axes4[0].set_title('应力矩阵 Ω', fontsize=11)
    axes4[0].set_xlabel('Agent j')
    axes4[0].set_ylabel('Agent i')
    plt.colorbar(im0, ax=axes4[0], shrink=0.8)

    # Ω_ff
    Omega_ff = Omega[np.ix_(follower_indices, follower_indices)]
    im1 = axes4[1].imshow(Omega_ff, cmap='RdBu_r', aspect='equal')
    axes4[1].set_title('Ω_ff (Follower子矩阵)', fontsize=11)
    axes4[1].set_xticks(range(len(follower_indices)))
    axes4[1].set_xticklabels(follower_indices)
    axes4[1].set_yticks(range(len(follower_indices)))
    axes4[1].set_yticklabels(follower_indices)
    plt.colorbar(im1, ax=axes4[1], shrink=0.8)

    # Ω_fl
    Omega_fl = Omega[np.ix_(follower_indices, leader_indices)]
    im2 = axes4[2].imshow(Omega_fl, cmap='RdBu_r', aspect='equal')
    axes4[2].set_title('Ω_fl (Follower-Leader子矩阵)', fontsize=11)
    axes4[2].set_xticks(range(len(leader_indices)))
    axes4[2].set_xticklabels(leader_indices)
    axes4[2].set_yticks(range(len(follower_indices)))
    axes4[2].set_yticklabels(follower_indices)
    plt.colorbar(im2, ax=axes4[2], shrink=0.8)

    fig4.suptitle('应力矩阵结构', fontsize=14)
    fig4.tight_layout()
    fig4.savefig('e:/crazyflie/src/fig4_stress_matrix.png',
                 dpi=200, bbox_inches='tight')
    print("  已保存: fig4_stress_matrix.png")

    # ==== 图5: 稀疏 vs 完全图拓扑对比 ====
    fig5 = plt.figure(figsize=(16, 7))

    ax_dense = fig5.add_subplot(121, projection='3d')
    plot_formation_3d(ax_dense, nominal_pos, leader_indices, adj_complete,
                      title=f'完全图 ({info["n_edges_complete"]} 条边, 度数=9)')

    ax_sparse = fig5.add_subplot(122, projection='3d')
    plot_formation_3d(ax_sparse, nominal_pos, leader_indices, adj,
                      title=f'稀疏图 ({info["n_edges_sparse"]} 条边, '
                            f'max度数={info["degree_max"]})')

    fig5.suptitle('通信拓扑优化: 完全图 vs Crazyflie 约束稀疏图', fontsize=14)
    fig5.tight_layout()
    fig5.savefig('e:/crazyflie/src/fig5_communication_graph.png',
                 dpi=200, bbox_inches='tight')
    print("  已保存: fig5_communication_graph.png")

    # ==== 图6: Crazyflie 通信指标对比 ====
    fig6, axes6 = plt.subplots(1, 3, figsize=(16, 5))

    # 柱状图：边数对比
    categories = ['完全图', '稀疏图']
    edge_counts = [info['n_edges_complete'], info['n_edges_sparse']]
    bars = axes6[0].bar(categories, edge_counts, color=['#FF6B6B', '#4ECDC4'],
                        edgecolor='black', linewidth=0.5)
    axes6[0].set_ylabel('边数')
    axes6[0].set_title('通信链路数量')
    for bar, v in zip(bars, edge_counts):
        axes6[0].text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                      str(v), ha='center', fontweight='bold')

    # 柱状图：度数分布
    degrees_complete = np.sum(adj_complete, axis=1)
    degrees_sparse = np.sum(adj, axis=1)
    x = np.arange(10)
    width = 0.35
    axes6[1].bar(x - width / 2, degrees_complete, width,
                 label='完全图', color='#FF6B6B', edgecolor='black', linewidth=0.5)
    axes6[1].bar(x + width / 2, degrees_sparse, width,
                 label='稀疏图', color='#4ECDC4', edgecolor='black', linewidth=0.5)
    axes6[1].axhline(y=CRAZYFLIE_COMM['max_neighbors'], color='red',
                     linestyle='--', alpha=0.7, linewidth=1.5)
    axes6[1].text(9.5, CRAZYFLIE_COMM['max_neighbors'] + 0.2,
                  f'上限={CRAZYFLIE_COMM["max_neighbors"]}',
                  color='red', fontsize=9, ha='right')
    axes6[1].set_xlabel('智能体编号')
    axes6[1].set_ylabel('度数 (邻居数)')
    axes6[1].set_title('节点度数分布')
    axes6[1].legend(loc='upper left')
    axes6[1].set_xticks(x)

    # 表格：性能对比
    axes6[2].axis('off')
    table_data = [
        ['边数', str(info['n_edges_complete']), str(info['n_edges_sparse'])],
        ['最大度数', '9', str(info['degree_max'])],
        ['平均度数', '9.0', f"{info['degree_mean']:.1f}"],
        ['λ_min(Ω_ff)', f"{info['t_max_dense']:.4f}",
         f"{info['min_eig_ff']:.4f}"],
        ['收敛速率', f"{gain * info['t_max_dense']:.4f}", f"{rate:.4f}"],
        ['通信负载', '100%',
         f"{info['n_edges_sparse'] / info['n_edges_complete'] * 100:.0f}%"],
    ]
    table = axes6[2].table(
        cellText=table_data,
        colLabels=['指标', '完全图', '稀疏图'],
        loc='center', cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    for j in range(3):
        table[0, j].set_text_props(fontweight='bold')
        table[0, j].set_facecolor('#E8E8E8')
    axes6[2].set_title('性能指标对比', fontsize=11, pad=20)

    fig6.suptitle('Crazyflie 通信约束优化效果', fontsize=14)
    fig6.tight_layout()
    fig6.savefig('e:/crazyflie/src/fig6_sparse_comparison.png',
                 dpi=200, bbox_inches='tight')
    print("  已保存: fig6_sparse_comparison.png")

    # ==== 图7: 控制输入与饱和约束分析 ====
    fig7, axes7 = plt.subplots(2, 2, figsize=(16, 10))

    # (a) 各跟随者控制输入范数（饱和）
    for i in range(controller.n_f):
        fi = follower_indices[i]
        axes7[0, 0].plot(times, ctrl_norms[:, i], linewidth=0.8, alpha=0.7,
                         label=f'Agent {fi}')
    axes7[0, 0].axhline(y=u_max, color='red', linestyle='--', linewidth=1.5,
                         label=f'u_max={u_max} m/s')
    axes7[0, 0].set_xlabel('时间 (s)')
    axes7[0, 0].set_ylabel('||u_i|| (m/s)')
    axes7[0, 0].set_title('(a) 控制输入范数 (smooth饱和)', fontsize=11)
    axes7[0, 0].legend(fontsize=8, loc='upper right')
    axes7[0, 0].grid(True, alpha=0.3)
    axes7[0, 0].set_xlim([0, T_total])

    # (b) 各跟随者控制输入范数（无饱和）
    for i in range(controller.n_f):
        fi = follower_indices[i]
        axes7[0, 1].plot(times_ns, ctrl_norms_ns[:, i], linewidth=0.8,
                         alpha=0.7, label=f'Agent {fi}')
    axes7[0, 1].axhline(y=u_max, color='red', linestyle='--', linewidth=1.5,
                         label=f'u_max={u_max} m/s')
    axes7[0, 1].set_xlabel('时间 (s)')
    axes7[0, 1].set_ylabel('||u_i|| (m/s)')
    axes7[0, 1].set_title('(b) 控制输入范数 (无饱和)', fontsize=11)
    axes7[0, 1].legend(fontsize=8, loc='upper right')
    axes7[0, 1].grid(True, alpha=0.3)
    axes7[0, 1].set_xlim([0, T_total])

    # (c) 编队误差对比
    axes7[1, 0].semilogy(times, errors + 1e-16, 'b-', linewidth=1.5,
                          label='smooth饱和')
    axes7[1, 0].semilogy(times_ns, errors_ns + 1e-16, 'g--', linewidth=1.5,
                          label='无饱和')
    axes7[1, 0].set_xlabel('时间 (s)')
    axes7[1, 0].set_ylabel('编队误差 ||p_f - p_f*||')
    axes7[1, 0].set_title('(c) 编队误差收敛对比', fontsize=11)
    axes7[1, 0].legend()
    axes7[1, 0].grid(True, alpha=0.3)
    axes7[1, 0].set_xlim([0, T_total])
    # 阶段背景色
    for i in range(len(phase_labels)):
        if i < len(phase_times) - 1:
            axes7[1, 0].axvspan(phase_times[i], phase_times[i + 1],
                                alpha=0.1, color=colors[i % len(colors)])

    # (d) 饱和率统计
    # 统计每个时刻有多少Agent处于饱和状态 (||u_raw|| > u_max * 0.95)
    u_raw_all = np.zeros_like(ctrl_inputs)
    for idx, t in enumerate(times):
        p_l = leader_traj(t)
        p_f_cur = pos_hist[idx, follower_indices]
        u_raw_all[idx] = -gain * (controller.Omega_ff @ p_f_cur
                                  + controller.Omega_fl @ p_l)
    raw_norms = np.linalg.norm(u_raw_all, axis=2)
    sat_ratio = np.mean(raw_norms > u_max * 0.95, axis=1) * 100  # %

    axes7[1, 1].fill_between(times, sat_ratio, alpha=0.3, color='coral')
    axes7[1, 1].plot(times, sat_ratio, 'r-', linewidth=1.0)
    axes7[1, 1].set_xlabel('时间 (s)')
    axes7[1, 1].set_ylabel('饱和比例 (%)')
    axes7[1, 1].set_title('(d) 跟随者饱和比例', fontsize=11)
    axes7[1, 1].set_ylim([0, 105])
    axes7[1, 1].grid(True, alpha=0.3)
    axes7[1, 1].set_xlim([0, T_total])
    for i in range(len(phase_labels)):
        if i < len(phase_times) - 1:
            axes7[1, 1].axvspan(phase_times[i], phase_times[i + 1],
                                alpha=0.1, color=colors[i % len(colors)])

    fig7.suptitle(f'输入饱和约束分析 (u_max = {u_max} m/s, smooth tanh)', fontsize=14)
    fig7.tight_layout()
    fig7.savefig('e:/crazyflie/src/fig7_saturation_analysis.png',
                 dpi=200, bbox_inches='tight')
    print("  已保存: fig7_saturation_analysis.png")

    # ==== 图8: 不同饱和上限对比 ====
    u_max_values = [0.5, 1.0, 2.0]
    fig8, axes8 = plt.subplots(1, 2, figsize=(14, 5))

    for um in u_max_values:
        ctrl_um = AFCController(Omega, leader_indices, gain=gain,
                                u_max=um, saturation_type='smooth')
        _, _, err_um, ci_um = simulate_first_order(
            ctrl_um, init_pos, leader_traj, (0, T_total), dt=dt
        )
        axes8[0].semilogy(times, err_um + 1e-16, linewidth=1.5,
                           label=f'u_max={um} m/s')
        ci_max_um = np.linalg.norm(ci_um, axis=2).max(axis=1)
        axes8[1].plot(times, ci_max_um, linewidth=1.2, alpha=0.8,
                       label=f'u_max={um} m/s')

    # 无饱和基线
    axes8[0].semilogy(times_ns, errors_ns + 1e-16, 'k--', linewidth=1.5,
                       alpha=0.6, label='无饱和')
    axes8[1].plot(times_ns, ctrl_norms_ns.max(axis=1), 'k--', linewidth=1.2,
                   alpha=0.6, label='无饱和')

    axes8[0].set_xlabel('时间 (s)')
    axes8[0].set_ylabel('编队误差')
    axes8[0].set_title('(a) 不同饱和上限下的误差收敛', fontsize=11)
    axes8[0].legend()
    axes8[0].grid(True, alpha=0.3)
    axes8[0].set_xlim([0, T_total])

    axes8[1].set_xlabel('时间 (s)')
    axes8[1].set_ylabel('max ||u_i|| (m/s)')
    axes8[1].set_title('(b) 最大控制输入', fontsize=11)
    axes8[1].legend()
    axes8[1].grid(True, alpha=0.3)
    axes8[1].set_xlim([0, T_total])

    fig8.suptitle('不同饱和上限对比分析', fontsize=14)
    fig8.tight_layout()
    fig8.savefig('e:/crazyflie/src/fig8_saturation_comparison.png',
                 dpi=200, bbox_inches='tight')
    print("  已保存: fig8_saturation_comparison.png")

    # ----------------------------------------------------------
    # Step 8: 验证仿射不变性
    # ----------------------------------------------------------
    print("\n[Step 8] 仿射不变性验证...")

    # 最终 leader 位置
    final_leaders = pos_hist[-1, leader_indices]
    # 理论 follower 稳态位置
    theoretical_followers = controller.steady_state(final_leaders)
    actual_followers = pos_hist[-1, follower_indices]

    print(f"  最终 Leader 位置:")
    for idx, li in enumerate(leader_indices):
        print(f"    Agent {li}: {final_leaders[idx]}")

    print(f"\n  Follower 位置对比 (实际 vs 理论):")
    for idx, fi in enumerate(follower_indices):
        actual = actual_followers[idx]
        theory = theoretical_followers[idx]
        err = np.linalg.norm(actual - theory)
        print(f"    Agent {fi}: 实际{actual.round(4)} "
              f"理论{theory.round(4)} 误差={err:.6f}")

    # 验证仿射关系
    A_total = A_rotate @ A_scale
    expected_followers = affine_transform(nominal_pos[follower_indices], A=A_total)
    affine_error = np.linalg.norm(theoretical_followers - expected_followers)
    print(f"\n  仿射不变性误差: {affine_error:.2e}")
    if affine_error < 1e-6:
        print("  ✓ 仿射不变性验证通过！")
    else:
        print("  ✗ 仿射不变性验证未通过")

    # ----------------------------------------------------------
    # 输出完整的应力矩阵数值
    # ----------------------------------------------------------
    print("\n[附录] 应力矩阵 Ω:")
    print(np.array2string(Omega, precision=4, suppress_small=True))

    print("\n[附录] Ω 的特征值:")
    eigvals = np.linalg.eigvalsh(Omega)
    print(f"  {np.array2string(eigvals, precision=6)}")

    plt.show()
    print("\n仿真完成！")


if __name__ == '__main__':
    main()
