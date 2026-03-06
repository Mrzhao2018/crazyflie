"""
animate_sim.py - 仿射编队控制多场景 3D 动画生成器

为所有仿真场景生成动画：
  Scenario 1 – 基线 AFC（编队建立 + 缩放 + 旋转）
  Scenario 2 – CBF 碰撞避免
  Scenario 3 – ESO 鲁棒抗扰（含风场对比）
  Scenario 4 – 事件触发通信（二阶积分器）
  Scenario 5 – 层级重组 RHF（U 形转弯）

用法：
    cd e:/crazyflie/src
    python animate_sim.py                  # 生成所有场景
    python animate_sim.py --scenario 1     # 仅生成指定场景
"""

import argparse
import shutil
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

from stress_matrix import compute_stress_matrix, compute_power_centric_stress_matrix
from formation import (
    double_pentagon, affine_transform, scale_matrix, rotation_matrix_z,
    create_leader_trajectory, smoothstep, CRAZYFLIE_COMM,
    check_affine_span, select_leaders_for_direction, compute_dwell_time,
)
from afc_controller import AFCController
from collision_avoidance import CBFSafetyFilter, CRAZYFLIE_SAFETY
from disturbance_observer import WindDisturbance, ExtendedStateObserver
from event_trigger import EventTriggerManager
from main_sim import (
    simulate_first_order, simulate_first_order_cbf,
    simulate_first_order_eso, simulate_second_order_et, simulate_rhf,
)
from archive import SimArchive

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _build_base_setup():
    """构建所有场景共用的基础参数（编队、控制器、Leader 轨迹）。

    Returns
    -------
    dict with keys:
        nominal_pos, leader_indices, follower_indices, adj,
        controller, leader_traj, init_pos, T_total,
        t_settle, t_trans, t_hold, Omega
    """
    nominal_pos, leader_indices, adj = double_pentagon(radius=1.0, height=1.0)
    follower_indices = sorted(set(range(10)) - set(leader_indices))

    Omega, _ = compute_stress_matrix(nominal_pos, adj, leader_indices, method='optimize')
    controller = AFCController(Omega, leader_indices, gain=5.0, u_max=1.0,
                               saturation_type='smooth')

    nominal_leaders = nominal_pos[leader_indices]
    A_scale = scale_matrix(3, 1.5)
    A_rotate = rotation_matrix_z(np.pi / 4)

    pos_phase1 = nominal_leaders.copy()
    pos_phase2 = affine_transform(nominal_leaders, A=A_scale)
    pos_phase3 = affine_transform(nominal_leaders, A=A_rotate @ A_scale)

    t_settle, t_trans, t_hold = 8.0, 5.0, 8.0
    phases = [
        {'start_positions': nominal_leaders, 't_start': 0.0, 't_end': 0.1, 'positions': pos_phase1},
        {'t_start': t_settle, 't_end': t_settle + t_trans, 'positions': pos_phase2},
        {'t_start': t_settle + t_trans + t_hold,
         't_end': t_settle + 2 * t_trans + t_hold, 'positions': pos_phase3},
    ]
    leader_traj = create_leader_trajectory(phases)
    T_total = t_settle + 2 * t_trans + 2 * t_hold

    np.random.seed(42)
    init_pos = nominal_pos.copy()
    init_pos[follower_indices] += np.random.randn(len(follower_indices), 3) * 0.5

    return dict(
        nominal_pos=nominal_pos, leader_indices=leader_indices,
        follower_indices=follower_indices, adj=adj,
        controller=controller, leader_traj=leader_traj,
        init_pos=init_pos, T_total=T_total,
        t_settle=t_settle, t_trans=t_trans, t_hold=t_hold,
        Omega=Omega,
    )


def run_baseline_scenario(setup):
    """场景1：基线 AFC（缩放 + 旋转变换）。"""
    c = setup['controller']
    times, pos_hist, errors, _ = simulate_first_order(
        c, setup['init_pos'], setup['leader_traj'],
        (0, setup['T_total']), dt=0.02,
    )
    t_settle, t_trans, t_hold = setup['t_settle'], setup['t_trans'], setup['t_hold']
    T = setup['T_total']
    phase_specs = [
        (0,                            t_settle,                   '#FFCCCC', '编队建立'),
        (t_settle,                     t_settle + t_trans,         '#CCFFCC', '缩放变换'),
        (t_settle + t_trans,           t_settle + t_trans + t_hold,'#CCCCFF', '缩放保持'),
        (t_settle + t_trans + t_hold,  t_settle + 2*t_trans+t_hold,'#FFFFCC', '旋转变换'),
        (t_settle + 2*t_trans + t_hold, T,                         '#FFCCFF', '旋转保持'),
    ]
    return {
        'title': '场景1：基线 AFC（缩放 + 旋转）',
        'output': 'afc_scene1_baseline',
        'times': times, 'pos_hist': pos_hist, 'errors': errors,
        'adj': setup['adj'],
        'leader_indices': setup['leader_indices'],
        'follower_indices': setup['follower_indices'],
        'phase_specs': phase_specs,
    }


def run_cbf_scenario(setup):
    """场景2：CBF 碰撞避免（大初始扰动，高碰撞风险）。"""
    d_safe   = CRAZYFLIE_SAFETY['safety_distance_m']
    gamma    = CRAZYFLIE_SAFETY['cbf_gamma']
    d_act    = CRAZYFLIE_SAFETY['activate_distance_m']
    cbf = CBFSafetyFilter(n_agents=10, leader_indices=setup['leader_indices'],
                          d_safe=d_safe, gamma=gamma, d_activate=d_act)

    np.random.seed(99)
    init_pos_cbf = setup['nominal_pos'].copy()
    init_pos_cbf[setup['follower_indices']] += (
        np.random.randn(len(setup['follower_indices']), 3) * 1.5
    )

    times, pos_hist, errors, _, cbf_data = simulate_first_order_cbf(
        setup['controller'], init_pos_cbf, setup['leader_traj'],
        (0, setup['T_total']), dt=0.02, cbf_filter=cbf,
    )
    T = setup['T_total']
    t_settle, t_trans, t_hold = setup['t_settle'], setup['t_trans'], setup['t_hold']
    phase_specs = [
        (0,                             t_settle,                    '#FFCCCC', '编队建立'),
        (t_settle,                      t_settle + t_trans,          '#CCFFCC', '缩放变换'),
        (t_settle + t_trans,            t_settle + t_trans + t_hold, '#CCCCFF', '缩放保持'),
        (t_settle + t_trans + t_hold,   t_settle+2*t_trans+t_hold,   '#FFFFCC', '旋转变换'),
        (t_settle + 2*t_trans + t_hold, T,                           '#FFCCFF', '旋转保持'),
    ]
    return {
        'title': '场景2：CBF 碰撞避免',
        'output': 'afc_scene2_cbf',
        'times': times, 'pos_hist': pos_hist, 'errors': errors,
        'adj': setup['adj'],
        'leader_indices': setup['leader_indices'],
        'follower_indices': setup['follower_indices'],
        'phase_specs': phase_specs,
        'extra': {'min_distances': cbf_data['min_distances'],
                  'd_safe': d_safe},
    }


def run_eso_scenario(setup):
    """场景3：ESO 鲁棒抗扰（有风 + ESO 补偿 vs 无 ESO）。"""
    n_f = setup['controller'].n_f
    wind = WindDisturbance(n_agents=n_f, dim=3,
                           w_const=np.array([0.2, 0.1, 0.05]),
                           ou_theta=0.5, ou_sigma=0.1, seed=123)
    eso  = ExtendedStateObserver(n_f, dim=3, omega_o=8.0)

    times, pos_hist, errors, _, eso_data = simulate_first_order_eso(
        setup['controller'], setup['init_pos'], setup['leader_traj'],
        (0, setup['T_total']), dt=0.02, wind=wind, eso=eso,
    )
    T = setup['T_total']
    t_settle, t_trans, t_hold = setup['t_settle'], setup['t_trans'], setup['t_hold']
    phase_specs = [
        (0,                             t_settle,                    '#FFCCCC', '编队建立'),
        (t_settle,                      t_settle + t_trans,          '#CCFFCC', '缩放变换'),
        (t_settle + t_trans,            t_settle + t_trans + t_hold, '#CCCCFF', '缩放保持'),
        (t_settle + t_trans + t_hold,   t_settle+2*t_trans+t_hold,   '#FFFFCC', '旋转变换'),
        (t_settle + 2*t_trans + t_hold, T,                           '#FFCCFF', '旋转保持'),
    ]
    return {
        'title': '场景3：ESO 鲁棒抗扰（ω₀=8 rad/s）',
        'output': 'afc_scene3_eso',
        'times': times, 'pos_hist': pos_hist, 'errors': errors,
        'adj': setup['adj'],
        'leader_indices': setup['leader_indices'],
        'follower_indices': setup['follower_indices'],
        'phase_specs': phase_specs,
        'extra': {'est_errors': eso_data['estimation_errors']},
    }


def run_et_scenario(setup):
    """场景4：自适应事件触发通信（二阶积分器）。"""
    et_mgr = EventTriggerManager(
        n_agents=10, d=3,
        follower_indices=setup['follower_indices'],
        leader_indices=setup['leader_indices'],
        Omega=setup['Omega'],
        mu=0.01, varpi=0.5, phi_0=1.0,
    )
    init_vel = np.zeros_like(setup['init_pos'])
    times, pos_hist, errors, _, et_data = simulate_second_order_et(
        setup['controller'], setup['init_pos'], init_vel,
        setup['leader_traj'], (0, setup['T_total']), dt=0.02,
        et_manager=et_mgr,
    )
    T = setup['T_total']
    t_settle, t_trans, t_hold = setup['t_settle'], setup['t_trans'], setup['t_hold']
    phase_specs = [
        (0,                             t_settle,                    '#FFCCCC', '编队建立'),
        (t_settle,                      t_settle + t_trans,          '#CCFFCC', '缩放变换'),
        (t_settle + t_trans,            t_settle + t_trans + t_hold, '#CCCCFF', '缩放保持'),
        (t_settle + t_trans + t_hold,   t_settle+2*t_trans+t_hold,   '#FFFFCC', '旋转变换'),
        (t_settle + 2*t_trans + t_hold, T,                           '#FFCCFF', '旋转保持'),
    ]
    comm = et_data['comm_rates']
    return {
        'title': f"场景4：事件触发通信（平均通信率 {comm['mean']:.1f}%）",
        'output': 'afc_scene4_et',
        'times': times, 'pos_hist': pos_hist, 'errors': errors,
        'adj': setup['adj'],
        'leader_indices': setup['leader_indices'],
        'follower_indices': setup['follower_indices'],
        'phase_specs': phase_specs,
        'extra': {'trigger_log': et_data['trigger_log'],
                  'comm_rates': comm},
    }


def run_rhf_scenario():
    """场景5：层级重组 RHF（U 形转弯，动态 Leader 切换）。"""
    nominal_pos, _, _ = double_pentagon(radius=1.0, height=1.0)

    # --- Phase 0 ---
    leaders_p0 = [0, 1, 2, 5]
    Omega_p0, info_p0 = compute_power_centric_stress_matrix(nominal_pos, leaders_p0)

    # --- Phase 1: +Y 方向 ---
    leaders_p1, _ = select_leaders_for_direction(nominal_pos, [0, 1, 0], n_leaders=4, d=3)
    Omega_p1, info_p1 = compute_power_centric_stress_matrix(nominal_pos, leaders_p1)

    # --- Phase 2: -X 方向 ---
    leaders_p2, _ = select_leaders_for_direction(nominal_pos, [-1, 0, 0], n_leaders=4, d=3)
    Omega_p2, info_p2 = compute_power_centric_stress_matrix(nominal_pos, leaders_p2)

    targets_p0 = nominal_pos[leaders_p0] + np.array([0.5, 0.0, 0.0])
    targets_p1 = nominal_pos[leaders_p1] + np.array([0.5, 1.0, 0.0])
    targets_p2 = nominal_pos[leaders_p2] + np.array([-0.5, 1.0, 0.0])

    rhf_schedule = [
        {'t_switch': 0.0,  'leader_indices': leaders_p0, 'leader_targets': targets_p0,
         't_transition': 3.0, 'omega': Omega_p0, 'adj': info_p0['adj_matrix'],
         'label': 'Phase 0: +X 平移'},
        {'t_switch': 35.0, 'leader_indices': leaders_p1, 'leader_targets': targets_p1,
         't_transition': 3.0, 'omega': Omega_p1, 'adj': info_p1['adj_matrix'],
         'label': 'Phase 1: +Y 转弯'},
        {'t_switch': 70.0, 'leader_indices': leaders_p2, 'leader_targets': targets_p2,
         't_transition': 3.0, 'omega': Omega_p2, 'adj': info_p2['adj_matrix'],
         'label': 'Phase 2: -X 转弯(U 形)'},
    ]
    T_rhf = 105.0

    rhf_ctrl = AFCController(Omega_p0, leaders_p0, gain=10.0, damping=1.0,
                             u_max=CRAZYFLIE_COMM['max_velocity'],
                             saturation_type='smooth')
    np.random.seed(123)
    init_pos = nominal_pos.copy()
    init_pos += np.random.randn(10, 3) * 0.1
    init_vel = np.zeros((10, 3))

    times, pos_hist, errors, _, rhf_data = simulate_rhf(
        rhf_ctrl, init_pos, init_vel, nominal_pos, rhf_schedule,
        (0, T_rhf), dt=0.02,
    )

    switch_times = [s['t_switch'] for s in rhf_schedule]
    phase_specs = [
        (0,    35.0,  '#FFCCCC', 'Phase 0: +X'),
        (35.0, 70.0,  '#CCFFCC', 'Phase 1: +Y'),
        (70.0, T_rhf, '#CCCCFF', 'Phase 2: -X'),
    ]
    return {
        'title': '场景5：层级重组 RHF（U 形转弯）',
        'output': 'afc_scene5_rhf',
        'times': times, 'pos_hist': pos_hist, 'errors': errors,
        'adj': info_p0['adj_matrix'],
        'leader_indices': leaders_p0,
        'follower_indices': sorted(set(range(10)) - set(leaders_p0)),
        'phase_specs': phase_specs,
        'extra': {'switch_times': switch_times,
                  'rhf_data': rhf_data,
                  'schedule': rhf_schedule},
    }


def _save_animation(scenario, output_dir='e:/crazyflie/src'):
    """通用 3D 动画生成器，接受统一 scenario dict 输出 GIF/MP4。"""
    times     = scenario['times']
    pos_hist  = scenario['pos_hist']
    errors    = scenario['errors']
    adj       = scenario['adj']
    leaders   = scenario['leader_indices']
    followers = scenario['follower_indices']
    phase_specs = scenario['phase_specs']
    title     = scenario['title']
    out_name  = scenario['output']
    extra     = scenario.get('extra', {})

    n_steps, n, d = pos_hist.shape
    skip = max(1, n_steps // 600)
    frame_indices = list(range(0, n_steps, skip))
    n_frames = len(frame_indices)
    print(f"  {title}: {n_steps} 步 → {n_frames} 帧动画")

    all_pos = pos_hist[frame_indices]
    margin = 0.5
    x_min, x_max = all_pos[:,:,0].min()-margin, all_pos[:,:,0].max()+margin
    y_min, y_max = all_pos[:,:,1].min()-margin, all_pos[:,:,1].max()+margin
    z_min, z_max = all_pos[:,:,2].min()-margin, all_pos[:,:,2].max()+margin

    fig = plt.figure(figsize=(14, 8))
    gs  = fig.add_gridspec(2, 2, width_ratios=[2, 1], height_ratios=[3, 1],
                           hspace=0.3, wspace=0.3)
    ax3d  = fig.add_subplot(gs[0, 0], projection='3d')
    ax_err = fig.add_subplot(gs[1, :])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis('off')

    trail_len = min(50, n_frames)
    edge_lines: list = []

    leader_scatter   = ax3d.scatter([], [], [], c='red', s=120, marker='^',  # type: ignore[arg-type]
                                    edgecolors='darkred', linewidths=0.8, zorder=5)
    follower_scatter = ax3d.scatter([], [], [], c='dodgerblue', s=70, marker='o',  # type: ignore[arg-type]
                                    edgecolors='navy', linewidths=0.5, zorder=5)
    trail_lines = []
    for i in range(n):
        color = 'red' if i in leaders else 'dodgerblue'
        line, = ax3d.plot([], [], [], color=color, alpha=0.35, linewidth=0.8)
        trail_lines.append(line)

    err_line, = ax_err.semilogy([], [], 'b-', linewidth=1.5)
    err_dot,  = ax_err.semilogy([], [], 'ro', markersize=6, zorder=5)
    ax_err.set_xlim(0, times[-1])
    ax_err.set_ylim(max(errors.min() * 0.5, 1e-4), errors.max() * 2)
    ax_err.set_xlabel('时间 (s)', fontsize=10)
    ax_err.set_ylabel('编队误差', fontsize=10)
    ax_err.set_title('编队误差', fontsize=11)
    ax_err.grid(True, alpha=0.3)

    for t0, t1, color, label in phase_specs:
        ax_err.axvspan(t0, t1, alpha=0.15, color=color)
        ax_err.text((t0+t1)/2, errors.max()*1.3, label,
                    ha='center', fontsize=7, alpha=0.7)

    # 场景专属指示线
    if 'switch_times' in extra:
        for st in extra['switch_times']:
            if st > 0:
                ax_err.axvline(st, color='orange', linestyle='--', linewidth=1.0, alpha=0.7)
    if 'd_safe' in extra:
        ax_err.axhline(extra['d_safe'], color='red', linestyle=':', linewidth=1.0, alpha=0.6,
                       label=f"d_safe={extra['d_safe']}m")

    info_text = ax_info.text(
        0.05, 0.95, '', transform=ax_info.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
    )

    def _phase_name(t):
        for t0, t1, _, name in phase_specs:
            if t0 <= t < t1:
                return name
        return phase_specs[-1][3]

    def init():
        ax3d.set_xlim(x_min, x_max)
        ax3d.set_ylim(y_min, y_max)
        ax3d.set_zlim(z_min, z_max)
        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Z (m)')
        return []

    def update(frame_idx):
        nonlocal edge_lines
        fi  = frame_indices[frame_idx]
        t   = times[fi]
        pos = pos_hist[fi]

        leader_scatter._offsets3d   = (pos[leaders, 0],   pos[leaders, 1],   pos[leaders, 2])   # type: ignore[attr-defined]
        follower_scatter._offsets3d = (pos[followers, 0], pos[followers, 1], pos[followers, 2]) # type: ignore[attr-defined]

        for ln in edge_lines:
            ln.remove()
        edge_lines = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i, j] > 0:
                    ln, = ax3d.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]],
                                    [pos[i,2], pos[j,2]], 'gray', alpha=0.1, linewidth=0.3)
                    edge_lines.append(ln)

        s0 = max(0, frame_idx - trail_len)
        for i in range(n):
            idx_list = [frame_indices[k] for k in range(s0, frame_idx+1)]
            trail_lines[i].set_data(pos_hist[idx_list, i, 0], pos_hist[idx_list, i, 1])
            trail_lines[i].set_3d_properties(pos_hist[idx_list, i, 2])

        ax3d.set_title(f'{title}  t = {t:.2f}s', fontsize=11)

        err_line.set_data(times[:fi+1], errors[:fi+1])
        err_dot.set_data([t], [errors[fi]])

        # 场景专属附加行
        extra_lines = ''
        if 'min_distances' in extra:
            extra_lines = f"\nMin dist: {extra['min_distances'][fi]:.3f}m"
        elif 'est_errors' in extra:
            extra_lines = f"\nESO err: {extra['est_errors'][fi]:.4f}"
        elif 'comm_rates' in extra:
            extra_lines = f"\nComm rate: {extra['comm_rates']['mean']:.1f}%"
        elif 'switch_times' in extra:
            phase_idx = sum(1 for st in extra['switch_times'] if st <= t) - 1
            extra_lines = f"\nPhase: {phase_idx}"

        info_str = (
            f"Time:  {t:.2f} s\n"
            f"Stage: {_phase_name(t)}\n"
            f"─────────────────\n"
            f"Error: {errors[fi]:.4f}\n"
            f"Leader:   {leaders}\n"
            f"Frame: {frame_idx+1}/{n_frames}"
            + extra_lines
        )
        info_text.set_text(info_str)
        return []

    anim = FuncAnimation(fig, update, init_func=init,
                         frames=n_frames, interval=50, blit=False)

    gif_path = f'{output_dir}/{out_name}.gif'
    gif_ok = False
    try:
        anim.save(gif_path, writer=PillowWriter(fps=20), dpi=90)
        print(f"  ✓ GIF 已保存: {gif_path}")
        gif_ok = True
    except Exception as e:
        print(f"  ✗ GIF 保存失败: {e}")

    mp4_path = f'{output_dir}/{out_name}.mp4'
    mp4_ok = False
    try:
        anim.save(mp4_path, writer=FFMpegWriter(fps=20, bitrate=1800), dpi=110)
        print(f"  ✓ MP4 已保存: {mp4_path}")
        mp4_ok = True
    except Exception as e:
        print(f"  ✗ MP4 需要 ffmpeg，跳过: {e}")

    plt.close(fig)

    # ---------- 存档 ----------
    arch = SimArchive(tag=out_name)
    arch.save_arrays(
        times=times, positions=pos_hist, errors=errors,
    )
    arch.save_params({
        'title': title,
        'output': out_name,
        'n_agents': n,
        'leader_indices': leaders,
        'follower_indices': followers,
        'T_total': float(times[-1]),
        'n_frames': n_frames,
        'phase_specs': [(t0, t1, color, label) for t0, t1, color, label in phase_specs],
    })
    if gif_ok:
        shutil.copy(gif_path, arch._tmp_dir)
    if mp4_ok:
        shutil.copy(mp4_path, arch._tmp_dir)
    zip_path = arch.finalize()
    print(f"  ✓ 存档: {zip_path}")


def main():
    parser = argparse.ArgumentParser(description='AFC 多场景 3D 动画生成器')
    parser.add_argument('--scenario', type=int, default=0,
                        help='场景编号 1-5，默认 0 表示全部生成')
    args = parser.parse_args()

    # 所有共享基础参数（场景1-4复用）
    print("初始化基础仿真参数...")
    setup = _build_base_setup()

    # 场景运行器列表（编号从 1 开始）
    runners = {
        1: lambda: run_baseline_scenario(setup),
        2: lambda: run_cbf_scenario(setup),
        3: lambda: run_eso_scenario(setup),
        4: lambda: run_et_scenario(setup),
        5: run_rhf_scenario,
    }

    targets = list(runners.keys()) if args.scenario == 0 else [args.scenario]

    for idx in targets:
        if idx not in runners:
            print(f"[警告] 未知场景编号 {idx}，跳过")
            continue
        print(f"\n{'='*50}")
        print(f"[场景 {idx}/{len(runners)}] 运行仿真...")
        sc = runners[idx]()
        _save_animation(sc)

    print("\n全部完成！")


if __name__ == '__main__':
    main()
