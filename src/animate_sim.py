"""
animate_sim.py - 仿射编队控制多场景动画系统

为 AFC 各验证场景生成具有场景特征的动画：
  1. Baseline  : 强调仿射变换、控制饱和、阶段切换
  2. CBF       : 强调安全距离、最近对、约束激活与控制修正
  3. ESO       : 强调风扰动、估计精度、补偿效果
  4. ET        : 强调事件触发时间线、通信节省、触发活跃度
  5. RHF       : 强调 Leader 重组、拓扑切换、误差瞬态

用法：
    在仓库根目录运行：
    python animate_sim.py
    python animate_sim.py --scenario 3

    或者在 src 目录运行：
    python animate_sim.py
"""

import argparse
import os
import shutil

import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from stress_matrix import compute_power_centric_stress_matrix, compute_stress_matrix
from formation import (
    CRAZYFLIE_COMM,
    affine_transform,
    create_leader_trajectory,
    double_pentagon,
    rotation_matrix_z,
    scale_matrix,
    select_leaders_for_direction,
)
from afc_controller import AFCController
from archive import SimArchive
from collision_avoidance import CBFSafetyFilter, CRAZYFLIE_SAFETY
from disturbance_observer import ExtendedStateObserver, WindDisturbance
from event_trigger import EventTriggerManager
from main_sim import (
    simulate_first_order,
    simulate_first_order_cbf,
    simulate_first_order_eso,
    simulate_rhf,
    simulate_second_order_et,
)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
VIDEO_DIR = os.path.join(OUTPUT_DIR, 'videos')
for _dir in (OUTPUT_DIR, FIGURE_DIR, VIDEO_DIR):
    os.makedirs(_dir, exist_ok=True)


def figure_path(name):
    return os.path.join(FIGURE_DIR, name)


def video_path(name):
    return os.path.join(VIDEO_DIR, name)


SCENE_THEMES = {
    'baseline': {
        'figure_face': '#f6fbff',
        'panel_face': '#edf7fb',
        'leader': '#d1495b',
        'leader_edge': '#8f2d3a',
        'follower': '#2b6f97',
        'accent': '#ff9f1c',
        'aux': '#3a86ff',
    },
    'cbf': {
        'figure_face': '#fff8f5',
        'panel_face': '#fff0eb',
        'leader': '#c44536',
        'leader_edge': '#7f2a1f',
        'follower': '#3a86ff',
        'accent': '#d00000',
        'aux': '#f77f00',
    },
    'eso': {
        'figure_face': '#f4fff9',
        'panel_face': '#e8fbf0',
        'leader': '#ef476f',
        'leader_edge': '#99314a',
        'follower': '#118ab2',
        'accent': '#06d6a0',
        'aux': '#1b9aaa',
    },
    'et': {
        'figure_face': '#fffdf6',
        'panel_face': '#fff5df',
        'leader': '#d1495b',
        'leader_edge': '#8f2d3a',
        'follower': '#577590',
        'accent': '#f8961e',
        'aux': '#bc6c25',
    },
    'rhf': {
        'figure_face': '#f7fbfa',
        'panel_face': '#ebf6f3',
        'leader': '#ee6c4d',
        'leader_edge': '#9c3f2c',
        'follower': '#52796f',
        'accent': '#2a9d8f',
        'aux': '#264653',
        'phase_colors': ['#1f77b4', '#ff7f0e', '#2ca02c'],
    },
}


def _base_phase_specs(setup):
    t_settle = setup['t_settle']
    t_trans = setup['t_trans']
    t_hold = setup['t_hold']
    total = setup['T_total']
    return [
        (0.0, t_settle, '#FFCCCC', '编队建立'),
        (t_settle, t_settle + t_trans, '#CCFFCC', '缩放变换'),
        (t_settle + t_trans, t_settle + t_trans + t_hold, '#CCCCFF', '缩放保持'),
        (t_settle + t_trans + t_hold, t_settle + 2 * t_trans + t_hold, '#FFFFCC', '旋转变换'),
        (t_settle + 2 * t_trans + t_hold, total, '#FFCCFF', '旋转保持'),
    ]


def _phase_name(t, phase_specs):
    for t0, t1, _, label in phase_specs:
        if t0 <= t < t1:
            return label
    return phase_specs[-1][3]


def _compute_bounds(pos_hist, frame_indices, margin=0.5):
    all_pos = pos_hist[frame_indices]
    x_min = all_pos[:, :, 0].min() - margin
    x_max = all_pos[:, :, 0].max() + margin
    y_min = all_pos[:, :, 1].min() - margin
    y_max = all_pos[:, :, 1].max() + margin
    z_min = all_pos[:, :, 2].min() - margin
    z_max = all_pos[:, :, 2].max() + margin
    return x_min, x_max, y_min, y_max, z_min, z_max


def _compute_min_pair_history(pos_hist):
    n_steps, n_agents, _ = pos_hist.shape
    pair_idx = np.zeros((n_steps, 2), dtype=int)
    pair_dist = np.zeros(n_steps)
    for k in range(n_steps):
        pos = pos_hist[k]
        best_dist = float('inf')
        best_pair = (0, 1)
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                dist = float(np.linalg.norm(pos[i] - pos[j]))
                if dist < best_dist:
                    best_dist = dist
                    best_pair = (i, j)
        pair_dist[k] = best_dist
        pair_idx[k] = best_pair
    return pair_idx, pair_dist


def _compute_cumulative_trigger_counts(times, trigger_log, follower_indices):
    counts = np.zeros((len(times), len(follower_indices)), dtype=int)
    mapping = {fi: idx for idx, fi in enumerate(follower_indices)}
    sorted_log = sorted(trigger_log, key=lambda item: item[0])
    cursor = 0
    running = np.zeros(len(follower_indices), dtype=int)
    for ti, t in enumerate(times):
        while cursor < len(sorted_log) and sorted_log[cursor][0] <= t + 1e-12:
            _, agent = sorted_log[cursor]
            if agent in mapping:
                running[mapping[agent]] += 1
            cursor += 1
        counts[ti] = running
    return counts


def _compute_total_trigger_curve(cumulative_counts):
    if cumulative_counts.size == 0:
        return np.zeros(1)
    return cumulative_counts.sum(axis=1)


def _current_rhf_phase(schedule, t):
    phase_idx = 0
    for idx, item in enumerate(schedule):
        if t >= item['t_switch']:
            phase_idx = idx
        else:
            break
    return phase_idx


def _build_base_setup():
    nominal_pos, leader_indices, adj = double_pentagon(radius=1.0, height=1.0)
    follower_indices = sorted(set(range(10)) - set(leader_indices))

    omega, _ = compute_stress_matrix(nominal_pos, adj, leader_indices, method='optimize')
    controller = AFCController(
        omega, leader_indices, gain=5.0, u_max=1.0, saturation_type='smooth'
    )

    nominal_leaders = nominal_pos[leader_indices]
    pos_scale = affine_transform(nominal_leaders, A=scale_matrix(3, 1.5))
    pos_rotate = affine_transform(
        nominal_leaders,
        A=rotation_matrix_z(np.pi / 4) @ scale_matrix(3, 1.5),
    )

    t_settle, t_trans, t_hold = 8.0, 5.0, 8.0
    phases = [
        {'start_positions': nominal_leaders, 't_start': 0.0, 't_end': 0.1, 'positions': nominal_leaders.copy()},
        {'t_start': t_settle, 't_end': t_settle + t_trans, 'positions': pos_scale},
        {
            't_start': t_settle + t_trans + t_hold,
            't_end': t_settle + 2 * t_trans + t_hold,
            'positions': pos_rotate,
        },
    ]
    leader_traj = create_leader_trajectory(phases)
    total = t_settle + 2 * t_trans + 2 * t_hold

    np.random.seed(42)
    init_pos = nominal_pos.copy()
    init_pos[follower_indices] += np.random.randn(len(follower_indices), 3) * 0.5

    return {
        'nominal_pos': nominal_pos,
        'leader_indices': leader_indices,
        'follower_indices': follower_indices,
        'adj': adj,
        'controller': controller,
        'leader_traj': leader_traj,
        'init_pos': init_pos,
        'T_total': total,
        't_settle': t_settle,
        't_trans': t_trans,
        't_hold': t_hold,
        'Omega': omega,
    }


def run_baseline_scenario(setup):
    controller = setup['controller']
    times, pos_hist, errors, ctrl_inputs = simulate_first_order(
        controller,
        setup['init_pos'],
        setup['leader_traj'],
        (0, setup['T_total']),
        dt=0.02,
    )
    ctrl_norms = np.linalg.norm(ctrl_inputs, axis=2)
    max_ctrl = ctrl_norms.max(axis=1)

    u_raw = np.zeros_like(ctrl_inputs)
    for idx, t in enumerate(times):
        p_l = setup['leader_traj'](t)
        p_f = pos_hist[idx, setup['follower_indices']]
        u_raw[idx] = -controller.gain * (controller.Omega_ff @ p_f + controller.Omega_fl @ p_l)
    raw_norms = np.linalg.norm(u_raw, axis=2)
    sat_ratio = np.mean(raw_norms > (controller.u_max or 0.0) * 0.95, axis=1) * 100.0

    return {
        'kind': 'baseline',
        'title': '场景1：基线 AFC（缩放 + 旋转）',
        'output': 'afc_scene1_baseline',
        'times': times,
        'pos_hist': pos_hist,
        'errors': errors,
        'adj': setup['adj'],
        'leader_indices': setup['leader_indices'],
        'follower_indices': setup['follower_indices'],
        'phase_specs': _base_phase_specs(setup),
        'extra': {
            'ctrl_norms': ctrl_norms,
            'max_ctrl': max_ctrl,
            'sat_ratio': sat_ratio,
            'u_max': controller.u_max,
        },
    }


def run_cbf_scenario(setup):
    d_safe = CRAZYFLIE_SAFETY['safety_distance_m']
    cbf = CBFSafetyFilter(
        n_agents=10,
        leader_indices=setup['leader_indices'],
        d_safe=d_safe,
        gamma=CRAZYFLIE_SAFETY['cbf_gamma'],
        d_activate=CRAZYFLIE_SAFETY['activate_distance_m'],
    )

    np.random.seed(99)
    init_pos = setup['nominal_pos'].copy()
    init_pos[setup['follower_indices']] += np.random.randn(len(setup['follower_indices']), 3) * 1.5

    times_no, _, _, _, cbf_no = simulate_first_order_cbf(
        setup['controller'], init_pos, setup['leader_traj'], (0, setup['T_total']), dt=0.02, cbf_filter=None
    )
    times_yes, pos_hist, errors, _, cbf_yes = simulate_first_order_cbf(
        setup['controller'], init_pos, setup['leader_traj'], (0, setup['T_total']), dt=0.02, cbf_filter=cbf
    )
    pair_hist, pair_dist = _compute_min_pair_history(pos_hist)

    return {
        'kind': 'cbf',
        'title': '场景2：CBF 碰撞避免',
        'output': 'afc_scene2_cbf',
        'times': times_yes,
        'pos_hist': pos_hist,
        'errors': errors,
        'adj': setup['adj'],
        'leader_indices': setup['leader_indices'],
        'follower_indices': setup['follower_indices'],
        'phase_specs': _base_phase_specs(setup),
        'extra': {
            'd_safe': d_safe,
            'min_dist_no': cbf_no['min_distances'],
            'min_dist_yes': cbf_yes['min_distances'],
            'n_active': cbf_yes['n_active'],
            'modifications': cbf_yes['modifications'],
            'times_no': times_no,
            'closest_pair': pair_hist,
            'closest_pair_dist': pair_dist,
        },
    }


def run_eso_scenario(setup):
    n_f = setup['controller'].n_f
    w_const = np.array([0.2, 0.1, 0.05])
    ou_theta = 0.5
    ou_sigma = 0.1
    wind_seed = 123
    omega_o = 8.0

    times_bl, _, err_bl, _, _ = simulate_first_order_eso(
        setup['controller'],
        setup['init_pos'],
        setup['leader_traj'],
        (0, setup['T_total']),
        dt=0.02,
        wind=None,
        eso=None,
    )
    times_nd, _, err_nd, _, _ = simulate_first_order_eso(
        setup['controller'],
        setup['init_pos'],
        setup['leader_traj'],
        (0, setup['T_total']),
        dt=0.02,
        wind=WindDisturbance(n_f, dim=3, w_const=w_const, ou_theta=ou_theta, ou_sigma=ou_sigma, seed=wind_seed),
        eso=None,
    )
    times_wd, pos_hist, errors, _, eso_data = simulate_first_order_eso(
        setup['controller'],
        setup['init_pos'],
        setup['leader_traj'],
        (0, setup['T_total']),
        dt=0.02,
        wind=WindDisturbance(n_f, dim=3, w_const=w_const, ou_theta=ou_theta, ou_sigma=ou_sigma, seed=wind_seed),
        eso=ExtendedStateObserver(n_f, dim=3, omega_o=omega_o),
    )

    rep_idx = 0
    disturbance_true_norm = np.linalg.norm(eso_data['disturbances_true'][:, rep_idx], axis=1)
    disturbance_est_norm = np.linalg.norm(eso_data['disturbances_est'][:, rep_idx], axis=1)
    per_agent_est_error = np.linalg.norm(
        eso_data['disturbances_true'] - eso_data['disturbances_est'], axis=2
    )

    return {
        'kind': 'eso',
        'title': '场景3：ESO 鲁棒抗扰',
        'output': 'afc_scene3_eso',
        'times': times_wd,
        'pos_hist': pos_hist,
        'errors': errors,
        'adj': setup['adj'],
        'leader_indices': setup['leader_indices'],
        'follower_indices': setup['follower_indices'],
        'phase_specs': _base_phase_specs(setup),
        'extra': {
            'err_bl': err_bl,
            'err_nd': err_nd,
            'times_bl': times_bl,
            'times_nd': times_nd,
            'disturbance_true_norm': disturbance_true_norm,
            'disturbance_est_norm': disturbance_est_norm,
            'estimation_errors': eso_data['estimation_errors'],
            'per_agent_est_error': per_agent_est_error,
            'omega_o': omega_o,
            'w_const': w_const,
            'rep_agent': setup['follower_indices'][rep_idx],
        },
    }


def run_et_scenario(setup):
    init_vel = np.zeros_like(setup['init_pos'])
    et_mgr = EventTriggerManager(
        n_agents=10,
        d=3,
        follower_indices=setup['follower_indices'],
        leader_indices=setup['leader_indices'],
        Omega=setup['Omega'],
        mu=0.01,
        varpi=0.5,
        phi_0=1.0,
    )
    times, pos_hist, errors, _, et_data = simulate_second_order_et(
        setup['controller'],
        setup['init_pos'],
        init_vel,
        setup['leader_traj'],
        (0, setup['T_total']),
        dt=0.02,
        et_manager=et_mgr,
    )

    cumulative_counts = _compute_cumulative_trigger_counts(times, et_data['trigger_log'], setup['follower_indices'])
    total_triggers = _compute_total_trigger_curve(cumulative_counts)
    continuous_baseline = np.arange(len(times), dtype=float) * len(setup['follower_indices'])

    return {
        'kind': 'et',
        'title': f"场景4：事件触发通信（平均通信率 {et_data['comm_rates']['mean']:.1f}%）",
        'output': 'afc_scene4_et',
        'times': times,
        'pos_hist': pos_hist,
        'errors': errors,
        'adj': setup['adj'],
        'leader_indices': setup['leader_indices'],
        'follower_indices': setup['follower_indices'],
        'phase_specs': _base_phase_specs(setup),
        'extra': {
            'trigger_log': et_data['trigger_log'],
            'comm_rates': et_data['comm_rates'],
            'cumulative_counts': cumulative_counts,
            'total_triggers': total_triggers,
            'continuous_baseline': continuous_baseline,
        },
    }


def run_rhf_scenario():
    nominal_pos, _, _ = double_pentagon(radius=1.0, height=1.0)

    leaders_p0 = [0, 1, 2, 5]
    omega_p0, info_p0 = compute_power_centric_stress_matrix(nominal_pos, leaders_p0)

    leaders_p1, _ = select_leaders_for_direction(nominal_pos, [0, 1, 0], n_leaders=4, d=3)
    omega_p1, info_p1 = compute_power_centric_stress_matrix(nominal_pos, leaders_p1)

    leaders_p2, _ = select_leaders_for_direction(nominal_pos, [-1, 0, 0], n_leaders=4, d=3)
    omega_p2, info_p2 = compute_power_centric_stress_matrix(nominal_pos, leaders_p2)

    targets_p0 = nominal_pos[leaders_p0] + np.array([0.5, 0.0, 0.0])
    targets_p1 = nominal_pos[leaders_p1] + np.array([0.5, 1.0, 0.0])
    targets_p2 = nominal_pos[leaders_p2] + np.array([-0.5, 1.0, 0.0])

    schedule = [
        {
            't_switch': 0.0,
            'leader_indices': leaders_p0,
            'leader_targets': targets_p0,
            't_transition': 3.0,
            'omega': omega_p0,
            'adj': info_p0['adj_matrix'],
            'label': 'Phase 0: +X 平移',
        },
        {
            't_switch': 35.0,
            'leader_indices': leaders_p1,
            'leader_targets': targets_p1,
            't_transition': 3.0,
            'omega': omega_p1,
            'adj': info_p1['adj_matrix'],
            'label': 'Phase 1: +Y 转弯',
        },
        {
            't_switch': 70.0,
            'leader_indices': leaders_p2,
            'leader_targets': targets_p2,
            't_transition': 3.0,
            'omega': omega_p2,
            'adj': info_p2['adj_matrix'],
            'label': 'Phase 2: -X 转弯(U 形)',
        },
    ]
    total = 105.0
    phase_specs = [
        (0.0, 35.0, '#D8EFF8', 'Phase 0: +X'),
        (35.0, 70.0, '#FFE7CC', 'Phase 1: +Y'),
        (70.0, total, '#DDF3E4', 'Phase 2: -X'),
    ]

    controller = AFCController(
        omega_p0,
        leaders_p0,
        gain=10.0,
        damping=1.0,
        u_max=CRAZYFLIE_COMM['max_velocity'],
        saturation_type='smooth',
    )
    np.random.seed(123)
    init_pos = nominal_pos.copy() + np.random.randn(10, 3) * 0.1
    init_vel = np.zeros((10, 3))

    times, pos_hist, errors, _, rhf_data = simulate_rhf(
        controller, init_pos, init_vel, nominal_pos, schedule, (0, total), dt=0.02
    )

    return {
        'kind': 'rhf',
        'title': '场景5：层级重组 RHF（U 形转弯）',
        'output': 'afc_scene5_rhf',
        'times': times,
        'pos_hist': pos_hist,
        'errors': errors,
        'adj': info_p0['adj_matrix'],
        'leader_indices': leaders_p0,
        'follower_indices': sorted(set(range(10)) - set(leaders_p0)),
        'phase_specs': phase_specs,
        'extra': {
            'schedule': schedule,
            'switch_times': [item['t_switch'] for item in schedule],
            'rhf_data': rhf_data,
            'phase_colors': SCENE_THEMES['rhf']['phase_colors'],
            'nominal_pos': nominal_pos,
        },
    }


def _style_axes(ax, theme, title=None):
    ax.set_facecolor(theme['panel_face'])
    if title is not None:
        ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.25)


def _save_animation(scenario):
    kind = scenario['kind']
    theme = SCENE_THEMES[kind]
    times = scenario['times']
    pos_hist = scenario['pos_hist']
    errors = scenario['errors']
    base_adj = scenario['adj']
    static_leaders = scenario['leader_indices']
    static_followers = scenario['follower_indices']
    phase_specs = scenario['phase_specs']
    title = scenario['title']
    out_name = scenario['output']
    extra = scenario.get('extra', {})

    n_steps, n_agents, _ = pos_hist.shape
    skip = max(1, n_steps // 650)
    frame_indices = list(range(0, n_steps, skip))
    if frame_indices[-1] != n_steps - 1:
        frame_indices.append(n_steps - 1)
    n_frames = len(frame_indices)
    print(f"  {title}: {n_steps} 步 -> {n_frames} 帧")

    fig = plt.figure(figsize=(16, 9), facecolor=theme['figure_face'])
    gs = fig.add_gridspec(
        3, 3,
        width_ratios=[1.7, 1.25, 1.05],
        height_ratios=[1.2, 1.2, 1.0],
        hspace=0.35,
        wspace=0.28,
    )
    ax3d = fig.add_subplot(gs[:2, :2], projection='3d')
    ax_a = fig.add_subplot(gs[0, 2])
    ax_b = fig.add_subplot(gs[1, 2])
    ax_err = fig.add_subplot(gs[2, :2])
    ax_info = fig.add_subplot(gs[2, 2])
    ax_info.axis('off')
    ax_info.set_facecolor(theme['panel_face'])

    ax3d.set_facecolor(theme['panel_face'])
    ax3d.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.65))  # type: ignore[attr-defined]
    ax3d.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.55))  # type: ignore[attr-defined]
    ax3d.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 0.45))  # type: ignore[attr-defined]
    ax3d.grid(True, alpha=0.18)

    _style_axes(ax_err, theme, '编队误差与阶段演化')
    _style_axes(ax_a, theme)
    _style_axes(ax_b, theme)

    x_min, x_max, y_min, y_max, z_min, z_max = _compute_bounds(pos_hist, frame_indices, margin=0.55)

    trail_len = min(70, n_frames)
    edge_lines = []
    highlight_lines = []
    phase_seg_lines = None

    leader_scatter = ax3d.scatter([], [], [], c=theme['leader'], s=130, marker='^',  # type: ignore[arg-type]
                                  edgecolors=theme['leader_edge'], linewidths=0.9, zorder=10)
    follower_scatter = ax3d.scatter([], [], [], c=theme['follower'], s=70, marker='o',  # type: ignore[arg-type]
                                    edgecolors='white', linewidths=0.35, zorder=9)

    if kind == 'rhf':
        phase_colors = extra['phase_colors']
        phase_seg_lines = []
        for _ in range(n_agents):
            per_agent = []
            for phase_color in phase_colors:
                seg_line, = ax3d.plot([], [], [], color=phase_color, linewidth=1.4, alpha=0.65)
                per_agent.append(seg_line)
            phase_seg_lines.append(per_agent)
        trail_lines = []
    else:
        trail_lines = []
        for agent in range(n_agents):
            line_color = theme['leader'] if agent in static_leaders else theme['follower']
            line, = ax3d.plot([], [], [], color=line_color, alpha=0.32, linewidth=1.0)
            trail_lines.append(line)

    ax_err.semilogy(times, errors + 1e-12, color=theme['aux'], linewidth=1.8)
    current_dot, = ax_err.semilogy([], [], 'o', color=theme['accent'], markersize=7)
    current_time_line = ax_err.axvline(0.0, color=theme['accent'], linestyle='--', linewidth=1.1, alpha=0.8)
    ax_err.set_xlim(0.0, times[-1])
    ax_err.set_ylim(max(float(np.min(errors + 1e-12)) * 0.7, 1e-5), float(np.max(errors + 1e-12)) * 2.0)
    ax_err.set_xlabel('时间 (s)')
    ax_err.set_ylabel('误差范数')
    for t0, t1, color, label in phase_specs:
        ax_err.axvspan(t0, t1, alpha=0.14, color=color)
        ax_err.text((t0 + t1) * 0.5, ax_err.get_ylim()[1] * 0.82, label,
                    ha='center', va='center', fontsize=7, alpha=0.75)

    info_text = ax_info.text(
        0.05, 0.95, '', transform=ax_info.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.9),
    )

    aux_current = {}

    if kind == 'baseline':
        ctrl_norms = extra['ctrl_norms']
        sat_ratio = extra['sat_ratio']
        max_ctrl = extra['max_ctrl']
        u_max = float(extra['u_max']) if extra['u_max'] is not None else None
        _style_axes(ax_a, theme, '控制输入峰值')
        _style_axes(ax_b, theme, '饱和比例')
        for idx, _ in enumerate(static_followers):
            alpha = 0.35 if idx != 0 else 0.75
            ax_a.plot(times, ctrl_norms[:, idx], color=theme['follower'], linewidth=0.8, alpha=alpha)
        ax_a.plot(times, max_ctrl, color=theme['accent'], linewidth=1.8, label='max ||u_i||')
        if u_max is not None:
            ax_a.axhline(u_max, color=theme['leader'], linestyle='--', linewidth=1.2, label=f'u_max={u_max:.1f}')
        ax_a.legend(fontsize=8, loc='upper right')
        ax_a.set_xlim(0.0, times[-1])
        ax_a.set_ylabel('m/s')
        ax_b.fill_between(times, sat_ratio, color=theme['accent'], alpha=0.28)
        ax_b.plot(times, sat_ratio, color=theme['accent'], linewidth=1.5)
        ax_b.set_xlim(0.0, times[-1])
        ax_b.set_ylim(0.0, 105.0)
        ax_b.set_ylabel('%')
        aux_current['a'] = ax_a.axvline(0.0, color=theme['accent'], linestyle='--', linewidth=1.0)
        aux_current['b'] = ax_b.axvline(0.0, color=theme['accent'], linestyle='--', linewidth=1.0)

    elif kind == 'cbf':
        _style_axes(ax_a, theme, '最小距离对比')
        _style_axes(ax_b, theme, '安全滤波激活 / 修正')
        ax_a.plot(extra['times_no'], extra['min_dist_no'], color='#4f6d7a', linewidth=1.2, label='无 CBF')
        ax_a.plot(times, extra['min_dist_yes'], color=theme['accent'], linewidth=1.8, label='有 CBF')
        ax_a.axhline(extra['d_safe'], color=theme['leader'], linestyle='--', linewidth=1.4, label='安全阈值')
        ax_a.fill_between(times, 0.0, extra['d_safe'], color='#f28482', alpha=0.12)
        ax_a.legend(fontsize=8, loc='lower right')
        ax_a.set_xlim(0.0, times[-1])
        ax_a.set_ylabel('m')
        ax_b.fill_between(times, extra['n_active'], color=theme['accent'], alpha=0.22)
        ax_b.plot(times, extra['n_active'], color=theme['accent'], linewidth=1.4, label='活跃约束数')
        ax_b_right = ax_b.twinx()
        ax_b_right.plot(times, extra['modifications'], color=theme['leader'], linewidth=1.2, label='||u_safe-u_nom||')
        ax_b.set_xlim(0.0, times[-1])
        ax_b.set_ylabel('约束数')
        ax_b_right.set_ylabel('m/s')
        aux_current['a'] = ax_a.axvline(0.0, color=theme['accent'], linestyle='--', linewidth=1.0)
        aux_current['b'] = ax_b.axvline(0.0, color=theme['accent'], linestyle='--', linewidth=1.0)
        aux_current['b_right'] = ax_b_right.axvline(0.0, color=theme['leader'], linestyle=':', linewidth=1.0)

    elif kind == 'eso':
        _style_axes(ax_a, theme, '扰动估计范数')
        _style_axes(ax_b, theme, '抗扰误差对比')
        ax_a.plot(times, extra['disturbance_true_norm'], color=theme['leader'], linewidth=1.5, label='真实扰动')
        ax_a.plot(times, extra['disturbance_est_norm'], color=theme['accent'], linewidth=1.5, linestyle='--', label='ESO 估计')
        ax_a.set_xlim(0.0, times[-1])
        ax_a.set_ylabel('m/s')
        ax_a.legend(fontsize=8, loc='upper right')
        ax_b.semilogy(extra['times_bl'], extra['err_bl'] + 1e-12, color='#6c757d', linewidth=1.2, label='无扰动')
        ax_b.semilogy(extra['times_nd'], extra['err_nd'] + 1e-12, color=theme['leader'], linewidth=1.4, label='有扰动无 ESO')
        ax_b.semilogy(times, errors + 1e-12, color=theme['accent'], linewidth=1.7, label='有扰动有 ESO')
        ax_b.set_xlim(0.0, times[-1])
        ax_b.set_ylabel('误差')
        ax_b.legend(fontsize=8, loc='upper right')
        aux_current['a'] = ax_a.axvline(0.0, color=theme['accent'], linestyle='--', linewidth=1.0)
        aux_current['b'] = ax_b.axvline(0.0, color=theme['accent'], linestyle='--', linewidth=1.0)
        wind_vec = extra['w_const']
        ax3d.quiver(
            x_min + 0.25, y_max - 0.55, z_max - 0.25,
            wind_vec[0], wind_vec[1], wind_vec[2],
            color=theme['accent'], linewidth=2.0, arrow_length_ratio=0.25,
        )
        ax3d.text(x_min + 0.25, y_max - 0.65, z_max - 0.15, 'Wind', color=theme['accent'], fontsize=9)

    elif kind == 'et':
        _style_axes(ax_a, theme, '触发事件时间线')
        _style_axes(ax_b, theme, '累计通信次数')
        for i_loc, fi in enumerate(static_followers):
            event_times = [ev[0] for ev in extra['trigger_log'] if ev[1] == fi]
            if event_times:
                ax_a.eventplot([event_times], lineoffsets=fi, linelengths=0.6, colors=[plt.colormaps['tab10'](i_loc % 10)])
        ax_a.set_yticks(static_followers)
        ax_a.set_xlim(0.0, times[-1])
        ax_a.set_ylabel('Follower ID')
        ax_b.plot(times, extra['total_triggers'], color=theme['accent'], linewidth=1.8, label='事件触发累计')
        ax_b.plot(times, extra['continuous_baseline'], color='#6c757d', linewidth=1.2, linestyle='--', label='连续通信基线')
        ax_b.set_xlim(0.0, times[-1])
        ax_b.set_ylabel('累计发送次数')
        ax_b.legend(fontsize=8, loc='upper left')
        aux_current['a'] = ax_a.axvline(0.0, color=theme['accent'], linestyle='--', linewidth=1.0)
        aux_current['b'] = ax_b.axvline(0.0, color=theme['accent'], linestyle='--', linewidth=1.0)

    elif kind == 'rhf':
        _style_axes(ax_a, theme, '切换误差与瞬态')
        _style_axes(ax_b, theme, '当前阶段通信拓扑')
        ax_a.plot(times, errors, color=theme['aux'], linewidth=1.6)
        for idx, st in enumerate(extra['switch_times']):
            if idx == 0:
                continue
            ax_a.axvline(st, color=extra['phase_colors'][idx], linestyle='--', linewidth=1.4, alpha=0.85)
        ax_a.set_xlim(0.0, times[-1])
        ax_a.set_ylabel('误差')
        aux_current['a'] = ax_a.axvline(0.0, color=theme['accent'], linestyle='--', linewidth=1.0)
        nominal_pos = extra['nominal_pos']
        ax_b.set_xlim(nominal_pos[:, 0].min() - 0.7, nominal_pos[:, 0].max() + 0.7)
        ax_b.set_ylim(nominal_pos[:, 1].min() - 0.7, nominal_pos[:, 1].max() + 0.7)
        ax_b.set_xlabel('X')
        ax_b.set_ylabel('Y')
        ax_b.set_aspect('equal')

    else:
        raise ValueError(f'未知场景类型: {kind}')

    def _update_scene_artists(fi, t, current_leaders, current_followers, pos):
        extra_lines = []

        if kind == 'baseline':
            aux_current['a'].set_xdata([t, t])
            aux_current['b'].set_xdata([t, t])
            extra_lines.append(f"u_peak: {extra['max_ctrl'][fi]:.3f} m/s")
            extra_lines.append(f"sat: {extra['sat_ratio'][fi]:.1f}%")

        elif kind == 'cbf':
            aux_current['a'].set_xdata([t, t])
            aux_current['b'].set_xdata([t, t])
            aux_current['b_right'].set_xdata([t, t])
            pair = extra['closest_pair'][fi]
            for line in highlight_lines:
                line.remove()
            highlight_lines.clear()
            i = int(pair[0])
            j = int(pair[1])
            hl, = ax3d.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], [pos[i, 2], pos[j, 2]],
                            color=theme['accent'], linewidth=3.0, alpha=0.95)
            highlight_lines.append(hl)
            extra_lines.append(f"closest: ({i},{j})")
            extra_lines.append(f"d_min: {extra['closest_pair_dist'][fi]:.3f} m")
            extra_lines.append(f"active: {int(extra['n_active'][fi])}")

        elif kind == 'eso':
            aux_current['a'].set_xdata([t, t])
            aux_current['b'].set_xdata([t, t])
            est_err_local = extra['per_agent_est_error'][fi]
            norm = float(np.max(extra['per_agent_est_error']) + 1e-12)
            colors = plt.colormaps['viridis'](np.clip(est_err_local / norm, 0.0, 1.0))
            follower_scatter.set_facecolors(colors)  # type: ignore[attr-defined]
            extra_lines.append(f"agent {extra['rep_agent']} |w-z2|: {extra['estimation_errors'][fi]:.3f}")
            extra_lines.append(f"w0: {extra['omega_o']:.1f} rad/s")

        elif kind == 'et':
            aux_current['a'].set_xdata([t, t])
            aux_current['b'].set_xdata([t, t])
            recent = {ev[1] for ev in extra['trigger_log'] if 0.0 <= t - ev[0] <= 0.8}
            trigger_colors = []
            trigger_sizes = []
            for follower in current_followers:
                if follower in recent:
                    trigger_colors.append(theme['accent'])
                    trigger_sizes.append(115.0)
                else:
                    trigger_colors.append(theme['follower'])
                    trigger_sizes.append(65.0)
            if len(trigger_colors) > 0:
                follower_scatter.set_facecolors(trigger_colors)  # type: ignore[attr-defined]
                follower_scatter.set_sizes(np.asarray(trigger_sizes))
            extra_lines.append(f"comm: {extra['comm_rates']['mean']:.1f}%")
            extra_lines.append(f"total trig: {int(extra['total_triggers'][fi])}")

        elif kind == 'rhf':
            aux_current['a'].set_xdata([t, t])
            ax_b.clear()
            _style_axes(ax_b, theme, '当前阶段通信拓扑')
            phase_idx = _current_rhf_phase(extra['schedule'], t)
            sched = extra['schedule'][phase_idx]
            nominal_pos = extra['nominal_pos']
            adj = sched['adj']
            phase_color = extra['phase_colors'][phase_idx]
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    if adj[i, j] > 0:
                        ax_b.plot(
                            [nominal_pos[i, 0], nominal_pos[j, 0]],
                            [nominal_pos[i, 1], nominal_pos[j, 1]],
                            color='#7a7a7a', linewidth=0.7, alpha=0.45,
                        )
            for i in range(n_agents):
                is_leader = i in sched['leader_indices']
                ax_b.scatter(
                    nominal_pos[i, 0], nominal_pos[i, 1],
                    s=180 if is_leader else 70,
                    c=phase_color if is_leader else '#d0d7de',
                    marker='*' if is_leader else 'o',
                    edgecolors='black',
                    linewidths=0.5,
                    zorder=5,
                )
                ax_b.text(nominal_pos[i, 0], nominal_pos[i, 1] + 0.08, str(i), fontsize=8, ha='center')
            ax_b.set_aspect('equal')
            ax_b.set_xlim(nominal_pos[:, 0].min() - 0.7, nominal_pos[:, 0].max() + 0.7)
            ax_b.set_ylim(nominal_pos[:, 1].min() - 0.7, nominal_pos[:, 1].max() + 0.7)
            extra_lines.append(sched['label'])
            extra_lines.append(f"leaders: {sched['leader_indices']}")

        return extra_lines

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
        fi = frame_indices[frame_idx]
        t = times[fi]
        pos = pos_hist[fi]

        if kind == 'rhf':
            phase_idx = _current_rhf_phase(extra['schedule'], t)
            current_adj = extra['schedule'][phase_idx]['adj']
            current_leaders = extra['schedule'][phase_idx]['leader_indices']
            current_followers = sorted(set(range(n_agents)) - set(current_leaders))
            leader_color = extra['phase_colors'][phase_idx]
            leader_scatter.set_facecolors([leader_color])  # type: ignore[attr-defined]
        else:
            current_adj = base_adj
            current_leaders = static_leaders
            current_followers = static_followers
            leader_scatter.set_facecolors([theme['leader']])  # type: ignore[attr-defined]

        leader_pos = pos[current_leaders]
        follower_pos = pos[current_followers]
        leader_scatter._offsets3d = (leader_pos[:, 0], leader_pos[:, 1], leader_pos[:, 2])  # type: ignore[attr-defined]
        follower_scatter._offsets3d = (follower_pos[:, 0], follower_pos[:, 1], follower_pos[:, 2])  # type: ignore[attr-defined]

        for line in edge_lines:
            line.remove()
        edge_lines = []
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                if current_adj[i, j] > 0:
                    line, = ax3d.plot(
                        [pos[i, 0], pos[j, 0]],
                        [pos[i, 1], pos[j, 1]],
                        [pos[i, 2], pos[j, 2]],
                        color='#8d99ae', linewidth=0.45, alpha=0.22,
                    )
                    edge_lines.append(line)

        if kind == 'rhf' and phase_seg_lines is not None:
            for agent in range(n_agents):
                for p_idx, sched in enumerate(extra['schedule']):
                    t_start = sched['t_switch']
                    if p_idx + 1 < len(extra['schedule']):
                        t_end = extra['schedule'][p_idx + 1]['t_switch']
                    else:
                        t_end = times[-1]
                    mask = (times[:fi + 1] >= t_start) & (times[:fi + 1] <= t_end)
                    seg = pos_hist[:fi + 1, agent][mask]
                    if len(seg) > 1:
                        phase_seg_lines[agent][p_idx].set_data(seg[:, 0], seg[:, 1])
                        phase_seg_lines[agent][p_idx].set_3d_properties(seg[:, 2])
                    else:
                        phase_seg_lines[agent][p_idx].set_data([], [])
                        phase_seg_lines[agent][p_idx].set_3d_properties([])
        else:
            start_idx = max(0, frame_idx - trail_len)
            idx_window = [frame_indices[k] for k in range(start_idx, frame_idx + 1)]
            for agent in range(n_agents):
                trail_lines[agent].set_data(pos_hist[idx_window, agent, 0], pos_hist[idx_window, agent, 1])
                trail_lines[agent].set_3d_properties(pos_hist[idx_window, agent, 2])

        ax3d.set_title(f'{title}\n{_phase_name(t, phase_specs)} | t = {t:.2f}s', fontsize=12)
        current_dot.set_data([t], [errors[fi] + 1e-12])
        current_time_line.set_xdata([t, t])

        extra_lines = _update_scene_artists(fi, t, current_leaders, current_followers, pos)

        info_lines = [
            f"Time: {t:.2f} s",
            f"Stage: {_phase_name(t, phase_specs)}",
            f"Error: {errors[fi]:.4f}",
            f"Leaders: {current_leaders}",
            f"Frame: {frame_idx + 1}/{n_frames}",
        ] + extra_lines
        info_text.set_text('\n'.join(info_lines))
        return []

    anim = FuncAnimation(fig, update, init_func=init, frames=n_frames, interval=50, blit=False)

    gif_path = video_path(f'{out_name}.gif')
    gif_ok = False
    try:
        anim.save(gif_path, writer=PillowWriter(fps=20), dpi=100)
        gif_ok = True
        print(f"  GIF 已保存: {gif_path}")
    except Exception as exc:
        print(f"  GIF 保存失败: {exc}")

    mp4_path = video_path(f'{out_name}.mp4')
    mp4_ok = False
    try:
        anim.save(mp4_path, writer=FFMpegWriter(fps=20, bitrate=2000), dpi=120)
        mp4_ok = True
        print(f"  MP4 已保存: {mp4_path}")
    except Exception as exc:
        print(f"  MP4 保存失败，可能缺少 ffmpeg: {exc}")

    preview_path = figure_path(f'{out_name}_poster.png')
    fig.savefig(preview_path, dpi=160, bbox_inches='tight')

    arch = SimArchive(tag=out_name)
    arch.save_figure(fig, f'{out_name}_poster')
    arch.save_arrays(times=times, positions=pos_hist, errors=errors)
    arch.save_params({
        'kind': kind,
        'title': title,
        'output': out_name,
        'n_agents': n_agents,
        'leader_indices': static_leaders,
        'follower_indices': static_followers,
        'T_total': float(times[-1]),
        'n_frames': n_frames,
        'phase_specs': phase_specs,
        'extra_keys': sorted(list(extra.keys())),
    })
    if gif_ok:
        shutil.copy(gif_path, arch._tmp_dir)
    if mp4_ok:
        shutil.copy(mp4_path, arch._tmp_dir)
    zip_path = arch.finalize()
    print(f"  存档已完成: {zip_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='AFC 多场景动画系统')
    parser.add_argument('--scenario', type=int, default=0, help='场景编号 1-5，0 表示全部生成')
    args = parser.parse_args()

    print('初始化基础仿真参数...')
    setup = _build_base_setup()

    runners = {
        1: lambda: run_baseline_scenario(setup),
        2: lambda: run_cbf_scenario(setup),
        3: lambda: run_eso_scenario(setup),
        4: lambda: run_et_scenario(setup),
        5: run_rhf_scenario,
    }

    targets = list(runners.keys()) if args.scenario == 0 else [args.scenario]
    for idx in targets:
        runner = runners.get(idx)
        if runner is None:
            print(f'未知场景编号 {idx}，跳过')
            continue
        print('\n' + '=' * 56)
        print(f'生成场景 {idx} 动画...')
        scenario = runner()
        _save_animation(scenario)

    print('\n全部完成。')


if __name__ == '__main__':
    main()
