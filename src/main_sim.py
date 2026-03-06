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
    在仓库根目录运行：
    python src/main_sim.py
    python src/main_sim.py --scenario pyramid

    或者在 src 目录运行：
    python main_sim.py
"""

import argparse
import json
import os

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
try:
    matplotlib.use('TkAgg')
except Exception:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from stress_matrix import (compute_stress_matrix, validate_stress_matrix,
                           print_validation, compute_sparse_stress_matrix,
                           compute_power_centric_stress_matrix)
from formation import (
    aerial_pyramid_10, double_pentagon, affine_transform, scale_matrix,
    rotation_matrix_z, create_leader_trajectory, graph_info, smoothstep,
    CRAZYFLIE_COMM, check_affine_span, build_power_centric_topology,
    select_leaders_for_direction, compute_dwell_time,
)
from afc_controller import AFCController
from archive import SimArchive
from collision_avoidance import CBFSafetyFilter, CRAZYFLIE_SAFETY
from disturbance_observer import ExtendedStateObserver, WindDisturbance
from event_trigger import EventTriggerManager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
PYRAMID_CONFIG_PATH = os.path.join(ROOT_DIR, 'pyramid_mission_config.json')
for _dir in (OUTPUT_DIR, FIGURE_DIR):
    os.makedirs(_dir, exist_ok=True)


DEFAULT_PYRAMID_MISSION_CONFIG = {
    'gain': 5.5,
    'd_safe': 0.16,
    'd_activate': 0.9,
    'cbf_gamma': 6.0,
    'total_time': 34.0,
    'init_radius': 1.9,
    'init_noise_std': 0.04,
    'rng_seed': 2026,
    'wind_const': [0.08, -0.03, 0.02],
    'ou_theta': 0.4,
    'ou_sigma': 0.04,
    'wind_seed': 321,
    'eso_omega': 8.0,
    'et_mu': 0.015,
    'et_varpi': 0.45,
    'et_phi_0': 1.0,
}


def figure_path(name):
    return os.path.join(FIGURE_DIR, name)


def load_pyramid_mission_config():
    """加载综合金字塔任务默认参数，若无配置文件则回退到内置默认值。"""
    config = DEFAULT_PYRAMID_MISSION_CONFIG.copy()
    if os.path.exists(PYRAMID_CONFIG_PATH):
        with open(PYRAMID_CONFIG_PATH, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            config.update(loaded)
    return config


def save_pyramid_mission_config(config):
    """保存综合金字塔任务默认参数到独立配置文件。"""
    merged = DEFAULT_PYRAMID_MISSION_CONFIG.copy()
    merged.update(config)
    with open(PYRAMID_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    return PYRAMID_CONFIG_PATH


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


def simulate_first_order_cbf(controller, initial_positions,
                              leader_traj_func, t_span, dt=0.02,
                              cbf_filter=None):
    """
    一阶积分器仿真（前向 Euler 积分），可选 CBF 碰撞避免。

    使用前向 Euler 而非 RK45，以配合 CBF 的离散时间安全保证。
    当 cbf_filter=None 时等价于普通 Euler 积分（用于公平对比）。

    Parameters
    ----------
    controller : AFCController
    initial_positions : ndarray (n, d)
    leader_traj_func : callable
    t_span : (float, float)
    dt : float
    cbf_filter : CBFSafetyFilter or None

    Returns
    -------
    times : ndarray (n_steps,)
    positions_history : ndarray (n_steps, n, d)
    errors : ndarray (n_steps,)
    control_inputs : ndarray (n_steps, n_f, d)
    cbf_data : dict
        min_distances, modifications, n_active
    """
    n = controller.n
    d = initial_positions.shape[1]
    f_idx = controller.follower_indices
    l_idx = controller.leader_indices
    n_f = controller.n_f

    times = np.arange(t_span[0], t_span[1] + dt / 2, dt)
    n_steps = len(times)

    positions_history = np.zeros((n_steps, n, d))
    errors = np.zeros(n_steps)
    control_inputs = np.zeros((n_steps, n_f, d))
    min_distances = np.zeros(n_steps)
    cbf_modifications = np.zeros(n_steps)
    n_active_constraints = np.zeros(n_steps, dtype=int)

    positions = initial_positions.copy()

    for idx in range(n_steps):
        t = times[idx]
        p_l = leader_traj_func(t)
        positions[l_idx] = p_l
        positions_history[idx] = positions.copy()

        p_f = positions[f_idx]

        # AFC 标称控制 + 饱和
        u_nom = -controller.gain * (controller.Omega_ff @ p_f
                                    + controller.Omega_fl @ p_l)
        u_nom = controller.saturate(u_nom)

        # CBF 安全滤波
        if cbf_filter is not None:
            eps = 1e-4
            v_l = (leader_traj_func(t + eps) - p_l) / eps
            u_safe, cbf_info = cbf_filter.filter(positions, u_nom, v_l)
            cbf_modifications[idx] = cbf_info['modification_norm']
            n_active_constraints[idx] = cbf_info['n_constraints']
        else:
            u_safe = u_nom

        control_inputs[idx] = u_safe

        # 最小距离
        min_d, _ = CBFSafetyFilter.min_distance(positions)
        min_distances[idx] = min_d

        # 编队误差
        p_f_star = controller.steady_state(p_l)
        errors[idx] = np.linalg.norm(p_f - p_f_star)

        # 前向 Euler 积分
        if idx < n_steps - 1:
            positions[f_idx] = p_f + dt * u_safe

    cbf_data = {
        'min_distances': min_distances,
        'modifications': cbf_modifications,
        'n_active': n_active_constraints,
    }
    return times, positions_history, errors, control_inputs, cbf_data


def simulate_first_order_eso(controller, initial_positions, leader_traj_func,
                              t_span, dt=0.02, wind=None, eso=None):
    """
    一阶积分器仿真，含外部扰动和可选 ESO 补偿。

    动力学：ṗ_f = u_f + w_f(t)
    ESO 补偿：u_f = u_nom − ẑ_w  (当 eso 不为 None 时)

    Parameters
    ----------
    controller : AFCController
    initial_positions : ndarray (n, d)
    leader_traj_func : callable
    t_span : (float, float)
    dt : float
    wind : WindDisturbance or None
        风场扰动模型
    eso : ExtendedStateObserver or None
        扩展状态观测器（None 则不补偿）

    Returns
    -------
    times : ndarray (n_steps,)
    positions_history : ndarray (n_steps, n, d)
    errors : ndarray (n_steps,)
    control_inputs : ndarray (n_steps, n_f, d)
    eso_data : dict
        disturbances_true, disturbances_est, estimation_errors
    """
    n = controller.n
    d = initial_positions.shape[1]
    f_idx = controller.follower_indices
    l_idx = controller.leader_indices
    n_f = controller.n_f

    times = np.arange(t_span[0], t_span[1] + dt / 2, dt)
    n_steps = len(times)

    positions_history = np.zeros((n_steps, n, d))
    errors = np.zeros(n_steps)
    control_inputs = np.zeros((n_steps, n_f, d))
    disturbances_true = np.zeros((n_steps, n_f, d))
    disturbances_est = np.zeros((n_steps, n_f, d))

    positions = initial_positions.copy()

    # 初始化 ESO
    if eso is not None:
        eso.reset(positions[f_idx])
    if wind is not None:
        wind.reset()

    for idx in range(n_steps):
        t = times[idx]
        p_l = leader_traj_func(t)
        positions[l_idx] = p_l
        positions_history[idx] = positions.copy()

        p_f = positions[f_idx]

        # 获取当前风扰动
        if wind is not None:
            w = wind.current()
            disturbances_true[idx] = w
        else:
            w = np.zeros((n_f, d))

        # AFC 标称控制
        u_nom = -controller.gain * (controller.Omega_ff @ p_f
                                    + controller.Omega_fl @ p_l)

        # ESO 补偿
        if eso is not None:
            w_hat = eso.disturbance_estimate()
            disturbances_est[idx] = w_hat
            u_compensated = u_nom - w_hat
        else:
            u_compensated = u_nom

        # 饱和
        u_sat = controller.saturate(u_compensated)
        control_inputs[idx] = u_sat

        # 编队误差
        p_f_star = controller.steady_state(p_l)
        errors[idx] = np.linalg.norm(p_f - p_f_star)

        # 前向 Euler 积分（含扰动）
        if idx < n_steps - 1:
            positions[f_idx] = p_f + dt * (u_sat + w)

            # 推进风场
            if wind is not None:
                wind.step(dt)

            # 更新 ESO（用实际施加的控制和新位置）
            if eso is not None:
                eso.update(positions[f_idx], u_sat, dt)

    est_errors = np.linalg.norm(disturbances_true - disturbances_est,
                                axis=(1, 2))
    eso_data = {
        'disturbances_true': disturbances_true,
        'disturbances_est': disturbances_est,
        'estimation_errors': est_errors,
    }
    return times, positions_history, errors, control_inputs, eso_data


def simulate_second_order_et(controller, initial_positions,
                              initial_velocities, leader_traj_func,
                              t_span, dt=0.02, et_manager=None):
    """
    二阶积分器仿真 + 自适应事件触发通信。

    动力学：p̈_f = u_f
    控制律：u_i = -K_p Σ_j ω_ij(p_i - p̂_j) - K_d v_i
      位置编队偏差项使用邻居广播位置 p̂_j，阻尼项使用本地实时速度 v_i。

    矩阵形式 (等价)：
      u_f = -K_p (Ω_ff p̂_f + Ω_fl p̂_l) - K_p Ω_ff (p_f - p̂_f) - K_d v_f

    Parameters
    ----------
    controller : AFCController
    initial_positions : ndarray (n, d)
    initial_velocities : ndarray (n, d) or None
    leader_traj_func : callable
    t_span : (float, float)
    dt : float
    et_manager : EventTriggerManager or None
        None 时退化为连续通信（二阶积分器 Euler 积分基线）

    Returns
    -------
    times : ndarray (n_steps,)
    positions_history : ndarray (n_steps, n, d)
    errors : ndarray (n_steps,)
    control_inputs : ndarray (n_steps, n_f, d)
    et_data : dict
        trigger_log, trigger_counts, comm_rates, phi_history
    """
    n = controller.n
    d = initial_positions.shape[1]
    f_idx = controller.follower_indices
    l_idx = controller.leader_indices
    n_f = controller.n_f

    Omega_ff = controller.Omega_ff
    Omega_fl = controller.Omega_fl

    times = np.arange(t_span[0], t_span[1] + dt / 2, dt)
    n_steps = len(times)

    positions_history = np.zeros((n_steps, n, d))
    velocities_history = np.zeros((n_steps, n, d))
    errors = np.zeros(n_steps)
    control_inputs = np.zeros((n_steps, n_f, d))
    phi_history = np.zeros((n_steps, n_f))

    positions = initial_positions.copy()
    velocities = (initial_velocities.copy() if initial_velocities is not None
                  else np.zeros_like(initial_positions))

    # 初始化 ET 管理器
    if et_manager is not None:
        et_manager.reset(positions)

    for idx in range(n_steps):
        t = times[idx]
        p_l = leader_traj_func(t)
        positions[l_idx] = p_l
        velocities[l_idx] = 0.0  # Leader 速度由轨迹决定（忽略）
        positions_history[idx] = positions.copy()
        velocities_history[idx] = velocities.copy()

        p_f = positions[f_idx]
        v_f = velocities[f_idx]

        if et_manager is not None:
            # Leader 广播位置直接更新
            et_manager.update_leaders(p_l)
            # 检查 follower 触发条件
            triggered, errors_sq = et_manager.check_and_trigger(t, positions)
            phi_history[idx] = et_manager.phi.copy()

            # 使用广播位置计算编队偏差
            p_f_hat = et_manager.p_hat[f_idx]
            p_l_hat = et_manager.p_hat[l_idx]

            # u = -K_p(Ω_ff p̂_f + Ω_fl p̂_l) - K_p Ω_ff (p_f - p̂_f) - K_d v_f
            #   = -K_p(Ω_ff p_f + Ω_fl p̂_l + Ω_ff(p̂_f - p_f) + Ω_ff(p_f - p̂_f)) 简化为
            #   = -K_p(Ω_ff p_f + Ω_fl p̂_l) - K_d v_f
            # 但邻居 j 的位置仍是 p̂_j，所以展开：
            u_f = -controller.gain * (Omega_ff @ p_f_hat + Omega_fl @ p_l_hat)
            # 修正自身位置：自身用实时值 p_i，邻居用广播值 p̂_j
            # 即 Ω_ff 行的对角项用 p_i - p̂_i 修正
            e_f = p_f - p_f_hat  # (n_f, d) 自身广播误差
            u_f -= controller.gain * np.diag(np.diag(Omega_ff)) @ e_f

            # 阻尼项（本地实时速度）
            u_f -= controller.damping * v_f
            u_f = controller.saturate(u_f)

            # 更新自适应参数
            et_manager.update_phi(errors_sq, dt)
        else:
            # 连续通信基线
            u_f = -controller.gain * (Omega_ff @ p_f + Omega_fl @ p_l)
            u_f -= controller.damping * v_f
            u_f = controller.saturate(u_f)

        control_inputs[idx] = u_f

        # 编队误差
        p_f_star = controller.steady_state(p_l)
        errors[idx] = np.linalg.norm(p_f - p_f_star)

        # 前向 Euler 积分（二阶）
        if idx < n_steps - 1:
            velocities[f_idx] = v_f + dt * u_f
            positions[f_idx] = p_f + dt * velocities[f_idx]

    # ET 统计
    if et_manager is not None:
        comm_rates = et_manager.communication_rates()
        et_data = {
            'trigger_log': et_manager.trigger_log.copy(),
            'trigger_counts': et_manager.trigger_counts.copy(),
            'comm_rates': comm_rates,
            'phi_history': phi_history,
        }
    else:
        et_data = {
            'trigger_log': [],
            'trigger_counts': np.zeros(n, dtype=int),
            'comm_rates': {'per_agent': np.zeros(n_f), 'mean': 100.0,
                           'total_triggers': n_f * n_steps,
                           'total_possible': n_f * n_steps},
            'phi_history': np.zeros((n_steps, n_f)),
        }

    return times, positions_history, errors, control_inputs, et_data


def simulate_integrated_first_order_rhf(controller, initial_positions,
                                        nominal_pos, rhf_schedule, t_span,
                                        dt=0.02, wind=None, eso_omega=8.0,
                                        cbf_kwargs=None, et_kwargs=None):
    """
    综合场景仿真：一阶 AFC + 输入饱和 + CBF + ESO + ET + RHF。

    说明：
      - 采用一阶位置动力学，以保持 CBF 安全滤波的理论假设一致；
      - 事件触发模块用于稀疏化邻居位置广播，控制律使用广播位置；
      - RHF 在阶段切换时重构 leader 集合、拓扑和应力矩阵。
    """
    n = controller.n
    d_dim = initial_positions.shape[1]

    times = np.arange(t_span[0], t_span[1] + dt / 2, dt)
    n_steps = len(times)

    positions_history = np.zeros((n_steps, n, d_dim))
    errors = np.zeros(n_steps)
    control_inputs = np.zeros((n_steps, n, d_dim))
    min_distances = np.zeros(n_steps)
    cbf_modifications = np.zeros(n_steps)
    n_active_constraints = np.zeros(n_steps, dtype=int)
    disturbances_true = np.zeros((n_steps, n, d_dim))
    disturbances_est = np.zeros((n_steps, n, d_dim))
    phase_indices = np.zeros(n_steps, dtype=int)
    phi_history = np.full((n_steps, n), np.nan)

    positions = initial_positions.copy()
    schedule = sorted(rhf_schedule, key=lambda item: item['t_switch'])

    switch_log = []
    leader_history = []
    trigger_log = []
    trigger_counts = np.zeros(n, dtype=int)
    follower_active_steps = np.zeros(n, dtype=int)

    current_phase = -1
    prev_leader_targets = positions[controller.leader_indices].copy()
    current_leader_targets = prev_leader_targets.copy()
    transition_start = t_span[0]
    transition_end = t_span[0]

    cbf_kwargs = cbf_kwargs or {}
    et_kwargs = et_kwargs or {}

    if wind is not None:
        wind.reset()

    cbf_filter = CBFSafetyFilter(
        n_agents=n,
        leader_indices=controller.leader_indices,
        d_safe=cbf_kwargs.get('d_safe', CRAZYFLIE_SAFETY['safety_distance_m']),
        gamma=cbf_kwargs.get('gamma', CRAZYFLIE_SAFETY['cbf_gamma']),
        d_activate=cbf_kwargs.get('d_activate', CRAZYFLIE_SAFETY['activate_distance_m']),
    )
    et_manager = EventTriggerManager(
        n_agents=n,
        d=d_dim,
        follower_indices=controller.follower_indices,
        leader_indices=controller.leader_indices,
        Omega=controller.Omega,
        mu=et_kwargs.get('mu', 0.01),
        varpi=et_kwargs.get('varpi', 0.5),
        phi_0=et_kwargs.get('phi_0', 1.0),
    )
    et_manager.reset(positions)
    eso = ExtendedStateObserver(controller.n_f, dim=d_dim, omega_o=eso_omega)
    eso.reset(positions[controller.follower_indices])
    leader_history.append((t_span[0], controller.leader_indices.copy()))

    def leader_velocity(t, p_start, p_end, t_start, t_end):
        if t <= t_start or t >= t_end or t_end <= t_start:
            return np.zeros_like(p_end)
        x = (t - t_start) / (t_end - t_start)
        alpha_dot = 6.0 * x * (1.0 - x) / (t_end - t_start)
        return alpha_dot * (p_end - p_start)

    for idx in range(n_steps):
        t = times[idx]

        while (current_phase + 1 < len(schedule)
               and t >= schedule[current_phase + 1]['t_switch']):
            current_phase += 1
            sched = schedule[current_phase]
            pre_error = errors[max(0, idx - 1)] if idx > 0 else 0.0
            prev_follower_indices = controller.follower_indices.copy()
            prev_eso = eso

            prev_leader_targets = positions[sched['leader_indices']].copy()
            switch_info = controller.update_omega(
                sched['omega'], sched['leader_indices']
            )
            current_leader_targets = sched['leader_targets']
            transition_start = sched['t_switch']
            transition_end = sched['t_switch'] + sched.get('t_transition', 5.0)

            cbf_filter = CBFSafetyFilter(
                n_agents=n,
                leader_indices=controller.leader_indices,
                d_safe=cbf_kwargs.get('d_safe', CRAZYFLIE_SAFETY['safety_distance_m']),
                gamma=cbf_kwargs.get('gamma', CRAZYFLIE_SAFETY['cbf_gamma']),
                d_activate=cbf_kwargs.get('d_activate', CRAZYFLIE_SAFETY['activate_distance_m']),
            )
            et_manager = EventTriggerManager(
                n_agents=n,
                d=d_dim,
                follower_indices=controller.follower_indices,
                leader_indices=controller.leader_indices,
                Omega=controller.Omega,
                mu=et_kwargs.get('mu', 0.01),
                varpi=et_kwargs.get('varpi', 0.5),
                phi_0=et_kwargs.get('phi_0', 1.0),
            )
            et_manager.reset(positions)

            eso = ExtendedStateObserver(controller.n_f, dim=d_dim, omega_o=eso_omega)
            eso.reset(positions[controller.follower_indices])
            prev_index_map = {fi: i_loc for i_loc, fi in enumerate(prev_follower_indices)}
            for i_loc, fi in enumerate(controller.follower_indices):
                old_loc = prev_index_map.get(fi)
                if old_loc is not None:
                    eso.z1[i_loc] = prev_eso.z1[old_loc]
                    eso.z2[i_loc] = prev_eso.z2[old_loc]

            switch_log.append({
                't_switch': sched['t_switch'],
                'label': sched.get('label', f'phase_{current_phase}'),
                'switch_info': switch_info,
                'pre_switch_error': pre_error,
                'step_idx': idx,
            })
            leader_history.append((sched['t_switch'], controller.leader_indices.copy()))

        if t <= transition_start:
            p_l = prev_leader_targets.copy()
        elif t >= transition_end:
            p_l = current_leader_targets.copy()
        else:
            alpha = smoothstep(t, transition_start, transition_end)
            p_l = (1.0 - alpha) * prev_leader_targets + alpha * current_leader_targets

        l_idx = controller.leader_indices
        f_idx = controller.follower_indices
        positions[l_idx] = p_l
        positions_history[idx] = positions.copy()
        phase_indices[idx] = max(current_phase, 0)

        follower_active_steps[f_idx] += 1
        et_manager.update_leaders(p_l)
        triggered, errors_sq = et_manager.check_and_trigger(t, positions)
        for fi in triggered:
            trigger_counts[fi] += 1
            trigger_log.append((t, fi))
        et_manager.update_phi(errors_sq, dt)
        for i_loc, fi in enumerate(f_idx):
            phi_history[idx, fi] = et_manager.phi[i_loc]

        p_f = positions[f_idx]
        p_f_hat = et_manager.p_hat[f_idx]
        p_l_hat = et_manager.p_hat[l_idx]

        u_nom = -controller.gain * (controller.Omega_ff @ p_f_hat + controller.Omega_fl @ p_l_hat)
        diag_ff = np.diag(np.diag(controller.Omega_ff))
        u_nom -= controller.gain * (diag_ff @ (p_f - p_f_hat))

        if wind is not None:
            w_full = wind.current()
            w_f = w_full[f_idx]
            disturbances_true[idx, f_idx] = w_f
        else:
            w_f = np.zeros((controller.n_f, d_dim))

        w_hat = eso.disturbance_estimate()
        disturbances_est[idx, f_idx] = w_hat
        u_cmd = controller.saturate(u_nom - w_hat)

        v_l = leader_velocity(t, prev_leader_targets, current_leader_targets,
                              transition_start, transition_end)
        u_safe, cbf_info = cbf_filter.filter(positions, u_cmd, v_l)
        cbf_modifications[idx] = cbf_info['modification_norm']
        n_active_constraints[idx] = cbf_info['n_constraints']
        control_inputs[idx, f_idx] = u_safe

        min_distances[idx], _ = CBFSafetyFilter.min_distance(positions)
        p_f_star = controller.steady_state(p_l)
        errors[idx] = np.linalg.norm(p_f - p_f_star)

        if idx < n_steps - 1:
            positions[f_idx] = p_f + dt * (u_safe + w_f)
            if wind is not None:
                wind.step(dt)
            eso.update(positions[f_idx], u_safe, dt)

    for log_entry in switch_log:
        step = log_entry['step_idx']
        recovery_threshold = max(0.35, log_entry['pre_switch_error'] * 1.25)
        recovery_step = None
        for s in range(step, min(step + int(40.0 / dt), n_steps)):
            if errors[s] < recovery_threshold:
                recovery_step = s
                break
        log_entry['recovery_time'] = (
            (times[recovery_step] - log_entry['t_switch'])
            if recovery_step is not None else float('inf')
        )
        log_entry['post_switch_peak_error'] = float(
            errors[step:min(step + int(10.0 / dt), n_steps)].max()
        )

    per_agent_comm = np.zeros(n)
    valid_agents = follower_active_steps > 0
    per_agent_comm[valid_agents] = (
        trigger_counts[valid_agents] / np.maximum(follower_active_steps[valid_agents], 1) * 100.0
    )
    mean_comm = float(per_agent_comm[valid_agents].mean()) if np.any(valid_agents) else 0.0

    mission_data = {
        'switch_log': switch_log,
        'leader_history': leader_history,
        'schedule': schedule,
        'phase_indices': phase_indices,
        'min_distances': min_distances,
        'cbf_modifications': cbf_modifications,
        'n_active_constraints': n_active_constraints,
        'disturbances_true': disturbances_true,
        'disturbances_est': disturbances_est,
        'trigger_log': trigger_log,
        'trigger_counts': trigger_counts,
        'phi_history': phi_history,
        'follower_active_steps': follower_active_steps,
        'comm_rates': {
            'per_agent': per_agent_comm,
            'mean': mean_comm,
            'total_triggers': int(trigger_counts.sum()),
            'total_possible': int(follower_active_steps.sum()),
        },
        'estimation_error': np.linalg.norm(disturbances_true - disturbances_est, axis=(1, 2)),
        'n_switches': len(switch_log),
    }

    return times, positions_history, errors, control_inputs, mission_data


# ============================================================
# 层级重组 (RHF) 仿真引擎
# ============================================================

def simulate_rhf(controller, initial_positions, initial_velocities,
                 nominal_pos, rhf_schedule, t_span, dt=0.02):
    """
    层级重组 (Reconfigurable Hierarchical Formation) 仿真。

    在仿真过程中按 rhf_schedule 定义的时间表切换层级：
      - 切换 leader/follower 角色
      - 重构通信拓扑（power-centric）
      - 重新计算应力矩阵
      - 更新控制器参数

    二阶积分器模型：p̈_f = -K_p(Ω_ff p_f + Ω_fl p_l) - K_d v_f

    参考: Li & Dong (2024) "A Flexible and Resilient Formation Approach
          based on Hierarchical Reorganization", arXiv:2406.11219

    Parameters
    ----------
    controller : AFCController
        初始控制器（将在运行中被 update_omega 修改）
    initial_positions : ndarray (n, d)
        初始位置
    initial_velocities : ndarray (n, d) or None
        初始速度
    nominal_pos : ndarray (n, d)
        标称编队位置（用于计算各层级的 leader 目标位置）
    rhf_schedule : list of dict
        层级切换时间表，每项包含：
        {
            't_switch': float,            # 切换时刻
            'leader_indices': list[int],  # 新 leader 集合
            'leader_targets': ndarray,    # 新 leader 的目标位置 (n_l, d)
            't_transition': float,        # leader 位置过渡时间
            'omega': ndarray,             # 预计算的应力矩阵
            'adj': ndarray,               # 邻接矩阵
            'label': str,                 # 阶段标签
        }
    t_span : (float, float)
    dt : float

    Returns
    -------
    times : ndarray (n_steps,)
    positions_history : ndarray (n_steps, n, d)
    errors : ndarray (n_steps,)
    control_inputs_history : list of ndarray
        每步的控制输入（因 n_f 可能变化，用 list 存储）
    rhf_data : dict
        层级切换的详细数据
    """
    n = controller.n
    d_dim = initial_positions.shape[1]

    times = np.arange(t_span[0], t_span[1] + dt / 2, dt)
    n_steps = len(times)

    positions_history = np.zeros((n_steps, n, d_dim))
    errors = np.zeros(n_steps)
    # 控制输入：因 follower 集合可能变化，记录全体 agent 的加速度
    accel_history = np.zeros((n_steps, n, d_dim))

    positions = initial_positions.copy()
    velocities = (initial_velocities.copy() if initial_velocities is not None
                  else np.zeros_like(initial_positions))

    # 状态追踪
    switch_log = []           # 每次切换的记录
    leader_history = []       # (t, leader_indices) 时间线
    current_phase = -1        # 当前阶段索引
    phase_start_time = t_span[0]

    # 预处理 schedule：按时间排序
    schedule = sorted(rhf_schedule, key=lambda s: s['t_switch'])

    # Leader 目标位置管理
    # 初始阶段：使用 controller 当前的 leader
    current_leader_targets = positions[controller.leader_indices].copy()
    prev_leader_targets = current_leader_targets.copy()
    transition_start = t_span[0]
    transition_end = t_span[0]

    leader_history.append((t_span[0], controller.leader_indices.copy()))

    for idx in range(n_steps):
        t = times[idx]

        # ---- 检查是否需要切换层级 ----
        while (current_phase + 1 < len(schedule) and
               t >= schedule[current_phase + 1]['t_switch']):
            current_phase += 1
            sched = schedule[current_phase]

            # 记录切换前的状态
            pre_error = errors[max(0, idx - 1)] if idx > 0 else 0.0

            # 保存切换前的 leader 位置（作为过渡起点）
            prev_leader_targets = positions[sched['leader_indices']].copy()

            # 执行切换
            switch_info = controller.update_omega(
                sched['omega'], sched['leader_indices']
            )

            # 设置 leader 目标和过渡
            current_leader_targets = sched['leader_targets']
            transition_start = sched['t_switch']
            transition_end = sched['t_switch'] + sched.get('t_transition', 5.0)

            switch_log.append({
                't_switch': sched['t_switch'],
                'label': sched.get('label', f'phase_{current_phase}'),
                'switch_info': switch_info,
                'pre_switch_error': pre_error,
                'step_idx': idx,
            })
            leader_history.append((sched['t_switch'],
                                   controller.leader_indices.copy()))

            phase_start_time = sched['t_switch']

        # ---- Leader 位置插值（smoothstep 过渡） ----
        if t <= transition_start:
            p_l = prev_leader_targets.copy()
        elif t >= transition_end:
            p_l = current_leader_targets.copy()
        else:
            alpha = smoothstep(t, transition_start, transition_end)
            p_l = (1 - alpha) * prev_leader_targets + alpha * current_leader_targets

        # ---- 获取当前 leader/follower 索引 ----
        l_idx = controller.leader_indices
        f_idx = controller.follower_indices
        n_f = controller.n_f

        # ---- 更新 leader 位置 ----
        positions[l_idx] = p_l

        # ---- 记录状态 ----
        positions_history[idx] = positions.copy()

        # ---- 计算 follower 控制输入 ----
        p_f = positions[f_idx]
        v_f = velocities[f_idx]

        u_f = -controller.gain * (controller.Omega_ff @ p_f
                                  + controller.Omega_fl @ p_l)
        u_f -= controller.damping * v_f
        u_f = controller.saturate(u_f)

        # 记录控制输入
        for k, fi in enumerate(f_idx):
            accel_history[idx, fi] = u_f[k]

        # ---- 编队误差 ----
        p_f_star = controller.steady_state(p_l)
        errors[idx] = np.linalg.norm(p_f - p_f_star)

        # ---- 前向 Euler 积分 ----
        if idx < n_steps - 1:
            velocities[f_idx] = v_f + dt * u_f
            positions[f_idx] = p_f + dt * velocities[f_idx]
            # Leader 速度近似为零（由外部轨迹驱动）
            velocities[l_idx] = 0.0

    # ---- 记录每次切换后的误差恢复 ----
    for log_entry in switch_log:
        step = log_entry['step_idx']
        # 找到切换后误差恢复到阈值以下的时刻
        recovery_threshold = max(0.5, log_entry['pre_switch_error'] * 1.5)
        recovery_step = None
        for s in range(step, min(step + int(60.0 / dt), n_steps)):
            if errors[s] < recovery_threshold:
                recovery_step = s
                break
        log_entry['recovery_time'] = (
            (times[recovery_step] - log_entry['t_switch'])
            if recovery_step is not None else float('inf')
        )
        log_entry['post_switch_peak_error'] = float(
            errors[step:min(step + int(15.0 / dt), n_steps)].max()
        )

    rhf_data = {
        'switch_log': switch_log,
        'leader_history': leader_history,
        'accel_history': accel_history,
        'schedule': schedule,
        'n_switches': len(switch_log),
    }

    return times, positions_history, errors, accel_history, rhf_data


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


def build_base_animation_setup():
    """构建场景 1-4 共用的基础编队与 Leader 轨迹。"""
    nominal_pos, leader_indices, adj = double_pentagon(radius=1.0, height=1.0)
    follower_indices = sorted(set(range(10)) - set(leader_indices))

    omega, _ = compute_stress_matrix(nominal_pos, adj, leader_indices, method='optimize')
    controller = AFCController(
        omega,
        leader_indices,
        gain=5.0,
        u_max=CRAZYFLIE_COMM['max_velocity'],
        saturation_type='smooth',
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


def build_base_phase_specs(setup):
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


def build_baseline_animation_scenario():
    setup = build_base_animation_setup()
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
        'phase_specs': build_base_phase_specs(setup),
        'extra': {
            'ctrl_norms': ctrl_norms,
            'max_ctrl': max_ctrl,
            'sat_ratio': sat_ratio,
            'u_max': controller.u_max,
        },
    }


def build_cbf_animation_scenario():
    setup = build_base_animation_setup()
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
        'phase_specs': build_base_phase_specs(setup),
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


def build_eso_animation_scenario():
    setup = build_base_animation_setup()
    n_f = setup['controller'].n_f
    w_const = np.array([0.2, 0.1, 0.05])
    ou_theta = 0.5
    ou_sigma = 0.1
    wind_seed = 123
    omega_o = 8.0

    times_bl, _, err_bl, _, _ = simulate_first_order_eso(
        setup['controller'], setup['init_pos'], setup['leader_traj'], (0, setup['T_total']), dt=0.02, wind=None, eso=None,
    )
    times_nd, _, err_nd, _, _ = simulate_first_order_eso(
        setup['controller'], setup['init_pos'], setup['leader_traj'], (0, setup['T_total']), dt=0.02,
        wind=WindDisturbance(n_f, dim=3, w_const=w_const, ou_theta=ou_theta, ou_sigma=ou_sigma, seed=wind_seed),
        eso=None,
    )
    times_wd, pos_hist, errors, _, eso_data = simulate_first_order_eso(
        setup['controller'], setup['init_pos'], setup['leader_traj'], (0, setup['T_total']), dt=0.02,
        wind=WindDisturbance(n_f, dim=3, w_const=w_const, ou_theta=ou_theta, ou_sigma=ou_sigma, seed=wind_seed),
        eso=ExtendedStateObserver(n_f, dim=3, omega_o=omega_o),
    )

    rep_idx = 0
    disturbance_true_norm = np.linalg.norm(eso_data['disturbances_true'][:, rep_idx], axis=1)
    disturbance_est_norm = np.linalg.norm(eso_data['disturbances_est'][:, rep_idx], axis=1)
    per_agent_est_error = np.linalg.norm(eso_data['disturbances_true'] - eso_data['disturbances_est'], axis=2)

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
        'phase_specs': build_base_phase_specs(setup),
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


def build_et_animation_scenario():
    setup = build_base_animation_setup()
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
        setup['controller'], setup['init_pos'], init_vel, setup['leader_traj'], (0, setup['T_total']), dt=0.02, et_manager=et_mgr,
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
        'phase_specs': build_base_phase_specs(setup),
        'extra': {
            'trigger_log': et_data['trigger_log'],
            'comm_rates': et_data['comm_rates'],
            'cumulative_counts': cumulative_counts,
            'total_triggers': total_triggers,
            'continuous_baseline': continuous_baseline,
        },
    }


def build_rhf_animation_scenario():
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
            'phase_colors': ['#1f77b4', '#ff7f0e', '#2ca02c'],
            'nominal_pos': nominal_pos,
        },
    }


def run_pyramid_integrated_mission(dt=0.02, archive=None, standalone_archive=False,
                                   config=None, render_outputs=True,
                                   verbose=True):
    """运行综合金字塔任务，可独立调用，也可嵌入完整主流程。"""
    base_config = load_pyramid_mission_config()
    if config:
        base_config.update(config)
    config = base_config

    if verbose:
        print("\n[Step 7.9] 综合场景：地面起飞并形成空中金字塔...")

    pyramid_nominal, _, _ = aerial_pyramid_10()
    pyramid_phase_defs = [
        {
            't_switch': 0.0,
            'leaders': [0, 1, 2, 9],
            'A': scale_matrix(3, [0.75, 0.75, 0.35]),
            'b': np.array([-1.2, 0.0, 0.15]),
            't_transition': 8.0,
            'label': 'Phase 0: 起飞展开',
        },
        {
            't_switch': 10.0,
            'leaders': [0, 2, 5, 9],
            'A': np.eye(3),
            'b': np.array([0.0, 0.0, 0.0]),
            't_transition': 7.0,
            'label': 'Phase 1: 金字塔成形',
        },
        {
            't_switch': 22.0,
            'leaders': [1, 3, 6, 9],
            'A': rotation_matrix_z(np.pi / 10) @ scale_matrix(3, [1.08, 1.08, 1.0]),
            'b': np.array([1.4, 0.6, 0.35]),
            't_transition': 8.0,
            'label': 'Phase 2: 空中巡航',
        },
    ]

    pyramid_schedule = []
    pyramid_phase_infos = []
    for phase_def in pyramid_phase_defs:
        span = check_affine_span(pyramid_nominal, phase_def['leaders'])
        omega_phase, phase_info = compute_power_centric_stress_matrix(
            pyramid_nominal, phase_def['leaders']
        )
        leader_targets = affine_transform(
            pyramid_nominal[phase_def['leaders']],
            A=phase_def['A'],
            b=phase_def['b'],
        )
        pyramid_schedule.append({
            't_switch': phase_def['t_switch'],
            'leader_indices': phase_def['leaders'],
            'leader_targets': leader_targets,
            't_transition': phase_def['t_transition'],
            'omega': omega_phase,
            'adj': phase_info['adj_matrix'],
            'label': phase_def['label'],
        })
        pyramid_phase_infos.append({
            'label': phase_def['label'],
            'leaders': phase_def['leaders'],
            'span_valid': span['valid'],
            'span_rank': span['rank'],
            'min_eig_ff': phase_info['min_eig_ff'],
            'n_edges': phase_info['n_edges'],
        })

    if verbose:
        for item in pyramid_phase_infos:
            print(f"  [{item['label']}] leaders={item['leaders']}, "
                  f"rank={item['span_rank']}, valid={item['span_valid']}, "
                  f"lambda_min={item['min_eig_ff']:.5f}, edges={item['n_edges']}")

    pyramid_controller = AFCController(
        pyramid_schedule[0]['omega'],
        pyramid_schedule[0]['leader_indices'],
        gain=config.get('gain', 5.5),
        u_max=CRAZYFLIE_COMM['max_velocity'],
        saturation_type='smooth',
    )
    pyramid_total_time = config.get('total_time', 34.0)
    pyramid_d_safe = config.get('d_safe', 0.16)
    pyramid_d_activate = config.get('d_activate', 0.9)
    pyramid_cbf_gamma = config.get('cbf_gamma', 6.0)
    ground_angles = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False)
    init_radius = config.get('init_radius', 1.9)
    pyramid_ground = np.column_stack([
        init_radius * np.cos(ground_angles),
        init_radius * np.sin(ground_angles),
        np.zeros(10),
    ])
    rng = np.random.default_rng(config.get('rng_seed', 2026))
    pyramid_init = pyramid_ground.copy()
    pyramid_init[:, :2] += rng.normal(
        scale=config.get('init_noise_std', 0.04), size=(10, 2)
    )

    wind_const = np.asarray(config.get('wind_const', [0.08, -0.03, 0.02]), dtype=float)
    ou_theta = config.get('ou_theta', 0.4)
    ou_sigma = config.get('ou_sigma', 0.04)
    wind_seed = config.get('wind_seed', 321)

    pyramid_wind = WindDisturbance(
        n_agents=10,
        dim=3,
        w_const=wind_const,
        ou_theta=ou_theta,
        ou_sigma=ou_sigma,
        seed=wind_seed,
    )
    eso_omega = config.get('eso_omega', 8.0)
    et_mu = config.get('et_mu', 0.015)
    et_varpi = config.get('et_varpi', 0.45)
    et_phi_0 = config.get('et_phi_0', 1.0)
    pyramid_times, pyramid_pos_hist, pyramid_errors, pyramid_ctrl, pyramid_data = (
        simulate_integrated_first_order_rhf(
            pyramid_controller,
            pyramid_init,
            pyramid_nominal,
            pyramid_schedule,
            (0.0, pyramid_total_time),
            dt=dt,
            wind=pyramid_wind,
            eso_omega=eso_omega,
            cbf_kwargs={
                'd_safe': pyramid_d_safe,
                'gamma': pyramid_cbf_gamma,
                'd_activate': pyramid_d_activate,
            },
            et_kwargs={
                'mu': et_mu,
                'varpi': et_varpi,
                'phi_0': et_phi_0,
            },
        )
    )

    pyramid_comm = pyramid_data['comm_rates']
    cbf_active_ratio = float(100.0 * np.mean(pyramid_data['n_active_constraints'] > 0))
    min_distance = float(pyramid_data['min_distances'].min())
    final_error = float(pyramid_errors[-1])
    final_est_error = float(pyramid_data['estimation_error'][-1])

    if verbose:
        print(f"  综合场景步数: {len(pyramid_times)}")
        print(f"  最终误差: {final_error:.6f}")
        print(f"  最小间距: {min_distance:.4f} m")
        print(f"  平均通信率: {pyramid_comm['mean']:.2f}%")
        print(f"  通信节省: {100.0 - pyramid_comm['mean']:.1f}%")
        print(f"  CBF 激活占比: {cbf_active_ratio:.1f}%")
        print(f"  ESO 最终估计误差: {final_est_error:.6f}")
        print(f"  RHF 切换次数: {pyramid_data['n_switches']}")

    if render_outputs:
        fig19 = plt.figure(figsize=(18, 5))
        snap_steps = [
            0,
            min(int(8.0 / dt), len(pyramid_times) - 1),
            min(int(17.0 / dt), len(pyramid_times) - 1),
            len(pyramid_times) - 1,
        ]
        snap_titles = [
            '(a) 地面待命',
            '(b) 起飞展开',
            '(c) 金字塔成形',
            '(d) 巡航保持',
        ]
        for subplot_idx, (step_idx, title) in enumerate(zip(snap_steps, snap_titles), start=1):
            ax = fig19.add_subplot(1, 4, subplot_idx, projection='3d')
            phase_idx = int(pyramid_data['phase_indices'][step_idx])
            plot_formation_3d(
                ax,
                pyramid_pos_hist[step_idx],
                pyramid_schedule[phase_idx]['leader_indices'],
                pyramid_schedule[phase_idx]['adj'],
                title=f'{title} t={pyramid_times[step_idx]:.1f}s',
            )
        fig19.suptitle('综合场景：从地面起飞到空中金字塔', fontsize=14, y=1.02)
        fig19.tight_layout()
        fig19.savefig(figure_path('fig19_pyramid_mission_snapshots.png'), dpi=200, bbox_inches='tight')
        if archive is not None:
            archive.save_figure(fig19, 'fig19_pyramid_mission_snapshots')
        if verbose:
            print("  已保存: fig19_pyramid_mission_snapshots.png")
        plt.close(fig19)

        fig20, axes20 = plt.subplots(2, 2, figsize=(16, 10))
        phase_switches = [item['t_switch'] for item in pyramid_schedule]

        axes20[0, 0].semilogy(pyramid_times, pyramid_errors + 1e-16, color='tab:blue', linewidth=1.6)
        for idx_phase, t_start in enumerate(phase_switches):
            t_end = pyramid_schedule[idx_phase + 1]['t_switch'] if idx_phase + 1 < len(pyramid_schedule) else pyramid_total_time
            axes20[0, 0].axvspan(t_start, t_end, alpha=0.08, color=f'C{idx_phase}')
            if idx_phase > 0:
                axes20[0, 0].axvline(t_start, color=f'C{idx_phase}', linestyle='--', linewidth=1.2)
        axes20[0, 0].set_xlabel('时间 (s)')
        axes20[0, 0].set_ylabel('编队误差')
        axes20[0, 0].set_title('(a) 综合场景误差收敛')
        axes20[0, 0].grid(True, alpha=0.3)

        axes20[0, 1].plot(pyramid_times, pyramid_data['min_distances'], color='tab:red', linewidth=1.5, label='最小间距')
        axes20[0, 1].axhline(pyramid_d_safe, color='black', linestyle='--', linewidth=1.2, label='安全阈值')
        ax20b = axes20[0, 1].twinx()
        ax20b.plot(pyramid_times, pyramid_data['n_active_constraints'], color='tab:orange', linewidth=1.0, alpha=0.8, label='活跃约束数')
        axes20[0, 1].set_xlabel('时间 (s)')
        axes20[0, 1].set_ylabel('距离 (m)')
        ax20b.set_ylabel('约束数')
        axes20[0, 1].set_title('(b) CBF 安全间距与约束激活')
        axes20[0, 1].grid(True, alpha=0.3)

        cumulative_triggers = np.cumsum([
            len([ev for ev in pyramid_data['trigger_log'] if abs(ev[0] - t) <= dt / 2])
            for t in pyramid_times
        ])
        continuous_baseline = np.arange(len(pyramid_times)) * 6
        axes20[1, 0].plot(pyramid_times, cumulative_triggers, color='tab:green', linewidth=1.5, label='事件触发累计')
        axes20[1, 0].plot(pyramid_times, continuous_baseline, 'k--', linewidth=1.2, alpha=0.5, label='连续通信基线')
        axes20[1, 0].set_xlabel('时间 (s)')
        axes20[1, 0].set_ylabel('累计通信次数')
        axes20[1, 0].set_title('(c) ET 通信节省')
        axes20[1, 0].legend(fontsize=8)
        axes20[1, 0].grid(True, alpha=0.3)

        axes20[1, 1].plot(pyramid_times, pyramid_data['estimation_error'], color='tab:purple', linewidth=1.5, label='||w-z_hat||')
        axes20[1, 1].plot(pyramid_times, pyramid_data['cbf_modifications'], color='tab:cyan', linewidth=1.2, label='||u_safe-u_nom||')
        axes20[1, 1].set_xlabel('时间 (s)')
        axes20[1, 1].set_ylabel('幅值')
        axes20[1, 1].set_title('(d) ESO 估计与安全修正')
        axes20[1, 1].legend(fontsize=8)
        axes20[1, 1].grid(True, alpha=0.3)

        fig20.suptitle('综合场景模块协同指标', fontsize=14)
        fig20.tight_layout()
        fig20.savefig(figure_path('fig20_pyramid_mission_metrics.png'), dpi=200, bbox_inches='tight')
        if archive is not None:
            archive.save_figure(fig20, 'fig20_pyramid_mission_metrics')
        if verbose:
            print("  已保存: fig20_pyramid_mission_metrics.png")
        plt.close(fig20)

        fig21 = plt.figure(figsize=(18, 5))
        for subplot_idx, sched in enumerate(pyramid_schedule, start=1):
            ax = fig21.add_subplot(1, len(pyramid_schedule), subplot_idx, projection='3d')
            plot_formation_3d(
                ax,
                pyramid_nominal,
                sched['leader_indices'],
                sched['adj'],
                title=sched['label'],
            )
        fig21.suptitle('综合场景 RHF 阶段拓扑', fontsize=14, y=1.02)
        fig21.tight_layout()
        fig21.savefig(figure_path('fig21_pyramid_rhf_topology.png'), dpi=200, bbox_inches='tight')
        if archive is not None:
            archive.save_figure(fig21, 'fig21_pyramid_rhf_topology')
        if verbose:
            print("  已保存: fig21_pyramid_rhf_topology.png")
        plt.close(fig21)

    mission_results = {
        'times': pyramid_times,
        'positions': pyramid_pos_hist,
        'errors': pyramid_errors,
        'control_inputs': pyramid_ctrl,
        'data': pyramid_data,
        'nominal': pyramid_nominal,
        'schedule': pyramid_schedule,
        'phase_info': pyramid_phase_infos,
        'total_time': pyramid_total_time,
        'd_safe': pyramid_d_safe,
        'd_activate': pyramid_d_activate,
        'cbf_gamma': pyramid_cbf_gamma,
        'comm': pyramid_comm,
        'summary': {
            'final_error': final_error,
            'min_distance': min_distance,
            'mean_comm_rate_pct': pyramid_comm['mean'],
            'comm_saving_pct': 100.0 - pyramid_comm['mean'],
            'cbf_active_ratio_pct': cbf_active_ratio,
            'final_estimation_error': final_est_error,
            'n_switches': pyramid_data['n_switches'],
        },
        'config': {
            'gain': pyramid_controller.gain,
            'd_safe': pyramid_d_safe,
            'd_activate': pyramid_d_activate,
            'cbf_gamma': pyramid_cbf_gamma,
            'total_time': pyramid_total_time,
            'init_radius': init_radius,
            'init_noise_std': config.get('init_noise_std', 0.04),
            'rng_seed': config.get('rng_seed', 2026),
            'wind_const': wind_const.tolist(),
            'ou_theta': ou_theta,
            'ou_sigma': ou_sigma,
            'wind_seed': wind_seed,
            'eso_omega': eso_omega,
            'et_mu': et_mu,
            'et_varpi': et_varpi,
            'et_phi_0': et_phi_0,
        },
    }

    if standalone_archive:
        standalone_arch = archive if archive is not None else SimArchive(tag='pyramid_mission')
        standalone_arch.save_arrays(
            pyramid_times=pyramid_times,
            pyramid_positions=pyramid_pos_hist,
            pyramid_errors=pyramid_errors,
            pyramid_control_inputs=pyramid_ctrl,
            pyramid_min_distances=pyramid_data['min_distances'],
            pyramid_cbf_modifications=pyramid_data['cbf_modifications'],
            pyramid_active_constraints=pyramid_data['n_active_constraints'],
            pyramid_disturbances_true=pyramid_data['disturbances_true'],
            pyramid_disturbances_est=pyramid_data['disturbances_est'],
            pyramid_phi_history=pyramid_data['phi_history'],
        )
        standalone_arch.save_params({
            'pyramid_integrated_mission': {
                'formation': 'aerial_pyramid_10',
                'total_time': pyramid_total_time,
                'dt': dt,
                'phase_info': pyramid_phase_infos,
                'initial_leaders': pyramid_schedule[0]['leader_indices'],
                'd_safe': pyramid_d_safe,
                'd_activate': pyramid_d_activate,
                'cbf_gamma': pyramid_cbf_gamma,
                'final_error': final_error,
                'min_distance': min_distance,
                'mean_comm_rate_pct': pyramid_comm['mean'],
                'comm_saving_pct': 100.0 - pyramid_comm['mean'],
                'total_triggers': pyramid_comm['total_triggers'],
                'total_possible': pyramid_comm['total_possible'],
                'cbf_active_ratio_pct': cbf_active_ratio,
                'final_estimation_error': final_est_error,
                'n_switches': pyramid_data['n_switches'],
                'config': mission_results['config'],
            },
        })
        standalone_arch.finalize()

    return mission_results


# ============================================================
# 主仿真流程
# ============================================================

def main(selected_scenario='all'):
    if selected_scenario == 'pyramid':
        print("=" * 60)
        print("仿射编队控制 (AFC) 仿真 - 综合金字塔任务单独运行")
        print("=" * 60)
        run_pyramid_integrated_mission(dt=0.02, archive=SimArchive(tag='pyramid_mission'), standalone_archive=True)
        print("\n仿真完成！")
        return

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

    # 初始化存档系统
    archive = SimArchive(tag='afc')

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
    fig1.savefig(figure_path('fig1_formation_snapshots.png'),
                 dpi=200, bbox_inches='tight')
    archive.save_figure(fig1, 'fig1_formation_snapshots')
    print("  已保存: fig1_formation_snapshots.png")

    # ==== 图2: 3D 轨迹 ====
    fig2 = plt.figure(figsize=(10, 8))
    ax_traj = fig2.add_subplot(111, projection='3d')
    plot_trajectories_3d(ax_traj, times, pos_hist, leader_indices)
    ax_traj.set_title('智能体 3D 轨迹', fontsize=13)
    ax_traj.legend(['Leader', 'Follower'], loc='upper left')
    fig2.tight_layout()
    fig2.savefig(figure_path('fig2_trajectories_3d.png'),
                 dpi=200, bbox_inches='tight')
    archive.save_figure(fig2, 'fig2_trajectories_3d')
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
    fig3.savefig(figure_path('fig3_convergence.png'),
                 dpi=200, bbox_inches='tight')
    archive.save_figure(fig3, 'fig3_convergence')
    print("  已保存: fig3_convergence.png")

    # ==== 图4: 应力矩阵热力图 ====
    fig4, axes4 = plt.subplots(1, 3, figsize=(15, 4))

    assert Omega is not None
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
    fig4.savefig(figure_path('fig4_stress_matrix.png'),
                 dpi=200, bbox_inches='tight')
    archive.save_figure(fig4, 'fig4_stress_matrix')
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
    fig5.savefig(figure_path('fig5_communication_graph.png'),
                 dpi=200, bbox_inches='tight')
    archive.save_figure(fig5, 'fig5_communication_graph')
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
    fig6.savefig(figure_path('fig6_sparse_comparison.png'),
                 dpi=200, bbox_inches='tight')
    archive.save_figure(fig6, 'fig6_sparse_comparison')
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
    fig7.savefig(figure_path('fig7_saturation_analysis.png'),
                 dpi=200, bbox_inches='tight')
    archive.save_figure(fig7, 'fig7_saturation_analysis')
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
    fig8.savefig(figure_path('fig8_saturation_comparison.png'),
                 dpi=200, bbox_inches='tight')
    archive.save_figure(fig8, 'fig8_saturation_comparison')
    print("  已保存: fig8_saturation_comparison.png")

    # ----------------------------------------------------------
    # Step 7.5: CBF 碰撞避免安全滤波验证
    # ----------------------------------------------------------
    print("\n[Step 7.5] CBF 碰撞避免安全滤波验证...")

    d_safe = CRAZYFLIE_SAFETY['safety_distance_m']
    gamma_cbf = CRAZYFLIE_SAFETY['cbf_gamma']
    d_activate = CRAZYFLIE_SAFETY['activate_distance_m']

    cbf = CBFSafetyFilter(
        n_agents=10, leader_indices=leader_indices,
        d_safe=d_safe, gamma=gamma_cbf, d_activate=d_activate,
    )

    print(f"  安全距离 d_s = {d_safe} m (Crazyflie 2.1)")
    print(f"  CBF 衰减率 γ = {gamma_cbf}")
    print(f"  激活距离 d_a = {d_activate} m")

    # 碰撞风险场景：大初始扰动（模拟密集起飞 / 初始定位误差大）
    cbf_noise_std = 1.5
    np.random.seed(99)
    init_pos_cbf = nominal_pos.copy()
    init_pos_cbf[follower_indices] += (
        np.random.randn(len(follower_indices), d) * cbf_noise_std
    )

    init_min_d, init_min_pair = CBFSafetyFilter.min_distance(init_pos_cbf)
    print(f"  碰撞风险场景: 初始扰动 σ = {cbf_noise_std} m")
    print(f"  初始最小距离: {init_min_d:.4f} m "
          f"(Agent {init_min_pair[0]}-{init_min_pair[1]})")

    # 无 CBF 仿真（Euler 积分，与 CBF 仿真相同方法以公平对比）
    print("  运行无 CBF 仿真 (Euler)...")
    (times_nocbf, pos_nocbf, err_nocbf, ctrl_nocbf,
     cbf_data_nocbf) = simulate_first_order_cbf(
        controller, init_pos_cbf, leader_traj, (0, T_total), dt=dt,
        cbf_filter=None,
    )
    print(f"    最终误差: {err_nocbf[-1]:.6f}")

    # 有 CBF 仿真
    print("  运行 CBF 安全滤波仿真...")
    (times_wcbf, pos_wcbf, err_wcbf, ctrl_wcbf,
     cbf_data_wcbf) = simulate_first_order_cbf(
        controller, init_pos_cbf, leader_traj, (0, T_total), dt=dt,
        cbf_filter=cbf,
    )
    print(f"    最终误差: {err_wcbf[-1]:.6f}")

    min_d_nocbf = cbf_data_nocbf['min_distances']
    min_d_wcbf = cbf_data_wcbf['min_distances']

    print(f"\n  === CBF 碰撞避免效果 ===")
    print(f"  无 CBF 最小距离: {min_d_nocbf.min():.4f} m "
          f"({'安全' if min_d_nocbf.min() > d_safe else '⚠ 低于安全阈值!'})")
    print(f"  有 CBF 最小距离: {min_d_wcbf.min():.4f} m "
          f"({'安全 ✓' if min_d_wcbf.min() > d_safe - 0.01 else '⚠ 需调参'})")
    n_cbf_active = int(np.sum(cbf_data_wcbf['n_active'] > 0))
    print(f"  CBF 约束激活步数: {n_cbf_active}/{len(times_wcbf)} "
          f"({100 * n_cbf_active / len(times_wcbf):.1f}%)")
    print(f"  最大修正幅度: {cbf_data_wcbf['modifications'].max():.4f} m/s")

    # ==== 图9: 最小智能体间距离对比 ====
    fig9, axes9 = plt.subplots(2, 1, figsize=(14, 8))

    # (a) 最小距离时间历程
    axes9[0].plot(times_nocbf, min_d_nocbf, 'b-', linewidth=1.2,
                  alpha=0.8, label='无 CBF')
    axes9[0].plot(times_wcbf, min_d_wcbf, 'r-', linewidth=1.5,
                  label='有 CBF 安全滤波')
    axes9[0].axhline(y=d_safe, color='darkred', linestyle='--', linewidth=2,
                     label=f'安全距离 d_s = {d_safe} m')
    axes9[0].fill_between(times_nocbf, 0, d_safe, alpha=0.08, color='red')
    # 标记违反区域
    violation_mask = min_d_nocbf < d_safe
    if np.any(violation_mask):
        axes9[0].fill_between(times_nocbf, min_d_nocbf, d_safe,
                              where=violation_mask, alpha=0.3, color='red',
                              label='安全违反区域')
    axes9[0].set_xlabel('时间 (s)')
    axes9[0].set_ylabel('最小智能体间距离 (m)')
    axes9[0].set_title('(a) 最小智能体间距离对比', fontsize=11)
    axes9[0].legend(loc='lower right')
    axes9[0].grid(True, alpha=0.3)
    axes9[0].set_xlim([0, T_total])
    axes9[0].set_ylim(bottom=0)

    # (b) 编队误差对比
    axes9[1].semilogy(times_nocbf, err_nocbf + 1e-16, 'b-', linewidth=1.2,
                      alpha=0.8, label='无 CBF')
    axes9[1].semilogy(times_wcbf, err_wcbf + 1e-16, 'r-', linewidth=1.5,
                      label='有 CBF 安全滤波')
    axes9[1].set_xlabel('时间 (s)')
    axes9[1].set_ylabel('编队误差 ||p_f - p_f*||')
    axes9[1].set_title('(b) 编队误差收敛对比 (碰撞风险场景)', fontsize=11)
    axes9[1].legend()
    axes9[1].grid(True, alpha=0.3)
    axes9[1].set_xlim([0, T_total])
    for i in range(len(phase_labels)):
        if i < len(phase_times) - 1:
            axes9[1].axvspan(phase_times[i], phase_times[i + 1],
                             alpha=0.1, color=colors[i % len(colors)])

    fig9.suptitle(f'CBF 碰撞避免效果验证 (d_s={d_safe}m, γ={gamma_cbf}, '
                  f'初始扰动σ={cbf_noise_std}m)', fontsize=13)
    fig9.tight_layout()
    fig9.savefig(figure_path('fig9_cbf_collision_avoidance.png'),
                 dpi=200, bbox_inches='tight')
    archive.save_figure(fig9, 'fig9_cbf_collision_avoidance')
    print("  已保存: fig9_cbf_collision_avoidance.png")

    # ==== 图10: CBF 安全滤波分析面板 ====
    fig10, axes10 = plt.subplots(2, 2, figsize=(16, 10))

    # (a) CBF 约束激活数量
    axes10[0, 0].fill_between(times_wcbf, cbf_data_wcbf['n_active'],
                               alpha=0.4, color='coral')
    axes10[0, 0].plot(times_wcbf, cbf_data_wcbf['n_active'],
                       'r-', linewidth=0.8)
    axes10[0, 0].set_xlabel('时间 (s)')
    axes10[0, 0].set_ylabel('活跃约束数')
    axes10[0, 0].set_title('(a) CBF 约束激活数量', fontsize=11)
    axes10[0, 0].grid(True, alpha=0.3)
    axes10[0, 0].set_xlim([0, T_total])

    # (b) 控制修正幅度
    axes10[0, 1].plot(times_wcbf, cbf_data_wcbf['modifications'],
                       'r-', linewidth=0.8)
    axes10[0, 1].fill_between(times_wcbf, cbf_data_wcbf['modifications'],
                               alpha=0.3, color='coral')
    axes10[0, 1].set_xlabel('时间 (s)')
    axes10[0, 1].set_ylabel('||u_safe - u_nom|| (m/s)')
    axes10[0, 1].set_title('(b) CBF 安全滤波修正幅度', fontsize=11)
    axes10[0, 1].grid(True, alpha=0.3)
    axes10[0, 1].set_xlim([0, T_total])

    # (c) 控制输入范数对比
    ctrl_norms_nocbf = np.linalg.norm(ctrl_nocbf, axis=2).max(axis=1)
    ctrl_norms_wcbf = np.linalg.norm(ctrl_wcbf, axis=2).max(axis=1)
    axes10[1, 0].plot(times_nocbf, ctrl_norms_nocbf, 'b-', linewidth=1.0,
                       alpha=0.7, label='无 CBF')
    axes10[1, 0].plot(times_wcbf, ctrl_norms_wcbf, 'r-', linewidth=1.0,
                       alpha=0.7, label='有 CBF')
    axes10[1, 0].axhline(y=u_max, color='gray', linestyle='--',
                          linewidth=1.5, alpha=0.5, label=f'u_max={u_max}')
    axes10[1, 0].set_xlabel('时间 (s)')
    axes10[1, 0].set_ylabel('max ||u_i|| (m/s)')
    axes10[1, 0].set_title('(c) 最大控制输入对比', fontsize=11)
    axes10[1, 0].legend(fontsize=9)
    axes10[1, 0].grid(True, alpha=0.3)
    axes10[1, 0].set_xlim([0, T_total])

    # (d) 所有成对跟随者距离（有 CBF）
    n_pairs = 0
    for i_loc in range(len(follower_indices)):
        fi = follower_indices[i_loc]
        for j_loc in range(i_loc + 1, len(follower_indices)):
            fj = follower_indices[j_loc]
            dists = np.linalg.norm(
                pos_wcbf[:, fi] - pos_wcbf[:, fj], axis=1)
            axes10[1, 1].plot(times_wcbf, dists, linewidth=0.6, alpha=0.5)
            n_pairs += 1
    axes10[1, 1].axhline(y=d_safe, color='darkred', linestyle='--',
                          linewidth=2, label=f'd_s = {d_safe} m')
    axes10[1, 1].set_xlabel('时间 (s)')
    axes10[1, 1].set_ylabel('成对距离 (m)')
    axes10[1, 1].set_title(f'(d) 跟随者成对距离 ({n_pairs} 对, 有CBF)',
                            fontsize=11)
    axes10[1, 1].legend(loc='lower right')
    axes10[1, 1].grid(True, alpha=0.3)
    axes10[1, 1].set_xlim([0, T_total])
    axes10[1, 1].set_ylim(bottom=0)

    fig10.suptitle('CBF 安全滤波器分析', fontsize=14)
    fig10.tight_layout()
    fig10.savefig(figure_path('fig10_cbf_analysis.png'),
                  dpi=200, bbox_inches='tight')
    archive.save_figure(fig10, 'fig10_cbf_analysis')
    print("  已保存: fig10_cbf_analysis.png")

    # ----------------------------------------------------------
    # Step 7.6: ESO 鲁棒抗扰验证
    # ----------------------------------------------------------
    print("\n[Step 7.6] ESO 鲁棒抗扰验证...")

    # 风场扰动参数
    w_const_vec = np.array([0.2, 0.1, 0.05])   # 恒定侧风 (m/s)
    ou_theta = 0.5      # OU 均值回复速率 (1/s)，τ_w = 2s
    ou_sigma = 0.1      # OU 波动强度 (m/s)
    wind_seed = 123

    # ESO 参数
    omega_o = 8.0       # 观测器带宽 (rad/s)

    print(f"  恒定风场: {w_const_vec} m/s (||w||={np.linalg.norm(w_const_vec):.3f})")
    print(f"  阵风模型: OU 过程 (θ={ou_theta}, σ={ou_sigma})")
    print(f"  ESO 带宽: w0 = {omega_o} rad/s")
    print(f"  ESO 增益: beta1 = {2*omega_o:.1f}, beta2 = {omega_o**2:.1f}")

    # --- 场景1: 无扰动基线 (使用标准初始位置) ---
    print("  运行场景1: 无扰动基线...")
    wind_none = WindDisturbance(controller.n_f, dim=d, seed=wind_seed)
    (times_eso_bl, pos_eso_bl, err_eso_bl,
     ctrl_eso_bl, eso_data_bl) = simulate_first_order_eso(
        controller, init_pos, leader_traj, (0, T_total), dt=dt,
        wind=None, eso=None,
    )
    print(f"    最终误差: {err_eso_bl[-1]:.6f}")

    # --- 场景2: 有扰动、无 ESO ---
    print("  运行场景2: 有扰动、无 ESO...")
    wind_noeso = WindDisturbance(
        controller.n_f, dim=d, w_const=w_const_vec,
        ou_theta=ou_theta, ou_sigma=ou_sigma, seed=wind_seed)
    (times_eso_nd, pos_eso_nd, err_eso_nd,
     ctrl_eso_nd, eso_data_nd) = simulate_first_order_eso(
        controller, init_pos, leader_traj, (0, T_total), dt=dt,
        wind=wind_noeso, eso=None,
    )
    print(f"    最终误差: {err_eso_nd[-1]:.6f}")

    # --- 场景3: 有扰动、有 ESO ---
    print("  运行场景3: 有扰动、有 ESO...")
    wind_weso = WindDisturbance(
        controller.n_f, dim=d, w_const=w_const_vec,
        ou_theta=ou_theta, ou_sigma=ou_sigma, seed=wind_seed)
    eso = ExtendedStateObserver(controller.n_f, dim=d, omega_o=omega_o)
    (times_eso_wd, pos_eso_wd, err_eso_wd,
     ctrl_eso_wd, eso_data_wd) = simulate_first_order_eso(
        controller, init_pos, leader_traj, (0, T_total), dt=dt,
        wind=wind_weso, eso=eso,
    )
    print(f"    最终误差: {err_eso_wd[-1]:.6f}")

    # 稳态误差分析
    # 理论稳态误差 (无ESO): δ_f = Ω_ff⁻¹ w_const / K_p
    Omega_ff_inv = np.linalg.inv(controller.Omega_ff)
    w_const_all = np.tile(w_const_vec, (controller.n_f, 1))
    delta_theory = Omega_ff_inv @ w_const_all / controller.gain
    delta_norm = np.linalg.norm(delta_theory)
    print(f"\n  === ESO 抗扰效果 ===")
    print(f"  理论稳态偏差 (无ESO, 恒定风): {delta_norm:.4f} m")
    print(f"  实际稳态误差 (无扰动): {err_eso_bl[-1]:.6f}")
    print(f"  实际稳态误差 (有扰动无ESO): {err_eso_nd[-1]:.6f}")
    print(f"  实际稳态误差 (有扰动有ESO): {err_eso_wd[-1]:.6f}")
    if err_eso_nd[-1] > 1e-6:
        reduction = (1 - err_eso_wd[-1] / err_eso_nd[-1]) * 100
        print(f"  误差降低: {reduction:.1f}%")

    # ==== 图11: ESO 扰动估计精度 ====
    fig11, axes11 = plt.subplots(2, 2, figsize=(16, 10))

    # 选择一个代表性跟随者 (agent 1, 即 follower index 0)
    fi_show = 0
    fi_global = follower_indices[fi_show]
    axis_labels = ['X', 'Y', 'Z']
    axis_colors = ['#1f77b4', '#2ca02c', '#d62728']

    # (a) 扰动真实值 vs ESO 估计（三个分量）
    for k in range(d):
        axes11[0, 0].plot(times_eso_wd,
                          eso_data_wd['disturbances_true'][:, fi_show, k],
                          color=axis_colors[k], linewidth=1.2, alpha=0.7,
                          label=f'真实 w_{axis_labels[k]}')
        axes11[0, 0].plot(times_eso_wd,
                          eso_data_wd['disturbances_est'][:, fi_show, k],
                          color=axis_colors[k], linewidth=1.2,
                          linestyle='--', alpha=0.9,
                          label=f'ESO z2_{axis_labels[k]}')
    axes11[0, 0].set_xlabel('时间 (s)')
    axes11[0, 0].set_ylabel('扰动 (m/s)')
    axes11[0, 0].set_title(f'(a) Agent {fi_global} 扰动估计', fontsize=11)
    axes11[0, 0].legend(fontsize=7, ncol=2)
    axes11[0, 0].grid(True, alpha=0.3)
    axes11[0, 0].set_xlim([0, T_total])

    # (b) 所有跟随者的扰动估计误差范数
    for i in range(controller.n_f):
        fi_g = follower_indices[i]
        est_err_i = np.linalg.norm(
            eso_data_wd['disturbances_true'][:, i]
            - eso_data_wd['disturbances_est'][:, i], axis=1)
        axes11[0, 1].plot(times_eso_wd, est_err_i, linewidth=0.8,
                          alpha=0.7, label=f'Agent {fi_g}')
    axes11[0, 1].set_xlabel('时间 (s)')
    axes11[0, 1].set_ylabel('||w - z2|| (m/s)')
    axes11[0, 1].set_title('(b) 扰动估计误差', fontsize=11)
    axes11[0, 1].legend(fontsize=8)
    axes11[0, 1].grid(True, alpha=0.3)
    axes11[0, 1].set_xlim([0, T_total])

    # (c) 编队误差三场景对比
    axes11[1, 0].semilogy(times_eso_bl, err_eso_bl + 1e-16, 'b-',
                           linewidth=1.5, label='无扰动')
    axes11[1, 0].semilogy(times_eso_nd, err_eso_nd + 1e-16, 'r-',
                           linewidth=1.5, alpha=0.8,
                           label='有扰动 无ESO')
    axes11[1, 0].semilogy(times_eso_wd, err_eso_wd + 1e-16, 'g-',
                           linewidth=1.5, label='有扰动 有ESO')
    axes11[1, 0].set_xlabel('时间 (s)')
    axes11[1, 0].set_ylabel('编队误差 ||p_f - p_f*||')
    axes11[1, 0].set_title('(c) 编队误差收敛对比', fontsize=11)
    axes11[1, 0].legend()
    axes11[1, 0].grid(True, alpha=0.3)
    axes11[1, 0].set_xlim([0, T_total])
    for i in range(len(phase_labels)):
        if i < len(phase_times) - 1:
            axes11[1, 0].axvspan(phase_times[i], phase_times[i + 1],
                                 alpha=0.1, color=colors[i % len(colors)])

    # (d) 控制输入范数对比（有扰动: 无ESO vs 有ESO）
    ctrl_norm_nd = np.linalg.norm(ctrl_eso_nd, axis=2).max(axis=1)
    ctrl_norm_wd = np.linalg.norm(ctrl_eso_wd, axis=2).max(axis=1)
    axes11[1, 1].plot(times_eso_nd, ctrl_norm_nd, 'r-', linewidth=1.0,
                       alpha=0.7, label='无 ESO')
    axes11[1, 1].plot(times_eso_wd, ctrl_norm_wd, 'g-', linewidth=1.0,
                       alpha=0.7, label='有 ESO')
    axes11[1, 1].axhline(y=u_max, color='gray', linestyle='--',
                          linewidth=1.5, alpha=0.5, label=f'u_max={u_max}')
    axes11[1, 1].set_xlabel('时间 (s)')
    axes11[1, 1].set_ylabel('max ||u_i|| (m/s)')
    axes11[1, 1].set_title('(d) 最大控制输入对比', fontsize=11)
    axes11[1, 1].legend()
    axes11[1, 1].grid(True, alpha=0.3)
    axes11[1, 1].set_xlim([0, T_total])

    fig11.suptitle(f'ESO 扰动估计与补偿效果 '
                   f'(w0={omega_o}, w_const={w_const_vec}, '
                   f'OU: theta={ou_theta}, sigma={ou_sigma})', fontsize=13)
    fig11.tight_layout()
    fig11.savefig(figure_path('fig11_eso_disturbance_rejection.png'),
                  dpi=200, bbox_inches='tight')
    archive.save_figure(fig11, 'fig11_eso_disturbance_rejection')
    print("  已保存: fig11_eso_disturbance_rejection.png")

    # ==== 图12: 不同 ESO 带宽对比 ====
    omega_values = [2.0, 5.0, 8.0, 15.0]
    fig12, axes12 = plt.subplots(1, 2, figsize=(14, 5))

    for wo in omega_values:
        wind_cmp = WindDisturbance(
            controller.n_f, dim=d, w_const=w_const_vec,
            ou_theta=ou_theta, ou_sigma=ou_sigma, seed=wind_seed)
        eso_cmp = ExtendedStateObserver(controller.n_f, dim=d, omega_o=wo)
        (t_cmp, _, err_cmp, ctrl_cmp,
         eso_data_cmp) = simulate_first_order_eso(
            controller, init_pos, leader_traj, (0, T_total), dt=dt,
            wind=wind_cmp, eso=eso_cmp,
        )
        axes12[0].semilogy(t_cmp, err_cmp + 1e-16, linewidth=1.5,
                            label=f'w0={wo}')
        # 稳态扰动估计误差（取最后 20% 时间平均）
        n_tail = max(1, int(len(t_cmp) * 0.2))
        mean_est_err = eso_data_cmp['estimation_errors'][-n_tail:].mean()
        axes12[1].bar(f'w0={wo}', mean_est_err, alpha=0.7)

    # 无 ESO 基线
    axes12[0].semilogy(times_eso_nd, err_eso_nd + 1e-16, 'k--',
                        linewidth=1.5, alpha=0.5, label='无 ESO')
    axes12[0].set_xlabel('时间 (s)')
    axes12[0].set_ylabel('编队误差 ||p_f - p_f*||')
    axes12[0].set_title('(a) 不同 ESO 带宽下的误差收敛', fontsize=11)
    axes12[0].legend()
    axes12[0].grid(True, alpha=0.3)
    axes12[0].set_xlim([0, T_total])

    axes12[1].set_ylabel('稳态估计误差 ||w - z2||')
    axes12[1].set_title('(b) 稳态扰动估计精度', fontsize=11)
    axes12[1].grid(True, alpha=0.3, axis='y')

    fig12.suptitle('ESO 带宽参数 w0 对抗扰性能的影响', fontsize=14)
    fig12.tight_layout()
    fig12.savefig(figure_path('fig12_eso_bandwidth_comparison.png'),
                  dpi=200, bbox_inches='tight')
    archive.save_figure(fig12, 'fig12_eso_bandwidth_comparison')
    print("  已保存: fig12_eso_bandwidth_comparison.png")

    # ----------------------------------------------------------
    # Step 7.7: 自适应事件触发通信验证（二阶积分器）
    # ----------------------------------------------------------
    print("\n[Step 7.7] 自适应事件触发通信验证 (二阶积分器)...")

    # ET 参数
    et_mu = 0.01        # 指数衰减项系数
    et_varpi = 0.5      # 指数衰减速率
    et_phi_0 = 1.0      # 自适应参数初始值
    K_d = controller.damping

    print(f"  动力学模型: 二阶积分器 p̈_f = u_f")
    print(f"  增益: K_p = {gain}, K_d = {K_d}")
    print(f"  ET 参数: mu = {et_mu}, varpi = {et_varpi}, phi0 = {et_phi_0}")

    # 初始速度为零
    init_vel = np.zeros_like(init_pos)

    # --- 基线: 二阶积分器 + 连续通信 (Euler 积分) ---
    print("  运行二阶基线 (连续通信, Euler)...")
    (times_2nd_cont, pos_2nd_cont, err_2nd_cont,
     ctrl_2nd_cont, et_data_cont) = simulate_second_order_et(
        controller, init_pos, init_vel, leader_traj, (0, T_total), dt=dt,
        et_manager=None,
    )
    print(f"    最终误差: {err_2nd_cont[-1]:.6f}")

    # --- 自适应事件触发 ---
    print("  运行二阶自适应事件触发...")
    et_mgr = EventTriggerManager(
        n_agents=10, d=d,
        follower_indices=follower_indices,
        leader_indices=leader_indices,
        Omega=Omega,
        mu=et_mu, varpi=et_varpi, phi_0=et_phi_0,
    )
    (times_2nd_et, pos_2nd_et, err_2nd_et,
     ctrl_2nd_et, et_data_et) = simulate_second_order_et(
        controller, init_pos, init_vel, leader_traj, (0, T_total), dt=dt,
        et_manager=et_mgr,
    )
    print(f"    最终误差: {err_2nd_et[-1]:.6f}")

    # 通信率统计
    comm = et_data_et['comm_rates']
    print(f"\n  === 事件触发通信效果 ===")
    print(f"  平均通信率: {comm['mean']:.2f}%")
    print(f"  总触发次数: {comm['total_triggers']} / {comm['total_possible']} "
          f"(连续通信需 {comm['total_possible']})")
    print(f"  各 Follower 通信率:")
    for i_loc, fi in enumerate(follower_indices):
        print(f"    Agent {fi}: {comm['per_agent'][i_loc]:.2f}% "
              f"({et_data_et['trigger_counts'][fi]} 次)")
    print(f"  通信节省: {100 - comm['mean']:.1f}%")

    # ==== 图13: 事件触发 vs 连续通信误差对比 ====
    fig13, axes13 = plt.subplots(2, 1, figsize=(14, 8))

    # (a) 编队误差对比
    axes13[0].semilogy(times_2nd_cont, err_2nd_cont + 1e-16, 'b-',
                       linewidth=1.5, label='连续通信 (100%)')
    axes13[0].semilogy(times_2nd_et, err_2nd_et + 1e-16, 'r-',
                       linewidth=1.5,
                       label=f'事件触发 ({comm["mean"]:.1f}%)')
    axes13[0].set_xlabel('时间 (s)')
    axes13[0].set_ylabel('编队误差 ||p_f - p_f*||')
    axes13[0].set_title('(a) 二阶积分器编队误差收敛对比', fontsize=11)
    axes13[0].legend()
    axes13[0].grid(True, alpha=0.3)
    axes13[0].set_xlim([0, T_total])
    for i in range(len(phase_labels)):
        if i < len(phase_times) - 1:
            axes13[0].axvspan(phase_times[i], phase_times[i + 1],
                              alpha=0.1, color=colors[i % len(colors)])

    # (b) 控制输入范数对比
    ctrl_norm_cont = np.linalg.norm(ctrl_2nd_cont, axis=2).max(axis=1)
    ctrl_norm_et = np.linalg.norm(ctrl_2nd_et, axis=2).max(axis=1)
    axes13[1].plot(times_2nd_cont, ctrl_norm_cont, 'b-', linewidth=1.0,
                   alpha=0.7, label='连续通信')
    axes13[1].plot(times_2nd_et, ctrl_norm_et, 'r-', linewidth=1.0,
                   alpha=0.7, label='事件触发')
    axes13[1].axhline(y=u_max, color='gray', linestyle='--', linewidth=1.5,
                      alpha=0.5, label=f'u_max={u_max}')
    axes13[1].set_xlabel('时间 (s)')
    axes13[1].set_ylabel('max ||u_i|| (m/s²)')
    axes13[1].set_title('(b) 最大控制输入对比', fontsize=11)
    axes13[1].legend()
    axes13[1].grid(True, alpha=0.3)
    axes13[1].set_xlim([0, T_total])

    fig13.suptitle('自适应事件触发通信 vs 连续通信 (二阶积分器)', fontsize=14)
    fig13.tight_layout()
    fig13.savefig(figure_path('fig13_et_error_comparison.png'),
                  dpi=200, bbox_inches='tight')
    archive.save_figure(fig13, 'fig13_et_error_comparison')
    print("  已保存: fig13_et_error_comparison.png")

    # ==== 图14: 事件触发时间线与通信分析 ====
    fig14, axes14 = plt.subplots(2, 2, figsize=(16, 10))

    # (a) 触发事件时间线
    for i_loc, fi in enumerate(follower_indices):
        t_events = [ev[0] for ev in et_data_et['trigger_log'] if ev[1] == fi]
        if t_events:
            axes14[0, 0].eventplot([t_events], lineoffsets=fi,
                                    linelengths=0.6, colors=[f'C{i_loc}'])
    axes14[0, 0].set_xlabel('时间 (s)')
    axes14[0, 0].set_ylabel('Agent ID')
    axes14[0, 0].set_title('(a) 各 Follower 触发事件时间线', fontsize=11)
    axes14[0, 0].set_yticks(follower_indices)
    axes14[0, 0].grid(True, alpha=0.3, axis='x')
    axes14[0, 0].set_xlim([0, T_total])
    for i in range(len(phase_labels)):
        if i < len(phase_times) - 1:
            axes14[0, 0].axvspan(phase_times[i], phase_times[i + 1],
                                  alpha=0.08, color=colors[i % len(colors)])

    # (b) 累积通信次数
    for i_loc, fi in enumerate(follower_indices):
        t_events = sorted([ev[0] for ev in et_data_et['trigger_log']
                           if ev[1] == fi])
        cum_count = np.arange(1, len(t_events) + 1)
        if t_events:
            axes14[0, 1].step(t_events, cum_count, where='post',
                              linewidth=1.2, label=f'Agent {fi}')
    # 连续通信基线
    axes14[0, 1].plot([0, T_total], [0, len(times_2nd_cont)], 'k--',
                      linewidth=1.5, alpha=0.4, label='连续通信')
    axes14[0, 1].set_xlabel('时间 (s)')
    axes14[0, 1].set_ylabel('累积通信次数')
    axes14[0, 1].set_title('(b) 累积通信次数', fontsize=11)
    axes14[0, 1].legend(fontsize=8)
    axes14[0, 1].grid(True, alpha=0.3)
    axes14[0, 1].set_xlim([0, T_total])

    # (c) 各 agent 通信率柱状图
    bar_x = np.arange(len(follower_indices))
    bar_vals = comm['per_agent']
    bar_colors = [f'C{i}' for i in range(len(follower_indices))]
    bars14 = axes14[1, 0].bar(bar_x, bar_vals, color=bar_colors,
                               edgecolor='black', linewidth=0.5)
    axes14[1, 0].axhline(y=100, color='gray', linestyle='--', linewidth=1.5,
                          alpha=0.4, label='连续通信 (100%)')
    axes14[1, 0].axhline(y=comm['mean'], color='red', linestyle='-.',
                          linewidth=1.5, alpha=0.7,
                          label=f'ET 平均 ({comm["mean"]:.1f}%)')
    axes14[1, 0].set_xlabel('Follower Agent ID')
    axes14[1, 0].set_ylabel('通信率 (%)')
    axes14[1, 0].set_title('(c) 各 Follower 通信率', fontsize=11)
    axes14[1, 0].set_xticks(bar_x)
    axes14[1, 0].set_xticklabels(follower_indices)
    axes14[1, 0].legend()
    axes14[1, 0].grid(True, alpha=0.3, axis='y')

    # (d) 自适应参数 φ_i 演化
    for i_loc, fi in enumerate(follower_indices):
        axes14[1, 1].plot(times_2nd_et,
                          et_data_et['phi_history'][:, i_loc],
                          linewidth=1.2, label=f'Agent {fi}')
    axes14[1, 1].set_xlabel('时间 (s)')
    axes14[1, 1].set_ylabel(r'$\phi_i(t)$')
    axes14[1, 1].set_title(r'(d) 自适应参数 $\phi_i$ 演化', fontsize=11)
    axes14[1, 1].legend(fontsize=8)
    axes14[1, 1].grid(True, alpha=0.3)
    axes14[1, 1].set_xlim([0, T_total])

    fig14.suptitle(
        rf'自适应事件触发通信分析 ($\mu$={et_mu}, $\varpi$={et_varpi}, $\phi_0$={et_phi_0})',
        fontsize=14,
    )
    fig14.tight_layout()
    fig14.savefig(figure_path('fig14_et_communication_analysis.png'),
                  dpi=200, bbox_inches='tight')
    archive.save_figure(fig14, 'fig14_et_communication_analysis')
    print("  已保存: fig14_et_communication_analysis.png")

    # ==== 图15: 不同 ET 参数对比 ====
    mu_values = [0.001, 0.01, 0.1]
    fig15, axes15 = plt.subplots(1, 2, figsize=(14, 5))

    for mu_v in mu_values:
        et_cmp = EventTriggerManager(
            n_agents=10, d=d,
            follower_indices=follower_indices,
            leader_indices=leader_indices,
            Omega=Omega,
            mu=mu_v, varpi=et_varpi, phi_0=et_phi_0,
        )
        (t_cmp, _, err_cmp, _, et_data_cmp) = simulate_second_order_et(
            controller, init_pos, init_vel, leader_traj, (0, T_total), dt=dt,
            et_manager=et_cmp,
        )
        cr = et_data_cmp['comm_rates']['mean']
        axes15[0].semilogy(t_cmp, err_cmp + 1e-16, linewidth=1.5,
                           label=f'μ={mu_v} ({cr:.1f}%)')
        axes15[1].bar(f'μ={mu_v}', cr, alpha=0.7, edgecolor='black',
                      linewidth=0.5)

    # 连续通信基线
    axes15[0].semilogy(times_2nd_cont, err_2nd_cont + 1e-16, 'k--',
                       linewidth=1.5, alpha=0.5, label='连续通信')
    axes15[0].set_xlabel('时间 (s)')
    axes15[0].set_ylabel('编队误差')
    axes15[0].set_title('(a) 不同 μ 下的误差收敛', fontsize=11)
    axes15[0].legend()
    axes15[0].grid(True, alpha=0.3)
    axes15[0].set_xlim([0, T_total])

    axes15[1].axhline(y=100, color='gray', linestyle='--', linewidth=1.5,
                      alpha=0.4)
    axes15[1].set_ylabel('平均通信率 (%)')
    axes15[1].set_title('(b) 通信率对比', fontsize=11)
    axes15[1].grid(True, alpha=0.3, axis='y')

    fig15.suptitle('ET 参数 μ 对性能的影响', fontsize=14)
    fig15.tight_layout()
    fig15.savefig(figure_path('fig15_et_parameter_comparison.png'),
                  dpi=200, bbox_inches='tight')
    archive.save_figure(fig15, 'fig15_et_parameter_comparison')
    print("  已保存: fig15_et_parameter_comparison.png")

    # ----------------------------------------------------------
    # Step 7.8: 层级重组 (RHF) 验证
    # ----------------------------------------------------------
    print("\n[Step 7.8] 层级重组 (Hierarchical Reorganization) 验证...")
    print("  参考: Li & Dong (2024), arXiv:2406.11219")

    # --- 场景设计: 编队 U 形转弯 ---
    # Phase 0 (0-10s): 编队建立, leaders=[0,1,2,5] (原始)
    # Phase 1 (10s切换): 编队向 +Y 转弯, 选新 leader
    # Phase 2 (25s切换): 编队向 -X 转弯, 再选新 leader
    # Phase 3 (40-50s): 稳态飞行

    rhf_gain = 10.0
    rhf_u_max = CRAZYFLIE_COMM['max_velocity']

    # 为初始层级预计算 power-centric 应力矩阵
    print("\n  [预计算] Phase 0: 原始层级 leaders=[0,1,2,5]...")
    leaders_p0 = [0, 1, 2, 5]
    span_p0 = check_affine_span(nominal_pos, leaders_p0)
    print(f"    仿射张成检查: rank={span_p0['rank']}, "
          f"required={span_p0['required_rank']}, valid={span_p0['valid']}")
    Omega_p0, info_p0 = compute_power_centric_stress_matrix(
        nominal_pos, leaders_p0)
    print(f"    λ_min(Ω_ff) = {info_p0['min_eig_ff']:.6f}, "
          f"边数 = {info_p0['n_edges']}")

    # Phase 1: 选择 +Y 方向最优 leader
    print("\n  [预计算] Phase 1: 向 +Y 方向选择 leader...")
    leaders_p1, sel_info_p1 = select_leaders_for_direction(
        nominal_pos, direction=[0, 1, 0], n_leaders=4, d=3)
    span_p1 = check_affine_span(nominal_pos, leaders_p1)
    print(f"    新 leader: {leaders_p1}")
    print(f"    选择方法: {sel_info_p1['method']}")
    print(f"    仿射张成: rank={span_p1['rank']}, valid={span_p1['valid']}")
    Omega_p1, info_p1 = compute_power_centric_stress_matrix(
        nominal_pos, leaders_p1)
    print(f"    λ_min(Ω_ff) = {info_p1['min_eig_ff']:.6f}, "
          f"边数 = {info_p1['n_edges']}")

    # Phase 2: 选择 -X 方向最优 leader
    print("\n  [预计算] Phase 2: 向 -X 方向选择 leader...")
    leaders_p2, sel_info_p2 = select_leaders_for_direction(
        nominal_pos, direction=[-1, 0, 0], n_leaders=4, d=3)
    span_p2 = check_affine_span(nominal_pos, leaders_p2)
    print(f"    新 leader: {leaders_p2}")
    print(f"    选择方法: {sel_info_p2['method']}")
    print(f"    仿射张成: rank={span_p2['rank']}, valid={span_p2['valid']}")
    Omega_p2, info_p2 = compute_power_centric_stress_matrix(
        nominal_pos, leaders_p2)
    print(f"    λ_min(Ω_ff) = {info_p2['min_eig_ff']:.6f}, "
          f"边数 = {info_p2['n_edges']}")

    # Dwell-time 计算
    dwell_p1 = compute_dwell_time(rhf_gain, info_p1['min_eig_ff'], 2.0, 0.1)
    dwell_p2 = compute_dwell_time(rhf_gain, info_p2['min_eig_ff'], 2.0, 0.1)
    print(f"\n  [驻留时间] Phase 1→2 最小驻留: {dwell_p1:.2f}s")
    print(f"  [驻留时间] Phase 2→3 最小驻留: {dwell_p2:.2f}s")

    # --- Leader 目标位置设计 (温和 U 形转弯) ---
    # Phase 0: 标称位置 + 小幅 +X 平移
    targets_p0 = nominal_pos[leaders_p0] + np.array([0.5, 0.0, 0.0])
    # Phase 1: 标称位置 + 向 +Y 平移
    targets_p1 = nominal_pos[leaders_p1] + np.array([0.5, 1.0, 0.0])
    # Phase 2: 标称位置 + 向 -X+Y 平移 (完成 U 形)
    targets_p2 = nominal_pos[leaders_p2] + np.array([-0.5, 1.0, 0.0])

    # 构建 RHF 调度表 — 阶段间隔满足驻留时间要求
    rhf_schedule = [
        {
            't_switch': 0.0,
            'leader_indices': leaders_p0,
            'leader_targets': targets_p0,
            't_transition': 3.0,
            'omega': Omega_p0,
            'adj': info_p0['adj_matrix'],
            'label': 'Phase 0: 建立+X平移',
        },
        {
            't_switch': 35.0,
            'leader_indices': leaders_p1,
            'leader_targets': targets_p1,
            't_transition': 3.0,
            'omega': Omega_p1,
            'adj': info_p1['adj_matrix'],
            'label': 'Phase 1: +Y转弯',
        },
        {
            't_switch': 70.0,
            'leader_indices': leaders_p2,
            'leader_targets': targets_p2,
            't_transition': 3.0,
            'omega': Omega_p2,
            'adj': info_p2['adj_matrix'],
            'label': 'Phase 2: -X转弯(U形)',
        },
    ]

    T_rhf = 105.0

    # --- 运行 RHF 仿真 ---
    print(f"\n  运行 RHF 仿真 (T={T_rhf}s, {len(rhf_schedule)} 阶段)...")
    rhf_controller = AFCController(
        Omega_p0, leaders_p0, gain=rhf_gain, damping=1.0,
        u_max=rhf_u_max, saturation_type='smooth')

    np.random.seed(123)
    rhf_init_pos = nominal_pos.copy()
    rhf_init_pos += np.random.randn(10, 3) * 0.1
    rhf_init_vel = np.zeros((10, 3))

    (times_rhf, pos_rhf, err_rhf, accel_rhf,
     rhf_data) = simulate_rhf(
        rhf_controller, rhf_init_pos, rhf_init_vel,
        nominal_pos, rhf_schedule, (0, T_rhf), dt=dt)

    print(f"    仿真步数: {len(times_rhf)}")
    print(f"    层级切换次数: {rhf_data['n_switches']}")
    for log in rhf_data['switch_log']:
        print(f"    [{log['label']}] t={log['t_switch']:.1f}s, "
              f"切换前误差={log['pre_switch_error']:.4f}, "
              f"峰值误差={log['post_switch_peak_error']:.4f}, "
              f"恢复时间={log['recovery_time']:.2f}s")

    # --- fig16: RHF 3D 轨迹 ---
    fig16 = plt.figure(figsize=(14, 10))
    ax16 = fig16.add_subplot(111, projection='3d')
    phase_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 3 阶段颜色
    switch_times = [s['t_switch'] for s in rhf_schedule]

    n_rhf = pos_rhf.shape[1]
    for i in range(n_rhf):
        traj = pos_rhf[:, i, :]
        for p_idx in range(len(rhf_schedule)):
            t_start = switch_times[p_idx]
            t_end = (switch_times[p_idx + 1]
                     if p_idx + 1 < len(switch_times) else T_rhf)
            mask = (times_rhf >= t_start) & (times_rhf <= t_end)
            seg = traj[mask]
            if len(seg) > 1:
                ax16.plot(seg[:, 0], seg[:, 1], seg[:, 2],
                          color=phase_colors[p_idx], alpha=0.5, linewidth=0.8)
        # 起点 / 终点标记
        ax16.scatter(*traj[0], marker='o', s=30, c='gray', zorder=5)
        ax16.scatter(*traj[-1], marker='s', s=30, c='red', zorder=5)
        ax16.text(traj[-1, 0], traj[-1, 1], traj[-1, 2],
                  f' {i}', fontsize=7)

    # 标注每阶段的 leader
    for p_idx, sch in enumerate(rhf_schedule):
        ls = sch['leader_indices']
        t_mid_idx = np.searchsorted(
            times_rhf,
            sch['t_switch'] + (switch_times[p_idx + 1] - sch['t_switch']) / 2
            if p_idx + 1 < len(switch_times)
            else sch['t_switch'] + (T_rhf - sch['t_switch']) / 2)
        t_mid_idx = min(t_mid_idx, len(times_rhf) - 1)
        for li in ls:
            mid_pos = pos_rhf[t_mid_idx, li, :]
            ax16.scatter(*mid_pos, marker='*', s=120,
                         c=phase_colors[p_idx], edgecolors='k',
                         zorder=10)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color=phase_colors[i], lw=2,
               label=rhf_schedule[i]['label'])
        for i in range(len(rhf_schedule))
    ]
    legend_elems.append(
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
               markersize=12, label='Leader'))
    ax16.legend(handles=legend_elems, fontsize=8, loc='upper left')
    ax16.set_xlabel('X (m)')
    ax16.set_ylabel('Y (m)')
    ax16.set_zlabel('Z (m)')
    ax16.set_title('Fig.16 层级重组 — 3D 轨迹 (U-Turn)')
    fig16.tight_layout()
    fig16.savefig(figure_path('fig16_rhf_3d_trajectory.png'), dpi=150)
    print("  已保存: fig16_rhf_3d_trajectory.png")

    # --- fig17: RHF 误差演化 + 切换标记 ---
    fig17, ax17 = plt.subplots(figsize=(12, 5))
    ax17.plot(times_rhf, err_rhf, 'b-', linewidth=1.0,
              label='编队误差 $\\|e(t)\\|$')

    for p_idx, sch in enumerate(rhf_schedule):
        if p_idx == 0:
            continue  # 初始阶段不画切换线
        ax17.axvline(sch['t_switch'], color=phase_colors[p_idx],
                     linestyle='--', linewidth=1.5, alpha=0.8,
                     label=f"切换@{sch['t_switch']:.0f}s: {sch['label']}")

    # 标注峰值误差和恢复时间
    for log in rhf_data['switch_log']:
        peak_t = log['t_switch'] + log['recovery_time'] / 2
        ax17.annotate(
            f"峰值={log['post_switch_peak_error']:.3f}\n"
            f"恢复={log['recovery_time']:.1f}s",
            xy=(log['t_switch'] + 1, log['post_switch_peak_error']),
            fontsize=7, color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=0.8),
            xytext=(log['t_switch'] + 3, log['post_switch_peak_error'] * 1.3))

    ax17.set_xlabel('时间 (s)')
    ax17.set_ylabel('编队误差')
    ax17.set_title('Fig.17 层级重组 — 误差演化与切换瞬态')
    ax17.legend(fontsize=8, loc='upper right')
    ax17.grid(True, alpha=0.3)
    fig17.tight_layout()
    fig17.savefig(figure_path('fig17_rhf_error_evolution.png'), dpi=150)
    print("  已保存: fig17_rhf_error_evolution.png")

    # --- fig18: 各阶段通信拓扑对比 ---
    fig18, axes18 = plt.subplots(1, len(rhf_schedule), figsize=(6 * len(rhf_schedule), 5))
    if len(rhf_schedule) == 1:
        axes18 = [axes18]
    for p_idx, sch in enumerate(rhf_schedule):
        ax = axes18[p_idx]
        adj = sch['adj']
        ls = sch['leader_indices']
        fs = [j for j in range(n_rhf) if j not in ls]
        # 使用标称位置的 x-y 坐标作为节点布局
        pos_2d = {j: (nominal_pos[j, 0], nominal_pos[j, 1]) for j in range(n_rhf)}
        # 画边
        for r in range(n_rhf):
            for c in range(r + 1, n_rhf):
                if adj[r, c] != 0:
                    x_vals = [pos_2d[r][0], pos_2d[c][0]]
                    y_vals = [pos_2d[r][1], pos_2d[c][1]]
                    ax.plot(x_vals, y_vals, 'k-', linewidth=0.6, alpha=0.4)
        # 画节点
        for j in range(n_rhf):
            color = phase_colors[p_idx] if j in ls else '#cccccc'
            marker = '*' if j in ls else 'o'
            size = 200 if j in ls else 80
            ax.scatter(pos_2d[j][0], pos_2d[j][1],
                       c=color, marker=marker, s=size,
                       edgecolors='k', zorder=5)
            ax.annotate(str(j), pos_2d[j], fontsize=8,
                        ha='center', va='bottom',
                        xytext=(0, 6), textcoords='offset points')
        ax.set_title(f"{sch['label']}\nLeaders={ls}, Edges={int(adj.sum()/2)}",
                     fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
    fig18.suptitle('Fig.18 层级重组 — 通信拓扑切换', fontsize=12)
    fig18.tight_layout()
    fig18.savefig(figure_path('fig18_rhf_topology.png'), dpi=150)
    print("  已保存: fig18_rhf_topology.png")

    # 恢复控制器到原始状态 (供后续 Step 8 使用)
    controller.update_omega(Omega, leader_indices)

    # ----------------------------------------------------------
    # Step 7.9: 综合场景 - 地面起飞并形成空中金字塔
    # ----------------------------------------------------------
    pyramid_results = run_pyramid_integrated_mission(dt=dt, archive=archive)
    pyramid_times = pyramid_results['times']
    pyramid_pos_hist = pyramid_results['positions']
    pyramid_errors = pyramid_results['errors']
    pyramid_ctrl = pyramid_results['control_inputs']
    pyramid_data = pyramid_results['data']
    pyramid_schedule = pyramid_results['schedule']
    pyramid_phase_infos = pyramid_results['phase_info']
    pyramid_total_time = pyramid_results['total_time']
    pyramid_d_safe = pyramid_results['d_safe']
    pyramid_d_activate = pyramid_results['d_activate']
    pyramid_cbf_gamma = pyramid_results['cbf_gamma']
    pyramid_comm = pyramid_results['comm']

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
    assert Omega is not None
    print("\n[附录] 应力矩阵 Ω:")
    print(np.array2string(Omega, precision=4, suppress_small=True))

    print("\n[附录] Ω 的特征值:")
    eigvals = np.linalg.eigvalsh(Omega)
    print(f"  {np.array2string(eigvals, precision=6)}")

    plt.close('all')

    # ----------------------------------------------------------
    # Step 9: 存档
    # ----------------------------------------------------------
    print("\n[Step 9] 打包存档...")

    # 保存所有数值数据
    archive.save_arrays(
        times=times,
        positions=pos_hist,
        errors=errors,
        control_inputs=ctrl_inputs,
        positions_nosat=pos_hist_ns,
        errors_nosat=errors_ns,
        control_inputs_nosat=ctrl_inputs_ns,
        nominal_positions=nominal_pos,
        initial_positions=init_pos,
        cbf_min_distances_nocbf=min_d_nocbf,
        cbf_min_distances_wcbf=min_d_wcbf,
        cbf_modifications=cbf_data_wcbf['modifications'],
        cbf_n_active=cbf_data_wcbf['n_active'],
        eso_errors_baseline=err_eso_bl,
        eso_errors_no_eso=err_eso_nd,
        eso_errors_with_eso=err_eso_wd,
        eso_disturbances_true=eso_data_wd['disturbances_true'],
        eso_disturbances_est=eso_data_wd['disturbances_est'],
        et_errors_continuous=err_2nd_cont,
        et_errors_triggered=err_2nd_et,
        et_phi_history=et_data_et['phi_history'],
        pyramid_times=pyramid_times,
        pyramid_positions=pyramid_pos_hist,
        pyramid_errors=pyramid_errors,
        pyramid_control_inputs=pyramid_ctrl,
        pyramid_min_distances=pyramid_data['min_distances'],
        pyramid_cbf_modifications=pyramid_data['cbf_modifications'],
        pyramid_active_constraints=pyramid_data['n_active_constraints'],
        pyramid_disturbances_true=pyramid_data['disturbances_true'],
        pyramid_disturbances_est=pyramid_data['disturbances_est'],
        pyramid_phi_history=pyramid_data['phi_history'],
    )

    # 保存应力矩阵与邻接矩阵
    archive.save_matrix_csv('stress_matrix', Omega,
                            header=','.join(f'agent_{i}' for i in range(10)))
    archive.save_matrix_csv('adj_sparse', adj)
    archive.save_matrix_csv('adj_complete', adj_complete)

    # 保存仿真参数
    archive.save_params({
        'formation': {
            'type': 'double_pentagon',
            'n_agents': 10,
            'radius': 1.0,
            'height': 1.0,
            'leader_indices': leader_indices,
            'follower_indices': follower_indices,
        },
        'controller': {
            'gain': gain,
            'u_max': u_max,
            'saturation_type': 'smooth',
            'damping': controller.damping,
        },
        'simulation': {
            'dt': dt,
            'T_total': T_total,
            't_settle': t_settle,
            't_trans': t_trans,
            't_hold': t_hold,
            'random_seed': 42,
            'init_noise_std': 0.5,
        },
        'sparse_design': info,
        'convergence': {
            'rate': rate,
            'time_constant': tau,
        },
        'results': {
            'initial_error': float(errors[0]),
            'final_error_sat': float(errors[-1]),
            'final_error_nosat': float(errors_ns[-1]),
            'max_ctrl_sat': float(ctrl_norms.max()),
            'max_ctrl_nosat': float(ctrl_norms_ns.max()),
            'affine_error': float(affine_error),
        },
        'cbf_collision_avoidance': {
            'd_safe': d_safe,
            'gamma': gamma_cbf,
            'd_activate': d_activate,
            'cbf_noise_std': cbf_noise_std,
            'min_dist_nocbf': float(min_d_nocbf.min()),
            'min_dist_wcbf': float(min_d_wcbf.min()),
            'cbf_active_steps': int(n_cbf_active),
            'max_modification': float(cbf_data_wcbf['modifications'].max()),
            'final_error_nocbf': float(err_nocbf[-1]),
            'final_error_wcbf': float(err_wcbf[-1]),
        },
        'eso_disturbance_rejection': {
            'omega_o': omega_o,
            'beta1': 2 * omega_o,
            'beta2': omega_o ** 2,
            'w_const': w_const_vec.tolist(),
            'ou_theta': ou_theta,
            'ou_sigma': ou_sigma,
            'wind_seed': wind_seed,
            'theoretical_steady_error': float(delta_norm),
            'final_error_baseline': float(err_eso_bl[-1]),
            'final_error_no_eso': float(err_eso_nd[-1]),
            'final_error_with_eso': float(err_eso_wd[-1]),
        },
        'stress_matrix_validation': results,
        'eigenvalues_omega': eigvals.tolist(),
        'event_triggered_communication': {
            'model': 'second_order_integrator',
            'mu': et_mu,
            'varpi': et_varpi,
            'phi_0': et_phi_0,
            'K_d': K_d,
            'mean_comm_rate_pct': comm['mean'],
            'total_triggers': comm['total_triggers'],
            'total_possible': comm['total_possible'],
            'per_agent_comm_rate': comm['per_agent'].tolist(),
            'final_error_continuous': float(err_2nd_cont[-1]),
            'final_error_et': float(err_2nd_et[-1]),
            'comm_saving_pct': 100 - comm['mean'],
        },
        'pyramid_integrated_mission': {
            'formation': 'aerial_pyramid_10',
            'total_time': pyramid_total_time,
            'dt': dt,
            'phase_info': pyramid_phase_infos,
            'initial_leaders': pyramid_schedule[0]['leader_indices'],
            'd_safe': pyramid_d_safe,
            'd_activate': pyramid_d_activate,
            'cbf_gamma': pyramid_cbf_gamma,
            'final_error': float(pyramid_errors[-1]),
            'min_distance': float(pyramid_data['min_distances'].min()),
            'mean_comm_rate_pct': pyramid_comm['mean'],
            'comm_saving_pct': 100.0 - pyramid_comm['mean'],
            'total_triggers': pyramid_comm['total_triggers'],
            'total_possible': pyramid_comm['total_possible'],
            'cbf_active_ratio_pct': float(100.0 * np.mean(pyramid_data['n_active_constraints'] > 0)),
            'final_estimation_error': float(pyramid_data['estimation_error'][-1]),
            'n_switches': pyramid_data['n_switches'],
        },
    })

    archive.finalize()

    print("\n仿真完成！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AFC 主仿真脚本')
    parser.add_argument(
        '--scenario',
        choices=['all', 'pyramid'],
        default='all',
        help='all: 完整主流程；pyramid: 仅运行综合金字塔任务',
    )
    args = parser.parse_args()
    main(selected_scenario=args.scenario)
