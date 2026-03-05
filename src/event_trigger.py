"""
event_trigger.py - 自适应事件触发通信管理器

二阶积分器仿射编队控制的事件触发通信机制：
  每个 Follower i 维护最近广播位置 p̂_i，仅在触发条件满足时广播更新。
  控制律使用邻居的广播位置计算编队偏差项，本地速度阻尼不受影响。

控制律（二阶积分器 + 事件触发）：
  ü_i = -K_p Σ_{j∈N(i)} ω_ij (p_i(t) - p̂_j) - K_d v_i

  等价矩阵形式：
    ü_f = -K_p (Ω_ff p̂_f + Ω_fl p̂_l) - K_p Ω_ff (p_f - p̂_f) - K_d v_f
  即：标称项（广播值）+ 自身修正项 + 阻尼项

自适应触发条件（简化 Liu et al. 2025, 适用于单/二阶积分器）：
  Agent i 广播更新当：
    ||e_i(t)||² > (1/φ_i(t)) ||ξ̂_i(t)||² + μ exp(-ϖ t)

  其中：
    e_i(t) = p_i(t) - p̂_i         广播误差
    ξ̂_i = Σ_{j∈N(i)} ω_ij(p̂_i - p̂_j)  基于广播值的编队偏差
    φ̇_i(t) = ||e_i(t)||²           自适应参数（单调递增）
    μ, ϖ > 0                       指数衰减项参数

  自适应机制的优势：
    - φ_i(t) 单调递增 → 1/φ_i(t) 单调递减
    - 编队接近稳态时，e_i 较小，阈值自动收紧
    - 不需要全局拓扑信息（如 ||Ω_ff||）来确定阈值

Zeno-free 保证：
  指数衰减项 μ exp(-ϖ t) > 0 确保阈值右端严格正，
  ||e_i|| 从 0 连续增长到触发需要正时间间隔。

参考文献：
  [1] C. Liu et al., "Distributed Affine Formation Control of Linear
      Multi-agent Systems with Adaptive Event-triggering," arXiv:2506.16797, 2025.
  [2] X. Yi et al., "Formation Control for Multi-Agent Systems with
      Connectivity Preservation and Event-Triggered Controllers,"
      arXiv:1611.03105, 2016.
"""

import numpy as np


class EventTriggerManager:
    """
    自适应事件触发通信管理器。

    Parameters
    ----------
    n_agents : int
        智能体总数
    d : int
        空间维数
    follower_indices : list of int
        Follower 全局索引
    leader_indices : list of int
        Leader 全局索引
    Omega : ndarray (n, n)
        应力矩阵
    mu : float
        指数衰减项系数 μ > 0
    varpi : float
        指数衰减速率 ϖ > 0
    phi_0 : float
        自适应参数 φ_i 的初始值 (> 0)
    """

    def __init__(self, n_agents, d, follower_indices, leader_indices,
                 Omega, mu=0.01, varpi=0.5, phi_0=1.0):
        self.n = n_agents
        self.d = d
        self.follower_indices = list(follower_indices)
        self.leader_indices = list(leader_indices)
        self.n_f = len(self.follower_indices)
        self.n_l = len(self.leader_indices)
        self.Omega = Omega
        self.mu = mu
        self.varpi = varpi
        self.phi_0 = phi_0

        # 每个 agent 的最近广播位置
        self.p_hat = np.zeros((n_agents, d))

        # 自适应参数 φ_i（仅 follower）
        self.phi = np.full(self.n_f, phi_0)

        # 统计
        self.trigger_counts = np.zeros(n_agents, dtype=int)
        self.trigger_log = []       # [(t, agent_global_id), ...]
        self.total_steps = 0

    def reset(self, positions):
        """初始化：所有 agent 的广播位置设为当前位置。"""
        self.p_hat = positions.copy()
        self.phi = np.full(self.n_f, self.phi_0)
        self.trigger_counts[:] = 0
        self.trigger_log.clear()
        self.total_steps = 0

    def _formation_error_hat(self, fi_global):
        """
        计算 agent fi_global 基于广播位置的编队偏差：
          ξ̂_i = Σ_{j∈N(i)} ω_ij (p̂_i - p̂_j)
        """
        xi = np.zeros(self.d)
        for j in range(self.n):
            if j != fi_global and self.Omega[fi_global, j] != 0:
                xi += self.Omega[fi_global, j] * (
                    self.p_hat[fi_global] - self.p_hat[j]
                )
        return xi

    def check_and_trigger(self, t, positions):
        """
        检查所有 Follower 的触发条件，更新广播位置。

        触发条件：
          ||e_i||² > (1/φ_i) ||ξ̂_i||² + μ exp(-ϖ t)

        同时更新自适应参数：
          φ_i += ||e_i||² * dt  （在仿真循环中由调用者推进）

        Parameters
        ----------
        t : float
            当前时间
        positions : ndarray (n, d)
            所有 agent 当前真实位置

        Returns
        -------
        triggered : list of int
            本步触发广播的 follower 全局索引
        errors_sq : ndarray (n_f,)
            各 follower 的广播误差平方 ||e_i||²
        """
        self.total_steps += 1
        triggered = []
        errors_sq = np.zeros(self.n_f)

        for i_loc, fi in enumerate(self.follower_indices):
            e_i = positions[fi] - self.p_hat[fi]
            e_sq = np.dot(e_i, e_i)
            errors_sq[i_loc] = e_sq

            xi_hat = self._formation_error_hat(fi)
            xi_sq = np.dot(xi_hat, xi_hat)

            threshold = xi_sq / self.phi[i_loc] + self.mu * np.exp(-self.varpi * t)

            if e_sq > threshold:
                self.p_hat[fi] = positions[fi].copy()
                self.trigger_counts[fi] += 1
                self.trigger_log.append((t, fi))
                triggered.append(fi)

        return triggered, errors_sq

    def update_phi(self, errors_sq, dt):
        """
        更新自适应参数 φ_i（前向 Euler）。

        φ̇_i = ||e_i||²  →  φ_i(t+dt) = φ_i(t) + ||e_i||² · dt

        Parameters
        ----------
        errors_sq : ndarray (n_f,)
            各 follower 的广播误差平方
        dt : float
            时间步长
        """
        self.phi += errors_sq * dt

    def update_leaders(self, leader_positions):
        """Leader 位置直接更新广播值（Leader 始终连续通信）。"""
        for i, li in enumerate(self.leader_indices):
            self.p_hat[li] = leader_positions[i].copy()

    def communication_rates(self):
        """
        计算通信率统计。

        Returns
        -------
        rates : dict
            per_agent: ndarray (n_f,)  每个 follower 的通信率 (%)
            mean: float                平均通信率 (%)
            total_triggers: int        总触发次数
            total_possible: int        总可能次数 (n_f × n_steps)
        """
        if self.total_steps == 0:
            return {'per_agent': np.zeros(self.n_f), 'mean': 0.0,
                    'total_triggers': 0, 'total_possible': 0}

        per_agent = np.zeros(self.n_f)
        for i_loc, fi in enumerate(self.follower_indices):
            per_agent[i_loc] = self.trigger_counts[fi] / self.total_steps * 100

        total_triggers = sum(self.trigger_counts[fi]
                             for fi in self.follower_indices)
        total_possible = self.n_f * self.total_steps

        return {
            'per_agent': per_agent,
            'mean': per_agent.mean(),
            'total_triggers': int(total_triggers),
            'total_possible': total_possible,
        }
