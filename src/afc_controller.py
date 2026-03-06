"""
afc_controller.py - 仿射编队控制(AFC)分布式控制律

仿射编队控制的核心思想：
  Leader 驱动编队的仿射变换（平移、旋转、缩放、剪切）
  Follower 通过分布式控制律自动收敛到正确位置

控制律（一阶积分器模型 ṗ_i = u_i）：
  对 Follower i:
    u_i = -K_p Σ_{j∈N(i)} ω_ij (p_i - p_j)

  矩阵形式：
    ṗ_f = -K_p (Ω_ff p_f + Ω_fl p_l)

  稳态解：
    p_f* = -Ω_ff^{-1} Ω_fl p_l

控制律（二阶积分器模型 p̈_i = u_i）：
  对 Follower i:
    u_i = -K_p Σ_{j∈N(i)} ω_ij (p_i - p_j) - K_d v_i

  矩阵形式：
    p̈_f = -K_p (Ω_ff p_f + Ω_fl p_l) - K_d ṗ_f

输入饱和约束：
  实际无人机的控制输入（速度/加速度）受物理限制，需要引入饱和函数：
    u_sat = σ(u_raw, u_max)

  支持三种饱和模式：
    - smooth: σ(u) = u_max · tanh(u / u_max)  （平滑、可微、保方向）
    - norm:   per-agent 范数裁剪，保持方向不变
    - clip:   per-component 硬裁剪

关键定理（仿射不变性）：
  若 leader 位置为标称位置的仿射像 p_l = A r_l + 1⊗b，
  则 follower 稳态位置为 p_f* = A r_f + 1⊗b
  即整个编队实现了相同的仿射变换。

证明：
  p_f* = -Ω_ff^{-1} Ω_fl (A r_l + 1_l ⊗ b)
       = A(-Ω_ff^{-1} Ω_fl r_l) + (-Ω_ff^{-1} Ω_fl 1_l) ⊗ b
       = A r_f + 1_f ⊗ b    ✓
  （利用了 Ω r_k = 0 和 Ω 1 = 0 的性质）
"""

import numpy as np


class AFCController:
    """
    仿射编队控制分布式控制器。

    Parameters
    ----------
    stress_matrix : ndarray (n, n)
        应力矩阵 Ω
    leader_indices : list of int
        Leader 智能体索引
    gain : float
        比例增益 K_p，控制收敛速度
    damping : float
        阻尼增益 K_d，仅用于二阶模型
    u_max : float or None
        控制输入饱和上限（每轴最大速度/加速度），None 表示无饱和
    saturation_type : str
        饱和类型: 'smooth' (tanh平滑饱和), 'norm' (范数裁剪), 'clip' (分量裁剪)
    """

    def __init__(self, stress_matrix, leader_indices, gain=1.0, damping=2.0,
                 u_max=None, saturation_type='smooth'):
        self.Omega = stress_matrix
        self.n = stress_matrix.shape[0]
        self.leader_indices = sorted(leader_indices)
        self.follower_indices = sorted(set(range(self.n)) - set(leader_indices))
        self.n_l = len(self.leader_indices)
        self.n_f = len(self.follower_indices)
        self.gain = gain
        self.damping = damping
        self.u_max = u_max
        self.saturation_type = saturation_type

        # 预提取子矩阵
        self.Omega_ff = self.Omega[np.ix_(self.follower_indices, self.follower_indices)]
        self.Omega_fl = self.Omega[np.ix_(self.follower_indices, self.leader_indices)]

    def update_omega(self, new_omega, new_leader_indices=None):
        """
        动态更新应力矩阵，支持层级重组（Hierarchical Reorganization）。

        切换时序：
          1. 外部计算新 Omega（基于新拓扑和新 leader 选择）
          2. 调用 update_omega() 更新控制器内部状态
          3. 下一个控制步自动使用新参数

        参考：Li & Dong (2024) "A Flexible and Resilient Formation Approach
              based on Hierarchical Reorganization", arXiv:2406.11219

        Parameters
        ----------
        new_omega : ndarray (n, n)
            新的应力矩阵
        new_leader_indices : list of int or None
            新的 leader 索引集合。若为 None，保持当前 leader 不变。

        Returns
        -------
        switch_info : dict
            切换信息，包含新旧参数对比
        """
        old_leader_indices = self.leader_indices.copy()
        old_min_eig = float(np.min(np.linalg.eigvalsh(self.Omega_ff)))

        self.Omega = new_omega
        if new_leader_indices is not None:
            self.leader_indices = sorted(new_leader_indices)
            self.follower_indices = sorted(
                set(range(self.n)) - set(self.leader_indices)
            )
            self.n_l = len(self.leader_indices)
            self.n_f = len(self.follower_indices)

        # 重新提取子矩阵
        self.Omega_ff = self.Omega[np.ix_(self.follower_indices, self.follower_indices)]
        self.Omega_fl = self.Omega[np.ix_(self.follower_indices, self.leader_indices)]

        new_min_eig = float(np.min(np.linalg.eigvalsh(self.Omega_ff)))

        return {
            'old_leaders': old_leader_indices,
            'new_leaders': self.leader_indices,
            'old_min_eig_ff': old_min_eig,
            'new_min_eig_ff': new_min_eig,
            'leader_changed': old_leader_indices != self.leader_indices,
        }

    def saturate(self, u):
        """
        对控制输入施加饱和约束。

        Parameters
        ----------
        u : ndarray (..., d)
            原始控制输入，可以是 (d,) 单智能体或 (n_f, d) 全体跟随者

        Returns
        -------
        u_sat : ndarray (..., d)
            饱和后的控制输入
        """
        if self.u_max is None:
            return u

        if self.saturation_type == 'smooth':
            # tanh 平滑饱和：σ(u) = u_max * tanh(u / u_max)
            # 性质：|σ(u)| < u_max, σ'(0)=1, 连续可微
            return self.u_max * np.tanh(u / self.u_max)

        elif self.saturation_type == 'norm':
            # Per-agent 范数裁剪：保持方向，限制合速度大小
            if u.ndim == 1:
                norm = np.linalg.norm(u)
                if norm > self.u_max:
                    return u * (self.u_max / norm)
                return u
            else:
                norms = np.linalg.norm(u, axis=1, keepdims=True)
                scale = np.where(norms > self.u_max,
                                 self.u_max / np.maximum(norms, 1e-12), 1.0)
                return u * scale

        else:  # 'clip'
            return np.clip(u, -self.u_max, self.u_max)

    def follower_input(self, agent_id, positions, velocities=None):
        """
        计算单个 Follower 的控制输入。

        u_i = -K_p Σ_j ω_ij (p_i - p_j) [- K_d v_i]

        Parameters
        ----------
        agent_id : int
            Follower 的全局索引
        positions : ndarray (n, d)
            所有智能体当前位置
        velocities : ndarray (n, d) or None
            所有智能体当前速度（二阶模型时需要）

        Returns
        -------
        u : ndarray (d,)
            控制输入向量
        """
        d = positions.shape[1]
        u = np.zeros(d)

        for j in range(self.n):
            if j != agent_id and self.Omega[agent_id, j] != 0:
                u -= self.gain * self.Omega[agent_id, j] * (
                    positions[agent_id] - positions[j]
                )

        if velocities is not None:
            u -= self.damping * velocities[agent_id]

        return self.saturate(u)

    def all_follower_inputs(self, positions, velocities=None):
        """
        使用矩阵形式计算所有 Follower 的控制输入（高效版本）。

        u_f = -K_p (Ω_ff p_f + Ω_fl p_l) [- K_d v_f]

        Parameters
        ----------
        positions : ndarray (n, d)
        velocities : ndarray (n, d) or None

        Returns
        -------
        u_f : ndarray (n_f, d)
            所有 Follower 的控制输入
        """
        p_f = positions[self.follower_indices]   # (n_f, d)
        p_l = positions[self.leader_indices]     # (n_l, d)

        u_f = -self.gain * (self.Omega_ff @ p_f + self.Omega_fl @ p_l)

        if velocities is not None:
            v_f = velocities[self.follower_indices]
            u_f -= self.damping * v_f

        return self.saturate(u_f)

    def steady_state(self, leader_positions):
        """
        计算给定 Leader 位置下的 Follower 稳态位置。

        p_f* = -Ω_ff^{-1} Ω_fl p_l

        Parameters
        ----------
        leader_positions : ndarray (n_l, d)
            Leader 目标位置

        Returns
        -------
        p_f_star : ndarray (n_f, d)
            Follower 稳态位置
        """
        return -np.linalg.solve(self.Omega_ff, self.Omega_fl @ leader_positions)

    def formation_error(self, positions, leader_positions):
        """
        计算当前编队误差。

        error = ||p_f - p_f*||

        Parameters
        ----------
        positions : ndarray (n, d)
            所有智能体当前位置
        leader_positions : ndarray (n_l, d)
            Leader 当前位置

        Returns
        -------
        error : float
            编队误差（Frobenius 范数）
        per_agent_error : ndarray (n_f,)
            每个 Follower 的位置误差
        """
        p_f = positions[self.follower_indices]
        p_f_star = self.steady_state(leader_positions)

        per_agent = np.linalg.norm(p_f - p_f_star, axis=1)
        return float(np.linalg.norm(p_f - p_f_star)), per_agent

    def convergence_rate_bound(self):
        """
        计算收敛速率的理论下界。

        一阶模型：收敛速率 = K_p * λ_min(Ω_ff)
        二阶模型：依赖于阻尼比

        Returns
        -------
        rate : float
            指数收敛速率
        time_constant : float
            时间常数 τ = 1/rate（秒）
        """
        min_eig = np.min(np.linalg.eigvalsh(self.Omega_ff))
        rate = self.gain * min_eig
        time_constant = 1.0 / rate if rate > 0 else float('inf')
        return rate, time_constant
