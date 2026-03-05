"""
collision_avoidance.py - 基于控制障碍函数 (CBF) 的碰撞避免安全滤波器

理论基础：
  - Ames, A.D. et al. "Control Barrier Functions: Theory and Applications"
    (European Control Conference, 2019)
  - Wang, L. et al. "Safety Barrier Certificates for Collisions-Free Multirobot
    Systems" (IEEE Trans. Robotics, 2017)

核心思想：
  在不改变仿射编队控制 (AFC) 架构的前提下，对标称控制输入施加最小侵入性的
  安全修正，保证智能体间距离在任意时刻不低于安全阈值 d_s。

障碍函数 (Barrier Function)：
  对每对智能体 (i,j)，定义：
    h_ij(p) = ||p_i - p_j||² - d_s²
  h_ij > 0 表示安全，h_ij = 0 为安全边界，h_ij < 0 表示碰撞。

CBF 条件（一阶积分器 ṗ = u）：
  ḣ_ij = 2(p_i - p_j)ᵀ (u_i - u_j) ≥ -γ h_ij
  其中 γ > 0 为 class-K 函数参数，控制趋近安全边界的最大速率。

安全滤波 QP（Quadratic Programming）：
  对全体跟随者，集中式求解：
    min  ||u_f - u_f^nom||²       (最小修正原则)
    s.t. 跟随者-跟随者 CBF 约束:
           2(p_i - p_j)ᵀ(u_i - u_j) ≥ -γ h_ij
         跟随者-领导者 CBF 约束:
           2(p_i - p_l)ᵀ u_i ≥ -γ h_il + 2(p_i - p_l)ᵀ v_l

关键性质：
  1. 前向不变性：若初始安全 h_ij(0) > 0，则在 CBF 约束下 h_ij(t) ≥ 0, ∀t ≥ 0
  2. 最小侵入性：QP 保证安全控制最接近标称 AFC 控制
  3. 与饱和兼容：先施加饱和，再施加 CBF 滤波
  4. 实时可解：QP 变量维度 n_f·d（本系统 6×3=18），约束数 ≤ C(10,2)=45

Crazyflie 2.1 安全参数：
  - 机架对角线: ~92mm (桨尖到桨尖)
  - 安全距离 d_s: 0.2m (考虑桨叶气流和定位误差)
  - 激活距离 d_a: 0.6m (3 × d_s，减少不必要的 QP 求解)
"""

import numpy as np
from scipy.optimize import minimize


# ============================================================
# Crazyflie 2.1 安全参数
# ============================================================

CRAZYFLIE_SAFETY = {
    'frame_diagonal_m': 0.092,     # 机架对角线 (m)
    'prop_diameter_m': 0.046,      # 螺旋桨直径 (m)
    'body_radius_m': 0.046,        # 机体等效半径 ≈ 对角线/2 (m)
    'safety_distance_m': 0.2,      # 安全距离 d_s (m), 中心到中心
    'activate_distance_m': 0.6,    # CBF 约束激活距离 (m)
    'cbf_gamma': 3.0,              # 默认 CBF 衰减率
}


# ============================================================
# CBF 安全滤波器
# ============================================================

class CBFSafetyFilter:
    """
    CBF-QP 碰撞避免安全滤波器。

    Parameters
    ----------
    n_agents : int
        智能体总数（含 Leader 和 Follower）
    leader_indices : list of int
        Leader 索引
    d_safe : float
        安全距离 d_s (m)
    gamma : float
        CBF class-K 函数衰减率 γ
    d_activate : float or None
        CBF 约束激活距离 (m)，None 则使用 3×d_safe
    """

    def __init__(self, n_agents, leader_indices, d_safe=0.2, gamma=3.0,
                 d_activate=None):
        self.n_agents = n_agents
        self.leader_indices = sorted(leader_indices)
        self.follower_indices = sorted(
            set(range(n_agents)) - set(leader_indices)
        )
        self.n_f = len(self.follower_indices)
        self.d_safe = d_safe
        self.gamma = gamma
        self.d_activate = d_activate if d_activate is not None else d_safe * 3.0

    def barrier(self, pi, pj):
        """障碍函数 h_ij = ||p_i - p_j||² - d_s²"""
        diff = pi - pj
        return float(np.dot(diff, diff) - self.d_safe ** 2)

    def filter(self, positions, u_nom, leader_velocities=None):
        """
        对标称控制输入施加 CBF 安全滤波。

        Parameters
        ----------
        positions : ndarray (n, d)
            所有智能体当前位置
        u_nom : ndarray (n_f, d)
            标称控制（来自 AFC + 饱和）
        leader_velocities : ndarray (n_l, d) or None
            Leader 瞬时速度

        Returns
        -------
        u_safe : ndarray (n_f, d)
            安全控制输入
        info : dict
            滤波信息：n_constraints, modified, modification_norm, pairs
        """
        dim = positions.shape[1]
        n_f = self.n_f

        # 构建 QP 线性约束 C u ≥ b
        C_rows = []
        b_vals = []
        pairs_info = []

        # --- Follower-Follower 对 ---
        for i_loc in range(n_f):
            i_glob = self.follower_indices[i_loc]
            for j_loc in range(i_loc + 1, n_f):
                j_glob = self.follower_indices[j_loc]
                diff = positions[i_glob] - positions[j_glob]
                dist_sq = float(np.dot(diff, diff))

                if dist_sq < self.d_activate ** 2:
                    h = dist_sq - self.d_safe ** 2
                    row = np.zeros(n_f * dim)
                    row[i_loc * dim:(i_loc + 1) * dim] = 2.0 * diff
                    row[j_loc * dim:(j_loc + 1) * dim] = -2.0 * diff
                    C_rows.append(row)
                    b_vals.append(-self.gamma * h)
                    pairs_info.append({
                        'type': 'ff', 'i': i_glob, 'j': j_glob,
                        'h': h, 'dist': np.sqrt(dist_sq),
                    })

        # --- Follower-Leader 对 ---
        for i_loc in range(n_f):
            i_glob = self.follower_indices[i_loc]
            for l_loc, l_glob in enumerate(self.leader_indices):
                diff = positions[i_glob] - positions[l_glob]
                dist_sq = float(np.dot(diff, diff))

                if dist_sq < self.d_activate ** 2:
                    h = dist_sq - self.d_safe ** 2
                    v_l = (leader_velocities[l_loc]
                           if leader_velocities is not None
                           else np.zeros(dim))
                    row = np.zeros(n_f * dim)
                    row[i_loc * dim:(i_loc + 1) * dim] = 2.0 * diff
                    C_rows.append(row)
                    b_vals.append(-self.gamma * h + 2.0 * np.dot(diff, v_l))
                    pairs_info.append({
                        'type': 'fl', 'i': i_glob, 'j': l_glob,
                        'h': h, 'dist': np.sqrt(dist_sq),
                    })

        info = {
            'n_constraints': len(C_rows),
            'modified': False,
            'modification_norm': 0.0,
            'pairs': pairs_info,
        }

        # 无活跃约束 → 直接返回标称控制
        if not C_rows:
            return u_nom.copy(), info

        # 求解 QP
        C = np.array(C_rows)
        b = np.array(b_vals)
        u_safe_flat = self._solve_qp(u_nom.flatten(), C, b)

        u_safe = u_safe_flat.reshape(n_f, dim)
        mod_norm = float(np.linalg.norm(u_safe - u_nom))
        info['modified'] = mod_norm > 1e-8
        info['modification_norm'] = mod_norm

        return u_safe, info

    def _solve_qp(self, u_nom_flat, C, b):
        """
        求解 CBF-QP：
          min  0.5 ||u - u_nom||²
          s.t. C u ≥ b

        Uses scipy SLSQP.
        """
        # 为每个约束生成闭包（用 i=i 绑定循环变量）
        constraints = [{
            'type': 'ineq',
            'fun': lambda u, i=i: float(C[i] @ u - b[i]),
            'jac': lambda u, i=i: C[i],
        } for i in range(len(b))]

        result = minimize(
            fun=lambda u: 0.5 * np.sum((u - u_nom_flat) ** 2),
            x0=u_nom_flat.copy(),
            jac=lambda u: u - u_nom_flat,
            constraints=constraints,
            method='SLSQP',
            options={'ftol': 1e-12, 'maxiter': 200},
        )

        if not result.success:
            return u_nom_flat  # 求解失败时回退到标称控制
        return result.x

    # ----------------------------------------------------------------
    # 距离分析工具
    # ----------------------------------------------------------------

    @staticmethod
    def min_distance(positions):
        """计算所有智能体之间的最小距离。"""
        n = positions.shape[0]
        min_dist = float('inf')
        min_pair = (-1, -1)
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(positions[i] - positions[j]))
                if d < min_dist:
                    min_dist = d
                    min_pair = (i, j)
        return min_dist, min_pair

    @staticmethod
    def all_min_distances_over_time(positions_history):
        """沿时间轴计算每时刻的最小智能体间距。"""
        n_steps = positions_history.shape[0]
        min_dists = np.zeros(n_steps)
        for idx in range(n_steps):
            md, _ = CBFSafetyFilter.min_distance(positions_history[idx])
            min_dists[idx] = md
        return min_dists

    @staticmethod
    def pairwise_distance_matrix(positions):
        """计算成对距离矩阵 D[i,j] = ||p_i - p_j||。"""
        n = positions.shape[0]
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = float(np.linalg.norm(
                    positions[i] - positions[j]
                ))
        return D
