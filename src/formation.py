"""
formation.py - 编队构型定义与仿射变换工具

提供以下功能：
1. 预定义编队构型（双层正五边形、六边形等）
2. 通信图拓扑生成
3. 仿射变换工具（缩放、旋转、平移）

仿射变换 T: R^d → R^d 定义为：
  T(p) = A p + b
  其中 A ∈ R^{d×d} 为线性变换矩阵，b ∈ R^d 为平移向量

仿射变换的类型：
  - 平移：A = I, b ≠ 0
  - 缩放：A = sI, b = 0
  - 旋转：A = R(θ), b = 0
  - 剪切：A 为上/下三角矩阵
  - 一般仿射：任意可逆 A 和 b
"""

import numpy as np


# ============================================================
# Crazyflie 2.1 通信参数
# ============================================================

CRAZYFLIE_COMM = {
    'p2p_range': 10.0,           # nRF51822 P2P 有效通信距离 (m), 室内环境
    'max_neighbors': 6,          # 单机最大 P2P 邻居数 (受信道时隙限制)
    'control_freq_hz': 50,       # 编队控制更新频率 (Hz)
    'p2p_slot_ms': 2.0,          # 单次 P2P 交换时隙 (ms): TX + ACK + guard
    'radio_chip': 'nRF51822',    # 无线芯片型号
    'radio_data_rate_mbps': 2.0, # 空中数据速率 (Mbps)
    'packet_payload_bytes': 32,  # 单包有效载荷 (bytes)
    'pos_data_bytes': 12,        # 位置数据量: 3 × float32 (bytes)
    'max_velocity': 1.0,         # 编队飞行最大速度 (m/s), 保守室内限制
    'max_acceleration': 5.0,     # 最大加速度 (m/s²), 用于二阶模型
    # 带宽推导:
    #   控制周期 T_c = 1/50 Hz = 20ms
    #   每条链路占用 ~2ms 时隙 (发送 + 应答 + 间隔)
    #   每周期可用时隙 = 20ms / 2ms = 10
    #   考虑信道竞争和重传冗余 (×0.6) → 有效邻居数 ≈ 6
}


# ============================================================
# 编队构型定义
# ============================================================

def double_pentagon(radius=1.0, height=1.0):
    """
    10 智能体双层正五边形编队（3D）。

    结构：
      底层 (z=0)：5 个智能体排列为正五边形
      顶层 (z=height)：5 个智能体排列为正五边形

    智能体编号：
      0-4: 底层（agent 0 在 x 轴正方向）
      5-9: 顶层（agent 5 在 x 轴正方向）

    Leader 选取: agents {0, 1, 2, 5}
      确保 4 个 leader 不共面（3D 仿射需要 d+1=4 个 leader）

    Returns
    -------
    positions : ndarray (10, 3)
    leader_indices : list of int
    adj : ndarray (10, 10)
    """
    n = 10
    positions = np.zeros((n, 3))

    # 底层正五边形
    for k in range(5):
        angle = 2 * np.pi * k / 5
        positions[k] = [radius * np.cos(angle), radius * np.sin(angle), 0.0]

    # 顶层正五边形——旋转 π/5 打破对称性（反棱柱构型）
    for k in range(5):
        angle = 2 * np.pi * k / 5 + np.pi / 5
        positions[k + 5] = [radius * np.cos(angle), radius * np.sin(angle), height]

    # 4 个 leader，不共面
    leader_indices = [0, 1, 2, 5]

    # 通信图：完全图（保证应力矩阵可解性）
    adj = np.ones((n, n), dtype=int) - np.eye(n, dtype=int)

    return positions, leader_indices, adj


def hexagon_2d(radius=1.0):
    """
    6 智能体正六边形编队（2D）。

    Leader 选取: agents {0, 1, 2}（2D 仿射需要 d+1=3 个 leader）

    Returns
    -------
    positions : ndarray (6, 2)
    leader_indices : list of int
    adj : ndarray (6, 6)
    """
    n = 6
    positions = np.zeros((n, 2))

    for k in range(6):
        angle = 2 * np.pi * k / 6
        positions[k] = [radius * np.cos(angle), radius * np.sin(angle)]

    leader_indices = [0, 1, 2]

    # 完全图（确保应力矩阵存在）
    adj = np.ones((n, n), dtype=int) - np.eye(n, dtype=int)

    return positions, leader_indices, adj


def grid_3d(nx=2, ny=2, nz=2, spacing=1.0):
    """
    3D 网格编队。

    Parameters
    ----------
    nx, ny, nz : int
        各方向的智能体数量
    spacing : float
        网格间距

    Returns
    -------
    positions : ndarray (n, 3)
    leader_indices : list of int
    adj : ndarray (n, n)
    """
    n = nx * ny * nz
    positions = np.zeros((n, 3))

    idx = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                positions[idx] = [ix * spacing, iy * spacing, iz * spacing]
                idx += 1

    # 选择 4 个不共面的角点作为 leader
    leader_indices = [0, nx - 1, nx * ny - 1, nx * ny * (nz - 1)]
    # 去重并限制范围
    leader_indices = sorted(set(i for i in leader_indices if i < n))
    if len(leader_indices) < 4 and n >= 4:
        for i in range(n):
            if i not in leader_indices:
                leader_indices.append(i)
                if len(leader_indices) >= 4:
                    break
        leader_indices = sorted(leader_indices)

    # 邻接关系：连接距离 ≤ spacing * √3 的所有点对
    adj = np.zeros((n, n), dtype=int)
    threshold = spacing * np.sqrt(3) + 1e-6
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(positions[i] - positions[j]) <= threshold:
                _add_edge(adj, i, j)

    return positions, leader_indices, adj


def aerial_pyramid_10(base_half=1.4, mid_radius=0.9,
                                            base_height=0.9, mid_height=1.9,
                                            top_height=2.8, apex_height=3.7):
        """
        10 智能体空中金字塔编队（3D）。

        结构：
            - 底层：4 个角点正方形
            - 中层：4 个轴向支撑点
            - 顶层：1 个中心平台点
            - 尖顶：1 个 apex 点

        Leader 选取: agents {0, 1, 2, 9}
            使用 3 个底层角点 + 1 个尖顶，保证 4 个 leader 不共面。

        Returns
        -------
        positions : ndarray (10, 3)
        leader_indices : list[int]
        adj : ndarray (10, 10)
        """
        positions = np.array([
                [-base_half, -base_half, base_height],
                [base_half, -base_half, base_height],
                [base_half, base_half, base_height],
                [-base_half, base_half, base_height],
                [0.0, -mid_radius, mid_height],
                [mid_radius, 0.0, mid_height],
                [0.0, mid_radius, mid_height],
                [-mid_radius, 0.0, mid_height],
                [0.0, 0.0, top_height],
                [0.0, 0.0, apex_height],
        ], dtype=float)

        leader_indices = [0, 1, 2, 9]
        adj = np.ones((10, 10), dtype=int) - np.eye(10, dtype=int)
        return positions, leader_indices, adj


def custom_formation(positions, leader_indices, adj_matrix=None, connect_radius=None):
    """
    自定义编队构型。

    Parameters
    ----------
    positions : ndarray (n, d)
        智能体位置
    leader_indices : list of int
        leader 索引
    adj_matrix : ndarray (n, n) or None
        邻接矩阵。若为 None，根据 connect_radius 生成
    connect_radius : float or None
        连接半径（仅当 adj_matrix 为 None 时使用）

    Returns
    -------
    positions, leader_indices, adj_matrix
    """
    n = positions.shape[0]

    if adj_matrix is None:
        if connect_radius is None:
            # 使用完全图
            adj_matrix = np.ones((n, n), dtype=int) - np.eye(n, dtype=int)
        else:
            adj_matrix = np.zeros((n, n), dtype=int)
            for i in range(n):
                for j in range(i + 1, n):
                    if np.linalg.norm(positions[i] - positions[j]) <= connect_radius:
                        _add_edge(adj_matrix, i, j)

    return positions, leader_indices, adj_matrix


def _add_edge(adj, i, j):
    """辅助函数：在邻接矩阵中添加无向边。"""
    adj[i, j] = 1
    adj[j, i] = 1


# ============================================================
# 仿射变换工具
# ============================================================

def affine_transform(positions, A=None, b=None):
    """
    对所有位置施加仿射变换 p' = A p + b。

    Parameters
    ----------
    positions : ndarray (n, d)
    A : ndarray (d, d) or None
        线性变换矩阵
    b : ndarray (d,) or None
        平移向量

    Returns
    -------
    new_positions : ndarray (n, d)
    """
    result = positions.copy()

    if A is not None:
        result = (A @ result.T).T

    if b is not None:
        result = result + b[np.newaxis, :]

    return result


def scale_matrix(d, factor):
    """
    创建缩放变换矩阵。

    Parameters
    ----------
    d : int
        空间维数
    factor : float or array-like
        缩放因子（标量为均匀缩放，向量为各维度独立缩放）
    """
    if np.isscalar(factor):
        return float(factor) * np.eye(d)  # type: ignore[arg-type]
    return np.diag(np.asarray(factor))


def rotation_matrix_z(theta):
    """
    绕 z 轴旋转矩阵（3D）。

    Parameters
    ----------
    theta : float
        旋转角度（弧度）

    Returns
    -------
    R : ndarray (3, 3)
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])


def rotation_matrix_2d(theta):
    """
    2D 旋转矩阵。

    Parameters
    ----------
    theta : float
        旋转角度（弧度）

    Returns
    -------
    R : ndarray (2, 2)
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s],
        [s,  c]
    ])


def rotation_matrix_axis(axis, theta):
    """
    绕任意轴旋转矩阵（3D，Rodrigues 公式）。

    Parameters
    ----------
    axis : ndarray (3,)
        旋转轴（将被归一化）
    theta : float
        旋转角度（弧度）

    Returns
    -------
    R : ndarray (3, 3)
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def shear_matrix_3d(sxy=0.0, sxz=0.0, syz=0.0):
    """
    3D 剪切变换矩阵。

    Returns
    -------
    S : ndarray (3, 3)
    """
    return np.array([
        [1,  sxy, sxz],
        [0,  1,   syz],
        [0,  0,   1]
    ])


# ============================================================
# Leader 轨迹生成
# ============================================================

def smoothstep(t, t_start, t_end):
    """
    平滑插值函数 (Hermite 平滑阶跃)。

    在 [t_start, t_end] 内从 0 平滑过渡到 1,
    其一阶导数在端点处为 0（无加速度突变）。

    Returns
    -------
    alpha : float in [0, 1]
    """
    if t <= t_start:
        return 0.0
    elif t >= t_end:
        return 1.0
    else:
        x = (t - t_start) / (t_end - t_start)
        return 3 * x**2 - 2 * x**3


def create_leader_trajectory(phases):
    """
    创建多阶段 Leader 轨迹函数。

    Parameters
    ----------
    phases : list of dict
        每个阶段的定义：
        {
            't_start': float,      # 该阶段开始时间
            't_end': float,        # 该阶段结束时间
            'positions': ndarray,  # 该阶段结束时的 Leader 目标位置 (n_l, d)
        }
        第一个阶段之前使用第一个 phase 的 'start_positions'

    Returns
    -------
    trajectory : callable
        trajectory(t) → ndarray (n_l, d)
    """
    def trajectory(t):
        # 在第一个阶段之前
        if t <= phases[0]['t_start']:
            return phases[0].get('start_positions', phases[0]['positions']).copy()

        # 逐阶段检查
        for i, phase in enumerate(phases):
            if t <= phase['t_end']:
                if i == 0:
                    p_start = phase.get('start_positions', phase['positions'])
                else:
                    p_start = phases[i - 1]['positions']
                p_end = phase['positions']
                alpha = smoothstep(t, phase['t_start'], phase['t_end'])
                return (1 - alpha) * p_start + alpha * p_end

        # 超过所有阶段
        return phases[-1]['positions'].copy()

    return trajectory


def graph_info(adj_matrix):
    """打印通信图的基本信息。"""
    n = adj_matrix.shape[0]
    n_edges = np.sum(adj_matrix) // 2
    degrees = np.sum(adj_matrix, axis=1)

    print(f"  智能体数量: {n}")
    print(f"  边数: {n_edges}")
    print(f"  度数分布: min={int(degrees.min())}, max={int(degrees.max())}, "
          f"mean={degrees.mean():.1f}")
    print(f"  邻接矩阵对称: {np.allclose(adj_matrix, adj_matrix.T)}")


# ============================================================
# 层级重组 (Hierarchical Reorganization) 工具
# ============================================================
# 参考: Li & Dong (2024) "A Flexible and Resilient Formation Approach
#       based on Hierarchical Reorganization", arXiv:2406.11219

def check_affine_span(positions, leader_indices, d=None):
    """
    检查 leader 集合是否仿射张成 R^d。

    仿射张成条件（Theorem IV.1 of Li & Dong 2024）：
      rank([r_l^T; 1^T]) = d + 1
    即 leader 位置加上常数行后的矩阵应满秩。

    在 3D 空间中：需要至少 4 个不共面的 leader。
    在 2D 空间中：需要至少 3 个不共线的 leader。

    Parameters
    ----------
    positions : ndarray (n, d)
        所有智能体的标称位置
    leader_indices : list of int
        候选 leader 索引
    d : int or None
        空间维数（None 则自动推断）

    Returns
    -------
    result : dict
        'valid': bool - 是否满足仿射张成条件
        'rank': int   - 实际秩
        'required_rank': int - 所需秩 (d+1)
        'condition_number': float - 条件数（越小越好）
    """
    if d is None:
        d = positions.shape[1]
    r_l = positions[leader_indices]  # (n_l, d)
    # 构造 [r_l, 1] 矩阵
    r_bar = np.column_stack([r_l, np.ones(len(leader_indices))])  # (n_l, d+1)
    rank = np.linalg.matrix_rank(r_bar, tol=1e-10)
    # 条件数
    s = np.linalg.svd(r_bar, compute_uv=False)
    cond = s[0] / s[-1] if s[-1] > 1e-15 else float('inf')
    return {
        'valid': rank >= d + 1,
        'rank': rank,
        'required_rank': d + 1,
        'condition_number': float(cond),
    }


def build_power_centric_topology(n, leader_indices, nominal_pos=None, d=3):
    """
    构建 power-centric 拓扑（Theorem IV.3 of Li & Dong 2024）。

    规则：
      - Leader 间全连接（保持 leader 层稳定性）
      - 每个 Follower 连接所有 Leader（保证 (d+1)-rooted）
      - Follower 间无连接

    此策略保证 Ω_ff 为对角矩阵（每个 follower 仅与 leader 耦合），
    确保 det(Ω_ff) > 0。

    Parameters
    ----------
    n : int
        智能体总数
    leader_indices : list of int
        leader 索引
    nominal_pos : ndarray (n, d) or None
        标称位置（用于可选的距离约束检查）
    d : int
        空间维数

    Returns
    -------
    adj : ndarray (n, n)
        邻接矩阵
    info : dict
        拓扑信息
    """
    leader_set = set(leader_indices)
    follower_indices = sorted(set(range(n)) - leader_set)

    adj = np.zeros((n, n), dtype=int)

    # Leader 间全连接
    for i in leader_indices:
        for j in leader_indices:
            if i != j:
                adj[i, j] = 1

    # 每个 follower 连接所有 leader
    for f in follower_indices:
        for l in leader_indices:
            adj[f, l] = 1
            adj[l, f] = 1

    n_edges = int(np.sum(adj)) // 2
    degrees = np.sum(adj, axis=1)
    n_l = len(leader_indices)
    n_f = len(follower_indices)

    info = {
        'type': 'power_centric',
        'n_edges': n_edges,
        'leader_degree': n_l - 1 + n_f,  # Leader连所有其他leader + 所有follower
        'follower_degree': n_l,           # Follower只连leader
        'degree_min': int(degrees.min()),
        'degree_max': int(degrees.max()),
    }

    return adj, info


def select_leaders_for_direction(positions, direction, n_leaders=4, d=3):
    """
    根据运动方向选择最优 leader 集合（Power-Centric 策略）。

    选取策略：
      1. 计算每个智能体在 direction 方向上的投影值
      2. 选择投影值最大的 n_leaders 个作为候选
      3. 验证仿射张成条件（rank check）
      4. 若不满足，迭代替换直到满足

    Parameters
    ----------
    positions : ndarray (n, d)
        所有智能体当前位置
    direction : ndarray (d,)
        目标运动方向（将被归一化）
    n_leaders : int
        需要的 leader 数量（3D至少4，2D至少3）
    d : int
        空间维数

    Returns
    -------
    leader_indices : list of int
        选取的 leader 索引
    info : dict
        选取信息
    """
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm < 1e-12:
        raise ValueError("方向向量不能为零向量")
    direction = direction / norm

    n = len(positions)
    # 计算每个智能体在目标方向上的投影
    projections = positions @ direction  # (n,)

    # 按投影值降序排列
    sorted_indices = np.argsort(-projections)

    # 首先尝试前 n_leaders 个
    candidates = list(sorted_indices[:n_leaders])
    span_result = check_affine_span(positions, candidates, d)

    if span_result['valid']:
        return candidates, {
            'method': 'direct_projection',
            'projections': projections,
            'affine_span': span_result,
        }

    # 如果不满足仿射张成，贪心搜索：保留投影最大的 d 个，
    # 然后从剩余中找能补全仿射张成的
    best_candidates = None
    best_score = -np.inf

    # 尝试所有 C(n, n_leaders) 组合（小规模可行）
    from itertools import combinations
    for combo in combinations(range(n), n_leaders):
        combo_list = list(combo)
        sr = check_affine_span(positions, combo_list, d)
        if sr['valid']:
            # 得分 = 投影之和（越大越好）- 条件数惩罚
            score = sum(projections[i] for i in combo_list) \
                    - 0.01 * sr['condition_number']
            if score > best_score:
                best_score = score
                best_candidates = combo_list

    if best_candidates is None:
        raise ValueError(
            f"无法找到满足仿射张成条件的 {n_leaders} 个 leader 组合。"
            f"请检查编队构型是否退化（如所有点共面/共线）。"
        )

    span_result = check_affine_span(positions, best_candidates, d)
    return best_candidates, {
        'method': 'combinatorial_search',
        'projections': projections,
        'affine_span': span_result,
    }


def compute_dwell_time(gain, min_eig_ff, epsilon_0, epsilon_target):
    """
    计算层级切换所需的最小驻留时间（Dwell-time 条件）。

    一阶收敛模型下：
      ||e(t)|| ≤ ε₀ exp(-K_p λ_min t)
      令 ε₀ exp(-K_p λ_min τ) = ε_target
      → τ_dwell = ln(ε₀/ε_target) / (K_p λ_min)

    Parameters
    ----------
    gain : float
        控制增益 K_p
    min_eig_ff : float
        Ω_ff 的最小特征值 λ_min
    epsilon_0 : float
        切换瞬间的初始误差上界
    epsilon_target : float
        切换前需要达到的误差阈值

    Returns
    -------
    dwell_time : float
        最小驻留时间（秒）
    """
    if min_eig_ff <= 0 or gain <= 0:
        return float('inf')
    rate = gain * min_eig_ff
    if epsilon_0 <= epsilon_target:
        return 0.0
    return np.log(epsilon_0 / epsilon_target) / rate
