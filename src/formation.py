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
        return factor * np.eye(d)
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


def shear_matrix_3d(sxy=0, sxz=0, syz=0):
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
