"""
stress_matrix.py - 仿射编队控制(AFC)的应力矩阵计算模块

应力矩阵 Ω 是 AFC 的核心数学对象，必须满足以下性质：
1. 对称性：Ω = Ω^T（无向通信图）
2. 零行和：Ω @ 1 = 0（应力平衡条件）
3. 零空间条件：Ω @ r_k = 0, 对每个坐标维度 k（编队形状不变性）
4. 稀疏性：ω_ij = 0 若 (i,j) ∉ E（通信图约束）
5. 稳定性：Ω_ff（follower子矩阵）正定（确保收敛）

数学推导：
设 n 个智能体在 d 维空间中的标称位置为 r_i ∈ R^d, i=1,...,n
通信图 G=(V,E) 中，边 (i,j) 对应应力权重 ω_ij
对角元素：ω_ii = -Σ_{j≠i} ω_ij

约束 Ω r_k = 0 等价于：
对每个节点 i 和每个坐标维度 k:
  Σ_{j∈N(i)} ω_ij (r_j^k - r_i^k) = 0

这构成关于边权重 w = [ω_{e_1},...,ω_{e_m}]^T 的线性方程组 C w = 0
其中 C 为约束矩阵，大小为 (n*d) × m
"""

import numpy as np
from scipy.linalg import null_space as scipy_null_space


def get_edges(adj_matrix):
    """
    从邻接矩阵提取无向边列表 (i < j)。

    Parameters
    ----------
    adj_matrix : ndarray (n, n)
        对称邻接矩阵，A[i,j] > 0 表示 (i,j) 之间有一条边

    Returns
    -------
    edges : list of (int, int)
        无向边列表，每条边 (i,j) 满足 i < j
    """
    n = adj_matrix.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i, j] > 0:
                edges.append((i, j))
    return edges


def build_constraint_matrix(positions, edges):
    """
    构建约束矩阵 C，使得 C @ w = 0 表示合法的应力权重。

    对每个节点 i 和维度 k：
        Σ_{j∈N(i)} w_{ij} (r_j^k - r_i^k) = 0

    Parameters
    ----------
    positions : ndarray (n, d)
        标称编队位置
    edges : list of (int, int)
        无向边列表

    Returns
    -------
    C : ndarray (n*d, m)
        约束矩阵，m = len(edges)
    """
    n, d = positions.shape
    m = len(edges)
    C = np.zeros((n * d, m))

    for e_idx, (i, j) in enumerate(edges):
        for k in range(d):
            # 节点 i 对应行：i*d + k
            C[i * d + k, e_idx] = positions[j, k] - positions[i, k]
            # 节点 j 对应行：j*d + k
            C[j * d + k, e_idx] = positions[i, k] - positions[j, k]

    return C


def weights_to_stress_matrix(weights, edges, n):
    """
    将边权重向量转换为 n×n 应力矩阵。

    Parameters
    ----------
    weights : ndarray (m,)
        边权重向量
    edges : list of (int, int)
        无向边列表
    n : int
        智能体总数

    Returns
    -------
    Omega : ndarray (n, n)
        应力矩阵，满足 Ω = Ω^T 且行和为零
    """
    Omega = np.zeros((n, n))
    for w_val, (i, j) in zip(weights, edges):
        Omega[i, j] = w_val
        Omega[j, i] = w_val
    # 对角元素：使行和为零
    for i in range(n):
        Omega[i, i] = -np.sum(Omega[i, :])
    return Omega


def compute_stress_matrix(positions, adj_matrix, leader_indices, method='optimize'):
    """
    计算仿射编队控制的应力矩阵。

    算法步骤：
    1. 提取通信图的边集 E
    2. 构建约束矩阵 C，满足 C w = 0
    3. 求 C 的零空间，得到可行权重的参数化：w = N c
    4. 在零空间中优化系数 c，使 Ω_ff 正定且最小特征值最大化

    Parameters
    ----------
    positions : ndarray (n, d)
        标称编队位置
    adj_matrix : ndarray (n, n)
        邻接矩阵
    leader_indices : list of int
        leader 智能体的索引
    method : str
        'optimize' - 使用 cvxpy SDP（推荐，数学最优）
        'random'   - 随机搜索（回退方案）

    Returns
    -------
    Omega : ndarray (n, n)
        应力矩阵
    info : dict
        计算信息，包含零空间维数、最小特征值等
    """
    n, d = positions.shape
    follower_indices = sorted(set(range(n)) - set(leader_indices))
    edges = get_edges(adj_matrix)
    m = len(edges)

    # Step 1 & 2: 构建约束矩阵
    C = build_constraint_matrix(positions, edges)

    # Step 3: 求零空间
    N = scipy_null_space(C)
    null_dim = N.shape[1]

    if null_dim == 0:
        raise ValueError(
            f"不存在可行的应力矩阵。当前图有 {m} 条边，"
            f"对 {n} 个智能体在 {d}D 空间中的编队可能不够。\n"
            f"建议：增加通信图的边数。"
        )

    # Step 4: 优化求解
    if method == 'optimize':
        Omega, info = _solve_sdp(N, edges, n, follower_indices, null_dim)
    else:
        Omega, info = _solve_random(N, edges, n, follower_indices, null_dim)

    info['null_dim'] = null_dim
    info['n_edges'] = m
    info['n_agents'] = n
    info['n_leaders'] = len(leader_indices)
    info['n_followers'] = len(follower_indices)
    return Omega, info


def _solve_sdp(N, edges, n, follower_indices, null_dim):
    """
    使用半定规划(SDP)求解最优应力矩阵。

    优化问题：
        max  t
        s.t. Σ_k c_k F_k ≥ t I   (Ω_ff 正定约束)
             ||c|| ≤ 1             (归一化)
             t ≥ 0

    其中 F_k 是第 k 个零空间基向量对应的 Ω_ff 子矩阵。
    """
    try:
        import cvxpy as cp
    except ImportError:
        print("[INFO] cvxpy 未安装，回退到随机搜索方法")
        return _solve_random(N, edges, n, follower_indices, null_dim)

    n_f = len(follower_indices)

    # 预计算每个零空间基向量对应的 Ω_ff
    F_list = []
    for k in range(null_dim):
        Omega_k = weights_to_stress_matrix(N[:, k], edges, n)
        F_k = Omega_k[np.ix_(follower_indices, follower_indices)]
        F_list.append(F_k)

    # SDP 变量
    c = cp.Variable(null_dim)
    t = cp.Variable()

    # Ω_ff(c) = Σ_k c_k F_k
    Omega_ff = sum(c[k] * F_list[k] for k in range(null_dim))

    constraints = [
        Omega_ff >> t * np.eye(n_f),   # 半正定约束
        cp.norm(c) <= 1.0,              # 归一化
        t >= 0,                         # 非负
    ]

    prob = cp.Problem(cp.Maximize(t), constraints)

    try:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=10000)
    except Exception as e:
        print(f"[WARN] SDP 求解失败: {e}，回退到随机搜索")
        return _solve_random(N, edges, n, follower_indices, null_dim)

    if prob.status in ['optimal', 'optimal_inaccurate'] and c.value is not None:
        weights = N @ c.value
        Omega = weights_to_stress_matrix(weights, edges, n)
        Omega_ff = Omega[np.ix_(follower_indices, follower_indices)]
        min_eig = float(np.min(np.linalg.eigvalsh(Omega_ff)))

        if min_eig > 1e-8:
            return Omega, {'method': 'SDP', 'min_eig_ff': min_eig,
                           'sdp_status': prob.status, 'sdp_optimal_t': float(prob.value)}

    print(f"[WARN] SDP 未找到正定解 (status={prob.status})，回退到随机搜索")
    return _solve_random(N, edges, n, follower_indices, null_dim)


def _solve_random(N, edges, n, follower_indices, null_dim, n_trials=100000):
    """
    随机搜索法求解应力矩阵。

    在零空间中随机采样系数向量 c，检查 Ω_ff 的正定性，
    保留使 λ_min(Ω_ff) 最大的解。
    """
    best_omega = None
    best_min_eig = -np.inf

    for _ in range(n_trials):
        c = np.random.randn(null_dim)
        c /= np.linalg.norm(c)

        weights = N @ c
        Omega = weights_to_stress_matrix(weights, edges, n)
        Omega_ff = Omega[np.ix_(follower_indices, follower_indices)]

        min_eig = np.min(np.linalg.eigvalsh(Omega_ff))

        if min_eig > best_min_eig:
            best_min_eig = min_eig
            best_omega = Omega.copy()

    if best_min_eig <= 0:
        raise ValueError(
            f"在 {n_trials} 次随机搜索中未找到正定 Ω_ff。\n"
            f"最佳最小特征值: {best_min_eig:.6e}\n"
            f"建议：\n"
            f"  1) 增加通信图的连通性\n"
            f"  2) 调整 leader 选取\n"
            f"  3) 增加搜索次数"
        )

    return best_omega, {'method': 'random', 'min_eig_ff': best_min_eig,
                        'n_trials': n_trials}


# ============================================================
# 稀疏应力矩阵设计（结合 Crazyflie 通信约束）
# ============================================================

def compute_sparse_stress_matrix(positions, leader_indices,
                                  comm_range=10.0, max_degree=6,
                                  convergence_ratio=0.5,
                                  prune_threshold=0.01):
    """
    结合 Crazyflie 通信约束的稀疏应力矩阵设计。

    三阶段优化流程：
      Stage 1 — 密集基准：在候选边集上求解标准 SDP，获取 λ_min 基准值 t_max
      Stage 2 — 稀疏优化：最小化距离加权 ℓ1 范数，同时保证 λ_min ≥ β·t_max
      Stage 3 — 后处理：裁剪近零边权、执行度约束、在稀疏图上重新优化

    Crazyflie 2.1 通信约束建模：
      - 通信范围：nRF51822 P2P 有效距离 comm_range (m)
      - 度数上限：单机受信道时隙限制，最多维护 max_degree 个邻居
      - 距离加权：优先保留短距离链路（SNR 更高、延迟更低）

    Parameters
    ----------
    positions : ndarray (n, d)
        标称编队位置
    leader_indices : list of int
        leader 索引
    comm_range : float
        P2P 最大通信距离 (m)
    max_degree : int
        每个节点的最大邻居数
    convergence_ratio : float
        相对于密集图的最小收敛速率比例 (0 < β ≤ 1)
    prune_threshold : float
        边权重裁剪阈值（相对于最大权重的比例）

    Returns
    -------
    Omega : ndarray (n, n)
        稀疏应力矩阵
    info : dict
        设计信息
    """
    n, d = positions.shape
    follower_indices = sorted(set(range(n)) - set(leader_indices))

    # === Stage 0: 通信范围筛选 ===
    candidate_edges = []
    edge_distances = []
    for i in range(n):
        for j in range(i + 1, n):
            dist_ij = float(np.linalg.norm(positions[i] - positions[j]))
            if dist_ij <= comm_range:
                candidate_edges.append((i, j))
                edge_distances.append(dist_ij)

    m_cand = len(candidate_edges)
    m_complete = n * (n - 1) // 2
    edge_distances = np.array(edge_distances)

    adj_cand = np.zeros((n, n), dtype=int)
    for i, j in candidate_edges:
        adj_cand[i, j] = 1
        adj_cand[j, i] = 1

    print(f"  通信范围 {comm_range}m → 候选边 {m_cand}/{m_complete}")

    # === Stage 1: 密集基准 ===
    print("  [Stage 1] 密集基准求解...")
    Omega_dense, dense_info = compute_stress_matrix(
        positions, adj_cand, leader_indices, method='optimize'
    )
    t_max = dense_info['min_eig_ff']
    t_target = convergence_ratio * t_max
    print(f"    λ_min(Ω_ff) 基准值: {t_max:.6f}")
    print(f"    稀疏目标下界 (×{convergence_ratio}): {t_target:.6f}")

    # === Stage 2: 距离加权 ℓ1 稀疏 SDP ===
    print("  [Stage 2] 距离加权 ℓ1 稀疏优化...")
    w_sparse = _solve_min_edges_sdp(
        positions, candidate_edges, edge_distances,
        n, follower_indices, t_target
    )

    # === Stage 3: 贪心拓扑构建 ===
    # 用 ℓ1 SDP 的边权重作为重要性排序，贪心地添加最重要的边，
    # 同时尊重度约束。若度约束过紧则逐步放宽。
    print("  [Stage 3] 贪心拓扑构建 (度约束)...")

    importance = np.abs(w_sparse)
    sorted_idx = np.argsort(importance)[::-1]  # 重要性降序

    # 贪心添加：优先加入最重要的边，跳过会违反度约束的边
    sparse_edges = []
    degrees = np.zeros(n, dtype=int)
    skipped = []  # 因度约束跳过的边 (按重要性排序)

    for idx in sorted_idx:
        edge = candidate_edges[idx]
        i, j = edge
        if degrees[i] < max_degree and degrees[j] < max_degree:
            sparse_edges.append(edge)
            degrees[i] += 1
            degrees[j] += 1
        else:
            skipped.append(edge)

    n_sparse = len(sparse_edges)
    print(f"    度约束 (max={max_degree}) → {n_sparse} 条边")
    print(f"    度数: min={int(degrees.min())}, max={int(degrees.max())}, "
          f"mean={degrees.mean():.1f}")

    # 构建邻接矩阵并验证可行性
    sparse_adj = np.zeros((n, n), dtype=int)
    for i, j in sparse_edges:
        sparse_adj[i, j] = 1
        sparse_adj[j, i] = 1

    # === Stage 4: 稀疏图上重新优化 ===
    print("  [Stage 4] 稀疏图重新优化 λ_min(Ω_ff)...")
    Omega, reopt_info = None, None

    try:
        Omega, reopt_info = compute_stress_matrix(
            positions, sparse_adj, leader_indices, method='optimize'
        )
        # 检查收敛性是否足够
        if reopt_info['min_eig_ff'] < t_target * 0.1:
            print(f"    λ_min={reopt_info['min_eig_ff']:.6f} 过低，需要更多边")
            Omega = None
    except ValueError:
        print("    约束不可行，需要更多边")

    # 若不可行，逐步添加被跳过的边（放宽度约束）
    if Omega is None:
        print("    逐步添加边以满足收敛性...")
        for edge in skipped:
            sparse_edges.append(edge)
            i, j = edge
            degrees[i] += 1
            degrees[j] += 1
            sparse_adj[i, j] = 1
            sparse_adj[j, i] = 1
            try:
                Omega, reopt_info = compute_stress_matrix(
                    positions, sparse_adj, leader_indices, method='optimize'
                )
                if reopt_info['min_eig_ff'] >= t_target * 0.1:
                    n_sparse = len(sparse_edges)
                    print(f"    {n_sparse} 条边时可行: "
                          f"λ_min={reopt_info['min_eig_ff']:.6f}")
                    break
                else:
                    Omega = None
            except ValueError:
                continue

    # 最终回退
    if Omega is None:
        print("  [WARN] 无法构建可行稀疏图，回退到密集图")
        Omega, reopt_info = Omega_dense, dense_info
        sparse_adj = adj_cand
        n_sparse = m_cand
        sparse_edges = candidate_edges

    n_sparse = len(sparse_edges)
    print(f"    最终稀疏 λ_min(Ω_ff): {reopt_info['min_eig_ff']:.6f}")
    print(f"    最终边数: {n_sparse} (削减率 "
          f"{(1.0 - n_sparse / m_cand) * 100:.1f}%)")

    # 度分布统计
    degrees = np.sum(sparse_adj, axis=1).astype(int)

    info = reopt_info.copy()
    info['t_max_dense'] = t_max
    info['n_edges_complete'] = m_complete
    info['n_edges_candidate'] = m_cand
    info['n_edges_sparse'] = n_sparse
    info['edge_reduction_pct'] = (1.0 - n_sparse / m_cand) * 100
    info['sparse_adj'] = sparse_adj
    info['dense_adj'] = adj_cand
    info['sparse_edges'] = sparse_edges
    info['degree_min'] = int(degrees.min())
    info['degree_max'] = int(degrees.max())
    info['degree_mean'] = float(degrees.mean())
    info['comm_range'] = comm_range
    info['max_degree_constraint'] = max_degree

    return Omega, info


def _solve_min_edges_sdp(positions, edges, distances, n, follower_indices, t_target):
    """
    距离加权 ℓ1 最小化 SDP。

    优化问题：
        min   Σ_e (d_e / d̄) · |w_e|     — 距离加权稀疏惩罚
        s.t.  C · w = 0                   — 应力平衡约束
              Ω_ff(w) ≽ t_target · I      — 收敛性保证
              ‖w‖₂ ≤ 1                    — 归一化

    距离加权的物理意义：
      偏好保留短距离通信链路，因为 Crazyflie P2P 信号强度
      随距离平方衰减，短链路有更好的 SNR 和更低的延迟。
    """
    try:
        import cvxpy as cp
    except ImportError:
        raise RuntimeError("cvxpy 是稀疏应力矩阵设计的必需依赖")

    n_f = len(follower_indices)
    m = len(edges)

    # 约束矩阵
    C = build_constraint_matrix(positions, edges)

    # 预计算每条边对 Ω_ff 的贡献矩阵 F_e
    F_list = []
    for e_idx in range(m):
        unit_w = np.zeros(m)
        unit_w[e_idx] = 1.0
        Omega_e = weights_to_stress_matrix(unit_w, edges, n)
        F_list.append(Omega_e[np.ix_(follower_indices, follower_indices)])

    # 距离权重归一化 (均值=1)
    d_weights = distances / np.mean(distances)

    # cvxpy 模型
    w = cp.Variable(m)
    Omega_ff_expr = sum(w[e] * F_list[e] for e in range(m))

    constraints = [
        C @ w == 0,                                   # 应力平衡
        Omega_ff_expr >> t_target * np.eye(n_f),      # 收敛性保证
        cp.norm(w, 2) <= 1.0,                         # 归一化
    ]

    # 目标：最小化距离加权 ℓ1 (鼓励稀疏 + 偏好短距离)
    objective = cp.Minimize(d_weights @ cp.abs(w))

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=20000)
    except Exception as e:
        print(f"    [SDP 异常] {e}")

    if prob.status in ['optimal', 'optimal_inaccurate'] and w.value is not None:
        print(f"    SDP 状态: {prob.status}, 目标值: {prob.value:.6f}")
        return w.value

    # 降低目标重试
    print(f"    t_target={t_target:.6f} 不可行, 降低至 50% 重试...")
    constraints[1] = Omega_ff_expr >> (t_target * 0.5) * np.eye(n_f)
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=30000)
    except Exception:
        pass

    if prob.status in ['optimal', 'optimal_inaccurate'] and w.value is not None:
        print(f"    SDP 状态 (降低后): {prob.status}")
        return w.value

    raise ValueError(f"稀疏 SDP 不可行 (status={prob.status})")


def _enforce_degree_constraint(edges, weights, n, max_degree):
    """
    后处理：按度约束移除最弱的边。

    当某节点的度数超过 max_degree 时，移除该节点相连的
    权重绝对值最小的边，直到所有节点满足约束。

    Parameters
    ----------
    edges : list of (int, int)
    weights : ndarray
    n : int
    max_degree : int

    Returns
    -------
    pruned_edges : list of (int, int)
    n_removed : int
    """
    edges = list(edges)
    weights = np.array(weights, dtype=float)

    degrees = np.zeros(n, dtype=int)
    for i, j in edges:
        degrees[i] += 1
        degrees[j] += 1

    n_removed = 0
    while np.max(degrees) > max_degree:
        over_nodes = np.where(degrees > max_degree)[0]
        node = over_nodes[np.argmax(degrees[over_nodes])]

        # 找该节点关联的所有边，按权重绝对值升序
        node_edge_info = []
        for idx, (i, j) in enumerate(edges):
            if i == node or j == node:
                node_edge_info.append((idx, abs(weights[idx])))
        node_edge_info.sort(key=lambda x: x[1])

        # 移除最弱的边
        idx_remove = node_edge_info[0][0]
        i, j = edges[idx_remove]
        degrees[i] -= 1
        degrees[j] -= 1
        edges.pop(idx_remove)
        weights = np.delete(weights, idx_remove)
        n_removed += 1

    return edges, n_removed


def validate_stress_matrix(Omega, positions, leader_indices, tol=1e-6):
    """
    验证应力矩阵的所有性质。

    Parameters
    ----------
    Omega : ndarray (n, n)
        应力矩阵
    positions : ndarray (n, d)
        标称编队位置
    leader_indices : list of int
        leader 索引
    tol : float
        数值容差

    Returns
    -------
    results : dict
        各项验证结果
    """
    n = Omega.shape[0]
    d = positions.shape[1]
    follower_indices = sorted(set(range(n)) - set(leader_indices))

    results = {}

    # 1. 对称性
    sym_error = np.max(np.abs(Omega - Omega.T))
    results['对称性'] = sym_error < tol
    results['对称性误差'] = float(sym_error)

    # 2. 零行和
    row_sums = Omega @ np.ones(n)
    results['零行和'] = np.allclose(row_sums, 0, atol=tol)
    results['最大行和误差'] = float(np.max(np.abs(row_sums)))

    # 3. 零空间条件 Ω r_k = 0
    null_errors = []
    for k in range(d):
        err = float(np.linalg.norm(Omega @ positions[:, k]))
        null_errors.append(err)
    results['零空间误差'] = null_errors
    results['零空间条件'] = all(e < tol for e in null_errors)

    # 4. Ω_ff 正定性
    Omega_ff = Omega[np.ix_(follower_indices, follower_indices)]
    eigvals_ff = np.linalg.eigvalsh(Omega_ff)
    results['Ω_ff特征值'] = eigvals_ff.tolist()
    results['Ω_ff最小特征值'] = float(eigvals_ff[0])
    results['Ω_ff正定'] = bool(eigvals_ff[0] > 0)

    # 5. Ω 的整体特征值与秩
    eigvals_omega = np.linalg.eigvalsh(Omega)
    results['Ω特征值'] = eigvals_omega.tolist()
    results['Ω的秩'] = int(np.sum(np.abs(eigvals_omega) > tol))

    # 总体判断
    results['全部通过'] = all([
        results['对称性'], results['零行和'],
        results['零空间条件'], results['Ω_ff正定']
    ])

    return results


def print_validation(results):
    """打印应力矩阵验证结果。"""
    print("=" * 50)
    print("应力矩阵验证结果")
    print("=" * 50)
    for key, val in results.items():
        if key == 'Ω特征值' or key == 'Ω_ff特征值':
            print(f"  {key}: [{', '.join(f'{v:.4f}' for v in val)}]")
        elif key == '零空间误差':
            print(f"  {key}: [{', '.join(f'{v:.2e}' for v in val)}]")
        elif isinstance(val, float):
            print(f"  {key}: {val:.6e}")
        else:
            print(f"  {key}: {val}")
    status = "✓ 全部通过" if results['全部通过'] else "✗ 存在未通过项"
    print(f"\n  总体状态: {status}")
    print("=" * 50)


# ============================================================
# 层级重组 (Hierarchical Reorganization) 应力矩阵计算
# ============================================================

def compute_power_centric_stress_matrix(positions, leader_indices):
    """
    为 Power-Centric 拓扑计算应力矩阵。

    Power-Centric 拓扑（Theorem IV.3 of Li & Dong 2024）：
      - Leader 间全连接
      - 每个 Follower 连接所有 Leader
      - Follower 间无连接

    此拓扑保证 Ω_ff 为对角矩阵，det(Ω_ff) > 0（Proposition 1）。
    因此非常适合层级重组场景——切换后立即可用。

    Parameters
    ----------
    positions : ndarray (n, d)
        标称编队位置
    leader_indices : list of int
        leader 索引

    Returns
    -------
    Omega : ndarray (n, n)
        应力矩阵
    info : dict
        计算信息
    """
    n, d = positions.shape
    leader_set = set(leader_indices)
    follower_indices = sorted(set(range(n)) - leader_set)

    # 构建 power-centric 邻接矩阵
    adj = np.zeros((n, n), dtype=int)
    for i in leader_indices:
        for j in leader_indices:
            if i != j:
                adj[i, j] = 1
    for f in follower_indices:
        for l in leader_indices:
            adj[f, l] = 1
            adj[l, f] = 1

    # 使用标准方法计算应力矩阵
    Omega, base_info = compute_stress_matrix(positions, adj, leader_indices,
                                             method='optimize')

    # 附加信息
    Omega_ff = Omega[np.ix_(follower_indices, follower_indices)]
    min_eig = float(np.min(np.linalg.eigvalsh(Omega_ff)))
    n_edges = int(np.sum(adj)) // 2

    info = {
        **base_info,
        'topology': 'power_centric',
        'adj_matrix': adj,
        'min_eig_ff': min_eig,
        'n_edges': n_edges,
        'leader_indices': list(leader_indices),
        'follower_indices': follower_indices,
        'omega_ff_diagonal': bool(np.allclose(
            Omega_ff, np.diag(np.diag(Omega_ff)), atol=1e-8
        )),
    }

    return Omega, info
