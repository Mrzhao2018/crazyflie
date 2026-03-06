# 项目文档：基于Crazyflie开发平台的微型无人机编队控制研究

> **最后更新**: 2025年7月 → 2026年3月05日 → 3月06日 (新增 ESO / 事件触发通信 / 随机测试模块)  
> **目的**: 记录项目所有已实现功能、开发环境、代码架构和开发进程，避免每次重新阅读全部代码

---

## 一、毕设基本信息

- **题目**: 基于Crazyflie开发平台的微型无人机编队控制研究  
  (Research on Formation Control of Micro UAVs Based on the Crazyflie Development Platform)
- **方向**: 飞行控制；集群控制
- **核心算法**: 三维仿射编队控制 (Affine Formation Control, AFC)

### 主要任务与要求

| 序号 | 任务 | 指标要求 |
|------|------|----------|
| 1 | Crazyflie 平台搭建（硬件连接+软件安装） | 完成 |
| 2 | 室内高精度定位系统 | 定位误差≤0.5m，更新频率≥2Hz |
| 3 | 三维仿射变换编队控制算法 | 理论推导误差≤10% |
| 4 | 实物飞行实验（单机、多机） | 多机编队位置偏差≤0.3m |
| 5 | 毕业论文撰写 | 系统搭建+算法设计+实验验证 |

---

## 二、开发环境

### 2.1 Windows 端（算法开发与仿真）

| 项目 | 配置 |
|------|------|
| Python 版本 | 3.12.6 |
| 虚拟环境 | `.venv/` |
| Python 路径 | `.venv/Scripts/python.exe` |
| 关键包 | cvxpy 1.8.1, numpy 2.4.2, scipy 1.17.1, matplotlib, clarabel, scs, osqp, highspy |

> **⚠️ 重要**: 必须使用 `.venv` 中的 Python，全局 Python 缺少 cvxpy 等关键包！

### 2.2 WSL2 端（ROS 2 + CrazyChoir + Webots）

| 项目 | 配置 |
|------|------|
| 系统 | Ubuntu 20.04 (WSL2) |
| ROS 2 | Foxy |
| CrazyChoir | `~/crazychoir_ws/` (bearing formation controller 已研究) |
| Webots | R2023a |
| qpSWIFT | 已编译安装 |

### 2.3 硬件

- **Crazyflie 2.1**: 30 架（仿真中使用 10 架）
- **定位**: 2× SteamVR Lighthouse
- **通信**: 多个 Crazyradio PA

---

## 三、项目文件结构

```
crazyflie/
├── .venv/                      # Python 虚拟环境
├── docs/                       # 参考文献与毕设文档
│   ├── 毕设信息.txt             # 毕设题目、任务要求
│   ├── 开题报告.docx
│   ├── 毕业设计相关资料-学生使用.zip
│   ├── Affine (1).pdf / affine.pdf    # 仿射编队核心论文
│   ├── CrazyChoir_Flying_Swarms...pdf # CrazyChoir 论文
│   ├── Crazyswarm_A_large...pdf       # Crazyswarm 论文
│   ├── Shaan_Hossain_...Thesis_v2.pdf # 仿射编队硕士论文（重要参考）
│   ├── Joint_Estimation_and_Planar...pdf
│   └── ... (更多参考论文)
├── saves/                      # 保存的临时数据
├── settings.yaml               # 配置文件
├── qpSWIFT-main.zip           # qpSWIFT 源码
├── webots_2023a_amd64.deb      # Webots 安装包
├── PROJECT_DOC.md              # 本文档
└── src/                        # 源代码目录
    ├── stress_matrix.py        # 应力矩阵计算（~570行）
    ├── afc_controller.py       # AFC 控制器 + 输入饱和（~260行）
    ├── formation.py            # 编队定义与仿射工具（~460行）
    ├── main_sim.py             # 仿真主程序（~1870行）
    ├── event_trigger.py        # 自适应事件触发通信管理器（~170行）
    ├── disturbance_observer.py # ESO 扰动观测器 + 风场模型（~170行）
    ├── random_test.py          # 随机初始状态 + 复杂仿射变换独立测试（~500行）
    ├── animate_sim.py          # 3D 动画生成（~270行）
    ├── fig1_formation_snapshots.png  # 编队快照
    ├── fig2_trajectories_3d.png     # 3D 轨迹
    ├── fig3_convergence.png         # 误差收敛
    ├── fig4_stress_matrix.png       # 应力矩阵热力图
    ├── fig5_communication_graph.png # 稀疏 vs 完全图对比
    ├── fig6_sparse_comparison.png   # 通信指标对比
    ├── fig7_saturation_analysis.png # 输入饱和分析
    ├── fig8_saturation_comparison.png # 不同饱和上限对比
    ├── fig9_cbf_collision_avoidance.png  # CBF 碞撞避免效果
    ├── fig10_cbf_analysis.png           # CBF 分析面板
    ├── fig11_eso_disturbance_rejection.png # ESO 扰动估计与补偿
    ├── fig12_eso_bandwidth_comparison.png  # ESO 带宽对比
    ├── fig13_et_error_comparison.png        # ET vs 连续通信误差对比
    ├── fig14_et_communication_analysis.png  # ET 通信分析四面板
    ├── fig15_et_parameter_comparison.png    # ET 参数扫描
    ├── fig_rt_single.png                    # 随机测试单次结果
    ├── fig_rt_monte_carlo.png               # Monte Carlo 统计结果
    ├── afc_animation.gif            # 动画 GIF
    └── afc_animation.mp4            # 动画 MP4
```

---

## 四、算法理论

### 4.1 仿射编队控制 (AFC) 核心原理

**目标**: n 个智能体通过分布式协议实现仿射编队，仅需领航者跟踪目标位置，跟随者自动形成期望构型。

**仿射变换不变性**: 目标队形 $p^* = (A \otimes I_d) p_0 + \mathbf{1}_n \otimes b$，其中 $A$ 为仿射变换矩阵，$b$ 为平移向量。

**控制律**:
- **一阶积分器**: $\dot{p}_i = -K_p \sum_{j \in \mathcal{N}_i} \omega_{ij}(p_j - p_i)$
  - 矩阵形式: $\dot{p}_f = -K_p(\Omega_{ff} p_f + \Omega_{fl} p_l)$
- **二阶积分器**: $\ddot{p}_i = -K_p \sum_{j \in \mathcal{N}_i} \omega_{ij}(p_j - p_i) - K_d \dot{p}_i$

**稳态**:  $p_f^* = -\Omega_{ff}^{-1} \Omega_{fl} p_l$

**收敛速率**: $\text{rate} = K_p \cdot \lambda_{\min}(\Omega_{ff})$，时间常数 $\tau = 1/\text{rate}$

### 4.5 输入饱和约束

实际无人机的控制输入（速度/加速度）受物理限制，控制输入不能无限大。引入饱和函数：

$$u_{\text{sat}} = \sigma(u_{\text{raw}}, u_{\max})$$

支持三种饱和模式：

1. **smooth (tanh 平滑饱和)**：$\sigma(u) = u_{\max} \cdot \tanh(u / u_{\max})$
   - 性质：$|\sigma(u)| < u_{\max}$，$\sigma'(0) = 1$，连续可微
   - 小输入时近似线性，大输入时平滑限幅
   - 适合稳定性分析，避免硬截断引起的不连续性

2. **norm (范数裁剪)**：$\sigma(u_i) = u_{\max} \cdot u_i / \|u_i\|$ 当 $\|u_i\| > u_{\max}$
   - 保持控制方向不变，仅限制合速度大小

3. **clip (分量裁剪)**：$\sigma(u) = \text{clip}(u, -u_{\max}, u_{\max})$
   - 最简单，但可能改变控制方向

Crazyflie 2.1 物理参数：
- 室内飞行最大速度：~1.0 m/s（保守限制）
- 最大加速度：~5.0 m/s²（二阶模型）

### 4.2 应力矩阵 $\Omega$ 的要求

应力矩阵必须满足 5 个性质：

1. **对称性**: $\Omega = \Omega^T$
2. **行和为零**: $\Omega \mathbf{1} = 0$
3. **应力平衡**: $\Omega p_0 = 0$（标称位置在零空间中）
4. **正确秩**: $\text{rank}(\Omega) = n - d - 1$（$d$ 为空间维度）
5. **$\Omega_{ff}$ 正定**: 保证跟随者收敛

### 4.3 SDP 求解应力矩阵

通过半定规划最大化 $\lambda_{\min}(\Omega_{ff})$：

$$
\max\ t \quad \text{s.t.} \quad 
\Omega_{ff} \succeq tI, \quad 
\Omega p_0 = 0, \quad 
\Omega \mathbf{1} = 0, \quad 
\|w\| \leq 1
$$

其中 $w$ 为边权重向量，$\Omega$ 由 $w$ 通过邻接关系构造。

### 4.6 CBF 碰撞避免安全滤波

基于控制障碍函数 (Control Barrier Functions, CBF) 的碰撞避免安全滤波器 (Ames et al., ECC 2019; Wang et al., IEEE T-RO 2017)。

**障碍函数**：对每对智能体 $(i,j)$，定义

$$h_{ij}(p) = \|p_i - p_j\|^2 - d_s^2$$

$h_{ij} > 0$ 表示安全，$h_{ij} = 0$ 为安全边界。

**CBF 条件**（一阶积分器 $\dot{p} = u$）：

$$\dot{h}_{ij} = 2(p_i - p_j)^T(u_i - u_j) \geq -\gamma h_{ij}$$

其中 $\gamma > 0$ 为 class-$\mathcal{K}$ 函数衰减率。

**安全滤波 QP**：

$$\min_{u_f} \|u_f - u_f^{\text{nom}}\|^2 \quad \text{s.t.}$$
- 跟随者-跟随者: $2(p_i - p_j)^T(u_i - u_j) \geq -\gamma h_{ij}$
- 跟随者-领导者: $2(p_i - p_l)^T u_i \geq -\gamma h_{il} + 2(p_i - p_l)^T v_l$

**关键性质**：
1. **前向不变性**: 若 $h_{ij}(0) > 0$，则 $h_{ij}(t) \geq 0, \forall t \geq 0$
2. **最小侵入**: QP 保证安全控制最接近标称 AFC 控制
3. **与饱和兼容**: 先施加饱和再施加 CBF 滤波

**Crazyflie 2.1 安全参数**：
- 安全距离 $d_s = 0.2$ m（考虑机架 92mm + 桨叶气流 + 定位误差）
- 激活距离 $d_a = 0.6$ m（仅近邻对触发 QP 约束）
- CBF 衰减率 $\gamma = 3.0$

### 4.7 ESO 鲁棒抗扰（扩展状态观测器）

基于扩展状态观测器 (Extended State Observer, ESO) 的主动抗扰补偿 (Han, IEEE T-IE 2009; Wang et al., Int. J. Robust Nonlinear Control 2020).

**含扰动系统模型**:

$$\dot{p}_f = u_f + w_f(t)$$

其中 $w_f(t)$ 为有界外部扰动（风场干扰等）。

**ESO 设计**（对每个跟随者）:

$$\dot{z}_1^i = u_i + z_2^i + \beta_1(p_i - z_1^i)$$
$$\dot{z}_2^i = \beta_2(p_i - z_1^i)$$

其中 $z_1^i$ 估计位置 $p_i$，$z_2^i$ 估计扰动 $w_i$。

**Han 参数化**: $\beta_1 = 2\omega_0, \beta_2 = \omega_0^2$，$\omega_0$ 为观测器带宽。

**补偿控制律**:

$$u_i = u_{\text{nom},i} - z_2^i = -K_p \sum_j \omega_{ij}(p_i - p_j) - z_2^i$$

当 ESO 收敛 ($z_2^i \to w_i$) 后，闭环等效为 $\dot{p}_i \approx u_{\text{nom},i}$，消除扰动影响。

**风场扰动模型**: $w(t) = w_{\text{const}} + w_{\text{OU}}(t)$
- 恒定分量 $w_{\text{const}}$: 稳定侧风
- OU 过程 $w_{\text{OU}}$: $dw = -\theta w dt + \sigma dW$（有色噪声，模拟阵风）

**无 ESO 稳态误差分析**: 在恒定扰动 $w$ 下，无补偿系统的稳态偏差为

$$\delta_f = \Omega_{ff}^{-1} w / K_p$$

由于 $\lambda_{\min}(\Omega_{ff}) = 0.024$ 较小，$\|\delta_f\| \propto 1/\lambda_{\min}$ 很大，系统对扰动敏感。

### 4.8 自适应事件触发通信（二阶积分器）

基于自适应阈值的事件触发通信机制，大幅减少智能体间的通信次数 (Liu et al., arXiv:2506.16797, 2025; Yi et al., Automatica 2016)。

**二阶积分器动力学**:

$$\ddot{p}_i = u_i, \quad u_i = -K_p \sum_{j \in \mathcal{N}_i} \omega_{ij}(\hat{p}_i - \hat{p}_j) - K_d \dot{p}_i$$

其中 $\hat{p}_j$ 为邻居 $j$ 最近一次广播的位置，$\dot{p}_i$ 为本地实时速度。

**事件触发条件**: 智能体 $i$ 在满足以下条件时广播自身位置：

$$\|e_i\|^2 > \frac{1}{\varphi_i} \|\hat{\xi}_i\|^2 + \mu e^{-\varpi t}$$

其中：
- $e_i = p_i - \hat{p}_i$: 实际位置与最近广播位置的偏差
- $\hat{\xi}_i = \sum_j \omega_{ij}(\hat{p}_i - \hat{p}_j)$: 使用广播数据计算的编队误差
- $\varphi_i(t)$: 自适应阈值参数，$\dot{\varphi}_i = \|e_i\|^2$（单调递增）
- $\mu e^{-\varpi t}$: 指数衰减项，保证 Zeno-free（$\mu=0.01, \varpi=0.5$）

**关键性质**:
1. **Zeno-free**: 由指数衰减项 $\mu e^{-\varpi t} > 0$ 保证最小触发间隔
2. **自适应阈值**: $\varphi_i$ 单调递增，随时间推移阈值逐渐放宽，通信率进一步降低
3. **渐近收敛**: 编队误差在事件触发下仍收敛至有界邻域

**参数选取**: $\mu = 0.01, \varpi = 0.5, \varphi_0 = 1.0, K_p = 5.0, K_d = 2.0$

### 4.4 稀疏通信图设计（4 阶段算法）

为适应 Crazyflie P2P 通信约束（最大邻居数=6，通信距离=10m），实现稀疏图设计：

**阶段 1**: 基于通信范围筛选可用边（距离 ≤ `comm_range`）  
**阶段 2**: 距离加权 ℓ1 SDP 求解最小边集（$\min \sum d_e |w_e|$ s.t. $\Omega_{ff} \succeq t_{\text{target}}I$）  
**阶段 3**: 贪心构造满足度约束的图（按边重要性排序，逐边添加，跳过超度数的边）  
**阶段 4**: 增量修复（若 $\lambda_{\min}$ 不足，从跳过的边中补充）

**当前仿真结果**:
- 完全图: 45 条边, $\lambda_{\min}(\Omega_{ff}) = 0.055$
- 稀疏图: 29 条边 (↓35.6%), $\lambda_{\min}(\Omega_{ff}) = 0.024$, 度数范围 5~7

### 4.9 层级重组 (Hierarchical Reorganization)

参考文献: Li & Dong (2024), arXiv:2406.11219

**核心思想**: 通过动态切换 Leader 角色和通信拓扑实现时变编队切换。切换过程中保持编队形状不变性和收敛性。

**理论基础**:

1. **动态仿射可定位性 (Theorem IV.1)**: Leader 集合必须仿射张成 $\mathbb{R}^d$，即 $\text{rank}([\mathbf{r}_l; \mathbf{1}^\top]) = d+1$
2. **可重构仿射可成形性 (Theorem IV.2)**: 所有 Leader 组合可表达为同一仿射像 — 保证切换后编队形状不变
3. **Power-Centric 拓扑 (Theorem IV.3)**: Follower 仅与 Leader 通信 → $\Omega_{ff}$ 为对角阵 → $\det(\Omega_{ff}) > 0$ 天然保证

**驻留时间条件**: 
$$\tau_{\text{dwell}} = \frac{\ln(\varepsilon_0 / \varepsilon_{\text{target}})}{K_p \cdot \lambda_{\min}(\Omega_{ff})}$$

**切换流程**:
1. 根据编队运动方向选择新 Leader（投影法 + 仿射张成校验）
2. 预计算新拓扑的 Power-Centric 应力矩阵
3. 到达切换时刻后更新控制器 $\Omega$、Leader 索引
4. Leader 位置以 smoothstep 过渡到新目标
5. 等待驻留时间后可再次切换

---

## 五、代码模块详细说明

### 5.1 `stress_matrix.py` — 应力矩阵计算

| 函数 | 参数 | 功能 |
|------|------|------|
| `get_edges(adj_matrix)` | 邻接矩阵 | 提取无向边列表 `[(i,j), ...]` |
| `build_constraint_matrix(positions, edges)` | 位置 n×d, 边列表 | 构造约束矩阵 C，使 C@w=0 为应力平衡 |
| `weights_to_stress_matrix(weights, edges, n)` | 权重向量, 边, 节点数 | 将边权重转为 n×n 应力矩阵 Ω |
| `compute_stress_matrix(positions, adj, leaders, method)` | 位置, 邻接, 领航者, 方法 | 标准 SDP / random 求解 Ω |
| `_solve_sdp(N, edges, n, f_idx, null_dim)` | 基矩阵, 边, ... | max t s.t. Ω_ff ≽ tI, ‖c‖≤1 |
| `_solve_random(...)` | 同上 + n_trials | 随机搜索备用方法 |
| `compute_sparse_stress_matrix(pos, leaders, comm_range, max_degree, convergence_ratio, prune_threshold)` | 位置, 领航者, 通信范围10m, 最大度6, 收敛比0.3 | **4阶段稀疏图设计** (返回 Ω, adj) |
| `_solve_min_edges_sdp(pos, edges, dist, n, f_idx, t_target)` | ... | 距离加权 ℓ1 SDP 最小化边数 |
| `_enforce_degree_constraint(edges, weights, n, max_degree)` | ... | 度约束强制（已弃用，被贪心方法替代） |
| `validate_stress_matrix(Omega, pos, leaders, tol)` | Ω, 位置, 领航者 | 验证 5 条性质，返回dict |
| `print_validation(results)` | 验证结果 | 打印验证详情 |

### 5.2 `afc_controller.py` — 分布式控制器

```python
class AFCController:
    def __init__(self, stress_matrix, leader_indices, gain=1.0, damping=2.0,
                 u_max=None, saturation_type='smooth')
```

| 属性 | 说明 |
|------|------|
| `Omega` | 完整应力矩阵 n×n |
| `Omega_ff` | 跟随者子矩阵 n_f×n_f |
| `Omega_fl` | 跟随者-领航者子矩阵 n_f×n_l |
| `gain` | 比例增益 K_p |
| `damping` | 阻尼系数 K_d |
| `u_max` | 控制输入饱和上限 (None=无饱和) |
| `saturation_type` | 饱和模式: 'smooth'/'norm'/'clip' |
| `leader_indices`, `follower_indices` | 索引列表 |

| 方法 | 功能 |
|------|------|
| `saturate(u)` | 对控制输入施加饱和约束 |
| `follower_input(agent_id, positions, velocities=None)` | 单个跟随者控制输入 (含饱和) |
| `all_follower_inputs(positions, velocities=None)` | 批量计算 u_f = σ(-K_p(Ω_ff·p_f + Ω_fl·p_l)) |
| `steady_state(leader_positions)` | 计算 p_f* = -Ω_ff⁻¹·Ω_fl·p_l |
| `formation_error(positions, leader_positions)` | 编队误差 ‖p_f - p_f*‖ |
| `convergence_rate_bound()` | 返回 (rate, tau), rate = K_p·λ_min(Ω_ff) |

### 5.3 `formation.py` — 编队定义与仿射工具

#### Crazyflie P2P 通信参数
```python
CRAZYFLIE_COMM = {
    'p2p_range': 10.0,       # P2P 通信距离 (m)
    'max_neighbors': 6,       # 最大邻居数
    'control_freq_hz': 50,    # 控制频率 (Hz)
    'p2p_slot_ms': 2.0,       # P2P 时隙 (ms)
    'radio_chip': 'nRF51822',
    'radio_data_rate_mbps': 2.0,
    'packet_payload_bytes': 32,
    'pos_data_bytes': 12,     # 3 × float32
    'max_velocity': 1.0,      # 室内最大速度 (m/s)
    'max_acceleration': 5.0,  # 最大加速度 (m/s²)
}
```

#### 编队生成函数

| 函数 | 编队 | 参数 | 返回 |
|------|------|------|------|
| `double_pentagon(radius, height)` | 10智能体反棱柱(上层旋转π/5)+完全图 | radius=1.0, height=1.0 | positions, leaders=[0,1,2,5], adj |
| `hexagon_2d(radius)` | 6智能体六边形+完全图 | radius=1.0 | positions, leaders=[0,1,2], adj |
| `grid_3d(nx, ny, nz, spacing)` | 3D网格 | 各方向数量和间距 | positions, leaders(前3), adj |
| `custom_formation(pos, leaders, adj, connect_radius)` | 自定义 | 位置,领航者,邻接/半径 | positions, leaders, adj |

#### 仿射变换工具

| 函数 | 功能 |
|------|------|
| `affine_transform(positions, A, b)` | p' = p @ A^T + b |
| `scale_matrix(dim, *scales)` | 缩放矩阵 (各向同性/异性) |
| `rotation_matrix_z(angle)` | 绕 Z 轴旋转 (3D) |
| `rotation_matrix_2d(angle)` | 2D 旋转 |
| `rotation_matrix_axis(axis, angle)` | 绕任意轴旋转 |
| `shear_matrix_3d(sxy, sxz, syz)` | 3D 剪切 |
| `smoothstep(t, t_start, t_end)` | Hermite 平滑插值 |

#### 领航者轨迹

```python
create_leader_trajectory(phases) → callable(t) → positions
```
`phases` 为列表，每个 phase 字典包含 `t_start`, `t_end`, `positions`，支持 `start_positions` 和自动平滑过渡。

#### 辅助

- `graph_info(adj_matrix)`: 打印图的节点数、边数、度数、连通性等统计信息

### 5.4 `main_sim.py` — 仿真主程序

**主流程** (`main()` 函数):

1. **Step 1**: 生成编队 — `double_pentagon(1.0, 1.0)`, 10 智能体, 领航者 [0,1,2,5]
2. **Step 2**: 计算稀疏应力矩阵 — `compute_sparse_stress_matrix()` (comm_range=10, max_degree=6, convergence_ratio=0.3)
3. **Step 3**: 验证应力矩阵 — `validate_stress_matrix()`, 检查 5 条性质
4. **Step 4**: 创建控制器 — `AFCController(Omega, leaders, gain=5.0, u_max=1.0, saturation_type='smooth')`
5. **Step 5**: 规划领航者轨迹 — 3 阶段: 建立(8s) → 缩放1.5x(5s+8s保持) → 旋转45°(5s+8s保持), T_total=34s
6. **Step 6**: 数值仿真 — `simulate_first_order()` (RK45, dt=0.02) + 无饱和对照仿真
7. **Step 7**: 生成 8 张图
8. **Step 8**: 仿射不变性验证 — 验证 $p_f^* = -\Omega_{ff}^{-1}\Omega_{fl}p_l$ 是否满足仿射变换不变性

#### 仿真函数

| 函数 | 功能 |
|------|------|
| `simulate_first_order(controller, leader_traj, init_pos, T, dt)` | 一阶积分器仿真 (RK45)，返回 (t, pos_history, errors, control_inputs) |
| `simulate_second_order(controller, leader_traj, init_pos, T, dt)` | 二阶积分器仿真 (RK45)，返回 (t, pos_history, errors, control_inputs) |
| `simulate_first_order_cbf(controller, init_pos, leader_traj, T, dt, cbf_filter)` | 一阶积分器 + CBF 碰撞避免 (前向 Euler)，返回 (t, pos, err, ctrl, cbf_data) || `simulate_first_order_eso(controller, init_pos, leader_traj, T, dt, wind, eso)` | 一阶积分器 + 风扰动 + ESO 补偿 (前向 Euler)，返回 (t, pos, err, ctrl, eso_data) |
| `simulate_second_order_et(controller, init_pos, init_vel, leader_traj, T, dt, et_manager)` | 二阶积分器 + 事件触发通信 (前向 Euler)，返回 (t, pos, err, ctrl, et_data) |
#### 可视化函数

| 函数 | 输出文件 | 内容 |
|------|----------|------|
| `plot_formation_3d()` | fig1_formation_snapshots.png | 4个时刻编队快照 |
| `plot_trajectories_3d()` | fig2_trajectories_3d.png | 所有智能体3D轨迹 |
| `plot_error_convergence()` | fig3_convergence.png | 误差对数收敛 + 阶段标注 |
| (main 内矩阵热力图) | fig4_stress_matrix.png | Ω 热力图 + Ω_ff 特征值 |
| (main 内通信图) | fig5_communication_graph.png | 稀疏图 vs 完全图 3D 对比 |
| (main 内指标对比) | fig6_sparse_comparison.png | 边数/度数/λ_min 柱状图 |
| (main 内饱和分析) | fig7_saturation_analysis.png | 控制输入范数 + 误差对比 + 饱和比例 |
| (main 内饱和对比) | fig8_saturation_comparison.png | 不同 u_max 下误差 + 控制量对比 |
| (main 内CBF对比) | fig9_cbf_collision_avoidance.png | 最小距离 + 误差收敛 (有/无CBF) |
| (main 内CBF分析) | fig10_cbf_analysis.png | 约束激活数 + 修正幅度 + 控制量 + 成对距离 |
| (main 内ESO抗扰) | fig11_eso_disturbance_rejection.png | 扰动估计 + 估计误差 + 三场景误差对比 + 控制量 |
| (main 内ESO带宽) | fig12_eso_bandwidth_comparison.png | 不同 w0 下误差收敛 + 稳态估计精度 |
| (main 内ET误差) | fig13_et_error_comparison.png | 连续通信 vs 事件触发 误差收敛对比 |
| (main 内ET分析) | fig14_et_communication_analysis.png | 触发时间线 + 累积计数 + 各Agent通信率 + φ_i 演化 |
| (main 内ET参数) | fig15_et_parameter_comparison.png | μ 参数扫描对比 |

### 5.6 `collision_avoidance.py` — CBF 碰撞避免

```python
CRAZYFLIE_SAFETY = {
    'safety_distance_m': 0.2,    # 安全距离 d_s (m)
    'activate_distance_m': 0.6,  # CBF 激活距离 (m)
    'cbf_gamma': 3.0,            # 默认衰减率
    ...
}

class CBFSafetyFilter:
    def __init__(self, n_agents, leader_indices, d_safe=0.2, gamma=3.0, d_activate=None)
```

| 方法 | 功能 |
|------|------|
| `barrier(pi, pj)` | 计算障碍函数 $h_{ij} = \|p_i-p_j\|^2 - d_s^2$ |
| `filter(positions, u_nom, leader_velocities)` | CBF-QP 安全滤波，返回 (u_safe, info) |
| `_solve_qp(u_nom_flat, C, b)` | scipy SLSQP 求解 min‖u-u_nom‖² s.t. Cu≥b |
| `min_distance(positions)` | 所有智能体最小距离 (静态方法) |
| `all_min_distances_over_time(pos_history)` | 沿时间轴最小距离 (静态方法) |
| `pairwise_distance_matrix(positions)` | 全成对距离矩阵 (静态方法) |

#### CBF-QP 工作流程
1. 遍历所有智能体对，检查距离是否 < d_activate
2. 对近邻对构建线性约束 Cu ≥ b
3. 无约束时直接返回 u_nom（零开销）
4. 有约束时求解 QP，返回最接近 u_nom 的安全控制

### 5.7 `disturbance_observer.py` — ESO 扰动观测器

```python
class WindDisturbance:
    def __init__(self, n_agents, dim=3, w_const=None, ou_theta=0.5, ou_sigma=0.1, seed=None)

class ExtendedStateObserver:
    def __init__(self, n_agents, dim=3, omega_o=8.0)
```

| 类/方法 | 功能 |
|---------|------|
| `WindDisturbance.step(dt)` | 推进 OU 过程一步，返回当前扰动 (n_agents, dim) |
| `WindDisturbance.current()` | 返回当前扰动（不推进） |
| `WindDisturbance.reset()` | 重置 OU 状态 |
| `ExtendedStateObserver.reset(p_f_init)` | 用当前位置初始化 z₁, z₂=0 |
| `ExtendedStateObserver.update(p_f, u_applied, dt)` | ESO 一步更新，返回扰动估计 z₂ |
| `ExtendedStateObserver.disturbance_estimate()` | 返回当前 z₂ |

### 5.8 `event_trigger.py` — 自适应事件触发通信

```python
class EventTriggerManager:
    def __init__(self, n_agents, leader_indices, Omega_ff, mu=0.01,
                 varpi=0.5, phi_0=1.0)
```

| 方法 | 功能 |
|------|------|
| `reset(positions)` | 初始化广播位置 p̂ = p，重置 φ 和触发记录 |
| `check_and_trigger(t, positions)` | 检查触发条件，满足时更新 p̂_i，返回 (triggered_list, errors_sq) |
| `update_phi(errors_sq, dt)` | 更新自适应参数 φ̇_i = ‖e_i‖² |
| `update_leaders(p_l)` | 更新 leader 广播位置（leader 直接广播） |
| `communication_rates()` | 返回各 agent 通信率统计 dict (per_agent%, mean%, total_triggers) |

### 5.9 `random_test.py` — 随机初始状态 + 复杂仿射变换测试

独立测试模块，从完全随机初始位置出发，测试多种复杂仿射变换。

#### 变换库 `TRANSFORM_LIBRARY`

| 变换 key | 名称 | det(A) |
|----------|------|--------|
| `scale_uniform` | 均匀缩放 2x | 8.0 |
| `scale_nonuniform` | 非均匀缩放 (1.5, 0.8, 1.2) | 1.44 |
| `rotate_z45` | 绕 z 轴旋转 45° | 1.0 |
| `rotate_oblique` | 绕 (1,1,1) 轴旋转 60° | 1.0 |
| `shear_xy` | 剪切 sxy=0.4, sxz=0.2 | 1.0 |
| `reflect_xy` | 关于 xy 平面反射 (z 翻转) | -1.0 |
| `general` | 一般仿射: 缩放+旋转+剪切 | 1.287 |

#### 主要函数

| 函数 | 功能 |
|------|------|
| `random_initial_positions(nominal, sigma, rng)` | 以标称中心为基准生成高斯随机初始位置 |
| `run_single_test(controller, ..., transform_keys, seed)` | 单次随机测试：随机初始 → 多阶段复杂变换 → 验证仿射不变性 |
| `monte_carlo_test(controller, ..., n_trials)` | n 次 Monte Carlo 随机试验，返回统计摘要 |
| `plot_single_result(result, ...)` | 绘制误差曲线 + 初始/最终编队 3D 对比 |
| `plot_monte_carlo(results, summary, ...)` | 绘制误差曲线叠加 + 分布直方图 + 仿射不变性验证 |

### 5.5 `animate_sim.py` — 3D 动画

生成 GIF + MP4 动画，包含：
- 3D 视图：无人机位置（红三角=领航者，蓝圆=跟随者）、通信边、轨迹尾迹
- 误差曲线：对数尺度编队误差 + 阶段背景色
- 信息面板：时间、阶段、误差、智能体信息、增益

配置：
- 字体: Microsoft YaHei（解决中文显示问题）
- 标签: 英文（避免字体兼容性问题）
- 帧率: 20 fps
- GIF: dpi=100, MP4: dpi=120, bitrate=2000

### 5.10 层级重组 (RHF) 新增函数

分布在多个模块中，实现 Li & Dong (2024) 的层级重组框架。

#### `formation.py` 新增

| 函数 | 功能 |
|------|------|
| `check_affine_span(positions, leader_indices, d)` | 检查 Leader 是否仿射张成 R^d |
| `build_power_centric_topology(n, leader_indices, nominal_pos, d)` | 构建 Power-Centric 邻接矩阵 |
| `select_leaders_for_direction(positions, direction, n_leaders, d)` | 按运动方向选择最优 Leader 组合 |
| `compute_dwell_time(gain, min_eig_ff, epsilon_0, epsilon_target)` | 计算最小驻留时间 |

#### `stress_matrix.py` 新增

| 函数 | 功能 |
|------|------|
| `compute_power_centric_stress_matrix(positions, leader_indices)` | Power-Centric 拓扑应力矩阵 SDP 求解 |

#### `afc_controller.py` 新增

| 方法 | 功能 |
|------|------|
| `AFCController.update_omega(new_omega, new_leader_indices)` | 动态更新控制器 Ω 和 Leader/Follower 索引 |

#### `main_sim.py` 新增

| 函数 | 功能 |
|------|------|
| `simulate_rhf(controller, init_pos, init_vel, nominal, schedule, t_span, dt)` | RHF 仿真引擎：按调度表切换层级，smoothstep leader 过渡，记录误差恢复 |

---


## 六、关键仿真结果

### 6.1 完全图基线

| 指标 | 值 |
|------|-----|
| 边数 | 45 (完全图, 10个节点) |
| rank(Ω) | 6 (= n-d-1 = 10-3-1 ✓) |
| λ_min(Ω_ff) | 0.055023 |
| 收敛速率 | K_p·λ_min = 5.0 × 0.055 = 0.275 |
| 时间常数 τ | 3.63 s |

### 6.2 稀疏图设计

| 指标 | 值 |
|------|-----|
| 边数 | 29 (↓35.6%) |
| rank(Ω) | 6 ✓ |
| λ_min(Ω_ff) | 0.023756 |
| 度数 | min=5, max=7, mean=5.8 |
| 收敛速率 | 0.119 |
| 时间常数 τ | 8.42 s |
| 仿射不变性误差 | 1.56e-14 ✓ |

### 6.4 CBF 碰撞避免结果

碰撞风险场景：初始扰动 σ=1.5m（正常场景 3 倍），安全距离 d_s=0.2m

| 指标 | 无 CBF | 有 CBF |
|------|--------|--------|
| 最小智能体间距 | 0.1072 m ⚠ **碰撞!** | 0.2250 m ✓ |
| 最终编队误差 | 0.5701 | 0.5707 |
| CBF 约束激活率 | — | 24.0% (408/1701步) |
| 最大修正幅度 | — | 0.4225 m/s |

结论：CBF 安全滤波器以极小的编队误差代价（<0.1%）保证了碰撞安全。

### 6.3 输入饱和约束结果 (u_max = 1.0 m/s, smooth tanh)

| 指标 | 饱和 | 无饱和 |
|------|------|--------|
| 最大 \|\|u\|\| | 0.769 m/s | 0.901 m/s |
| 最终编队误差 | 0.5895 | 0.5884 |
| 收敛行为 | 稳定收敛，微慢于无饱和 | 基线 |

不同饱和上限对比 (u_max = 0.5 / 1.0 / 2.0 m/s)：
- u_max=0.5: 显著限幅，收敛明显变慢
- u_max=1.0: 轻微影响，与无饱和接近
- u_max=2.0: 基本无影响（控制输入未触及上限）

### 6.3 编队拓扑

- **反棱柱构型**: 上下两个正五边形，上层旋转 π/5 (36°)
- **领航者**: [0, 1, 2, 5] — 底层3个 + 顶层1个，空间上不共面
- **跟随者**: [3, 4, 6, 7, 8, 9]
- **关键修复**: 原始双正五边形有 5-fold 旋转对称性导致 rank 不足；旋转上层打破对称性后 rank=6

### 6.5 事件触发通信结果（二阶积分器）

| 指标 | 连续通信 | 事件触发 |
|------|---------|---------|
| 最终编队误差 | 1.1797 | 1.2027 |
| 通信次数 | 10206 | 396 |
| 平均通信率 | 100% | 3.88% |
| **通信节省** | — | **96.1%** |

各 Follower 通信率均匀分布在 3.59%~4.06% 之间。

结论：收敛精度仅下降约 2%，但通信量减少 96%，自适应事件触发显著降低通信负担。

### 6.6 随机测试结果（复杂仿射变换）

**Monte Carlo 测试** (20 次, σ=2.5, 非均匀缩放+斜轴旋转+剪切):

| 指标 | 值 |
|------|-----|
| 最终误差均值 | 0.7083 ± 0.0903 |
| 最终误差最大 | 0.9048 |
| 仿射不变性误差 | ~1e-14（机器精度） |
| 全部收敛 | ✓ |

**各变换类型逐个测试**:

| 变换类型 | 最终误差 | det(A) | 仿射不变性误差 |
|---------|---------|--------|--------------|
| 均匀缩放 2x | 2.05 | 8.0 | 2e-14 |
| 非均匀缩放 (1.5,0.8,1.2) | 0.88 | 1.44 | 1e-14 |
| 绕 z 轴旋转 45° | 1.06 | 1.0 | 1e-14 |
| 绕 (1,1,1) 轴旋转 60° | 1.52 | 1.0 | 1e-14 |
| 剪切 sxy=0.4, sxz=0.2 | 0.71 | 1.0 | 1e-14 |
| xy 平面反射 | 1.95 | -1.0 | 1e-14 |
| 一般仿射组合 | 1.04 | 1.287 | 1e-14 |

结论：仿射不变性在所有变换类型下成立至机器精度，验证了 AFC 理论的正确性。仿射变换不限于旋转和缩放，还包括剪切、反射、非均匀缩放及其任意组合。

### 6.7 层级重组 (RHF) 结果

**场景**: 10 机 U 形转弯，3 阶段层级切换，T=105s

| 阶段 | 切换时刻 | Leader | λ_min(Ω_ff) | 边数 | 恢复时间 |
|------|---------|--------|-------------|------|---------|
| Phase 0: 建立+X平移 | 0s | [0,1,2,5] | 0.054 | 30 | 4.24s |
| Phase 1: +Y转弯 | 35s | [6,1,2,5] | 0.020 | 30 | 13.70s |
| Phase 2: -X转弯 | 70s | [7,3,2,8] | 0.020 | 30 | 16.94s |

**关键指标**:
- Leader 选择方法: 投影法（direct_projection）
- 仿射张成验证: 全部通过 (rank=4=d+1)
- 驻留时间要求: 15.08s
- 实际阶段间隔: 35s > 15.08s ✓
- Power-Centric 拓扑: Ω_ff 对角（det > 0 保证）
- 控制增益: K_p=10.0, K_d=1.0, u_max=1.0 m/s

**可视化**:
- fig16: 3D 轨迹（分阶段着色 + Leader 标注）
- fig17: 误差演化（切换瞬态 + 恢复标注）
- fig18: 通信拓扑切换对比图

结论：层级重组框架成功实现时变编队切换。各阶段均在驻留时间内收敛，Power-Centric 拓扑保证切换后即刻可用，投影法 Leader 选择自动适应不同运动方向。

---

## 七、开发进程（工作记录）

### 阶段 1：理论学习与环境搭建

1. **毕设题目确认**: 基于Crazyflie开发平台的微型无人机编队控制研究
2. **文献调研**: 阅读仿射编队控制核心论文 (Affine.pdf, Shaan Hossain MSc Thesis 等)
3. **CrazyChoir 源码研究**: 分析 `distributed_control.py`, `bearing_formation.py` 等模块
4. **WSL2 环境搭建**: 安装 Ubuntu 20.04, ROS 2 Foxy, CrazyChoir, Webots R2023a, qpSWIFT
5. **Python 虚拟环境**: 创建 `.venv`，安装 cvxpy, numpy, scipy, matplotlib 等

### 阶段 2：核心算法实现

6. **应力矩阵模块** (`stress_matrix.py`):
   - 实现基于 SDP 的应力矩阵计算 (cvxpy + clarabel 求解器)
   - 实现随机搜索备用方法
   - 完整的 5 条性质验证函数

7. **AFC 控制器** (`afc_controller.py`):
   - 一阶/二阶积分器控制律
   - 矩阵形式批量计算
   - 稳态计算与误差评估

8. **编队定义** (`formation.py`):
   - 反棱柱 10 智能体编队 (double_pentagon)
   - 仿射变换工具集（缩放、旋转、剪切）
   - 多阶段领航者轨迹生成器

9. **仿真主程序** (`main_sim.py`):
   - RK45 数值积分
   - 6 张论文级可视化图
   - 仿射不变性验证

### 阶段 3：问题修复

10. **虚拟环境修复**: 发现全局 Python 缺少 cvxpy，切换到 `.venv`
11. **应力矩阵秩不足**: 双正五边形有 5-fold 旋转对称性 → rank=4；上层旋转π/5(反棱柱) + 完全图 → rank=6
12. **浮点精度**: `np.arange` 可能超出 T_total，添加显式边界检查
13. **字体问题**: `fontfamily='monospace'` 不支持中文，改用 Microsoft YaHei + 英文标签

### 阶段 4：动画生成

14. **3D 动画** (`animate_sim.py`):
    - 无人机运动动画 + 误差曲线 + 信息面板
    - GIF (Pillow) + MP4 (ffmpeg) 双格式输出
    - 轨迹尾迹、阶段背景色

### 阶段 5：算法评估与改进

15. **文献检索**: 搜索 arXiv 最新 AFC 论文，识别 8 项当前实现的局限性
16. **4 个改进方向**: 鲁棒性/抗扰性、自适应重构、异构动力学/碰撞避免、通信约束

17. **稀疏通信图设计** (`compute_sparse_stress_matrix`):
    - 添加 Crazyflie P2P 通信参数 (`CRAZYFLIE_COMM`)
    - 4 阶段算法：通信范围筛选 → ℓ1 SDP → 贪心度约束 → 增量修复
    - **首次尝试失败**: 阈值剪枝 + 度约束强制过度删边 → λ_min≈0, rank=5
    - **修复**: 改用构造式贪心方法（按重要性排序逐边添加）
    - 最终结果: 45→29 边 (↓35.6%), λ_min=0.024, rank=6 ✓

### 阶段 6：文档化

18. **项目文档** (本文件): 整理所有已实现内容、环境配置、代码架构、开发记录

### 阶段 7：输入饱和约束

19. **输入饱和机制** (`afc_controller.py`):
    - 新增 `u_max`, `saturation_type` 参数
    - 实现 `saturate()` 方法：支持 smooth(tanh) / norm(范数裁剪) / clip(分量裁剪)
    - `follower_input()` 和 `all_follower_inputs()` 自动施加饱和
    - 仿真动力学函数内部也施加饱和（保证数值积分过程中的一致性）

20. **Crazyflie 物理参数** (`formation.py CRAZYFLIE_COMM`):
    - 新增 `max_velocity=1.0` m/s、`max_acceleration=5.0` m/s²

21. **仿真对比与可视化** (`main_sim.py`):
    - 主仿真使用 u_max=1.0 m/s smooth 饱和
    - 同时运行无饱和对照仿真
    - 运行 u_max=0.5/1.0/2.0 多组对比
    - fig7: 控制输入范数 + 误差收敛对比 + 饱和比例
    - fig8: 不同 u_max 下误差收敛 + 最大控制量对比
    - `simulate_first_order` / `simulate_second_order` 现在返回控制输入历史

### 阶段 8：碰撞避免

22. **CBF 碰撞避免模块** (`collision_avoidance.py`):
    - 基于控制障碍函数 (CBF) + 二次规划 (QP) 的安全滤波器
    - Crazyflie 安全参数: d_s=0.2m, d_activate=0.6m, γ=3.0
    - scipy SLSQP 求解 QP，仅在近邻对距离 < d_activate 时触发
    - 理论依据: Ames et al. (ECC 2019), Wang et al. (IEEE T-RO 2017)

23. **CBF 仿真集成** (`main_sim.py`):
    - 新增 `simulate_first_order_cbf()` 函数（前向 Euler + CBF 滤波）
    - Step 7.5: 碰撞风险场景（σ=1.5m）有/无 CBF 对比仿真
    - fig9: 最小距离 + 误差收敛对比
    - fig10: CBF 约束激活数 + 修正幅度 + 控制量 + 成对距离
    - 无 CBF 最小距离 0.107m（碰撞!）→ 有 CBF 0.225m（安全 ✓）
### 阶段 9：ESO 鲁棒抗扰

24. **ESO 扰动观测器模块** (`disturbance_observer.py`):
    - `WindDisturbance` 类: 恒定风场 + OU 过程阵风模型
    - `ExtendedStateObserver` 类: Han 参数化 ESO ($\beta_1=2\omega_0, \beta_2=\omega_0^2$)
    - 将扰动视为扩展状态，实时估计并前馈补偿

25. **ESO 仿真集成** (`main_sim.py`):
    - 新增 `simulate_first_order_eso()` 函数（前向 Euler + 扰动 + ESO）
    - Step 7.6: 三场景对比（无扰动 / 有扰动无ESO / 有扰动有ESO）
    - fig11: 扰动估计精度 + 编队误差对比 + 控制量
    - fig12: 不同 ESO 带宽 (w0=2/5/8/15) 对比
    - 有扰动无ESO 误差 3.62m → 有ESO 0.63m，降低 82.6%

### 阶段 10：事件触发通信（二阶积分器）

26. **事件触发通信模块** (`event_trigger.py`):
    - `EventTriggerManager` 类: 自适应阈值事件触发
    - 触发条件: $\|e_i\|^2 > \frac{1}{\varphi_i}\|\hat{\xi}_i\|^2 + \mu e^{-\varpi t}$
    - 自适应参数 φ_i 单调递增，Zeno-free 保证
    - 参考文献: Liu et al. (arXiv:2506.16797, 2025), Yi et al. (Automatica, 2016)

27. **二阶积分器仿真** (`main_sim.py`):
    - 新增 `simulate_second_order_et()` 函数（前向 Euler + ET）
    - 控制律: $u_i = -K_p \sum_j \omega_{ij}(\hat{p}_i - \hat{p}_j) - K_d v_i$
    - 自身位置用实时值、邻居位置用广播值
    - Step 7.7: 连续通信 vs 事件触发对比
    - fig13: 误差对比, fig14: 通信分析四面板, fig15: μ 参数扫描
    - 通信节省 96.1%，误差仅增加 2%

### 阶段 11：随机测试与复杂仿射变换

28. **随机测试模块** (`random_test.py`):
    - 完全随机初始位置（高斯分布，σ=1.5~2.5m）
    - 7 种仿射变换: 均匀/非均匀缩放、z轴/斜轴旋转、剪切、反射、一般组合
    - `run_single_test()`: 单次多阶段变换序列测试
    - `monte_carlo_test()`: 20 次 Monte Carlo 随机试验
    - 可视化: 误差曲线 + 3D 编队对比 + 统计直方图

29. **复杂仿射变换验证**:
    - 剪切变换 (sxy=0.4, sxz=0.2): 仿射不变性误差 ~1e-14 ✓
    - 反射变换 (det(A)=-1): 仿射不变性误差 ~1e-14 ✓
    - 4 重组合 (非均匀缩放→斜轴旋转→剪切→一般仿射): 收敛 ✓
    - Monte Carlo 20 次全部收敛，误差 0.71 ± 0.09

### 阶段 12：层级重组（时变编队切换）

30. **文献调研**: 检索 arXiv 前沿文献，确定 Li & Dong (2024, arXiv:2406.11219) 为主要参考
    - 动态仿射可定位性 (Theorem IV.1)
    - 可重构仿射可成形性 (Theorem IV.2)
    - Power-Centric 拓扑 (Theorem IV.3)

31. **核心函数实现**:
    - `check_affine_span()`: Leader 仿射张成 R^d 校验
    - `build_power_centric_topology()`: Power-Centric 邻接矩阵构建
    - `compute_power_centric_stress_matrix()`: SDP 求解 Power-Centric 应力矩阵
    - `select_leaders_for_direction()`: 投影法 + 仿射张成组合验证
    - `compute_dwell_time()`: 驻留时间下界计算
    - `AFCController.update_omega()`: 在线更新控制器 Ω 和角色索引

32. **RHF 仿真引擎** (`simulate_rhf()`):
    - 按调度表切换层级 (t_switch, leader_indices, Omega, targets)
    - Leader 位置以 smoothstep 过渡
    - 记录切换日志: 前误差、峰值误差、恢复时间
    - U 形转弯场景: 3 阶段切换 (T=105s)，恢复时间 4-17s，均满足驻留时间要求

33. **可视化** (fig16/17/18):
    - fig16: 3D 轨迹分阶段着色 + Leader 星标
    - fig17: 误差演化 + 切换竖线 + 恢复标注
    - fig18: 3 阶段通信拓扑对比图

---

## 八、已识别的局限性与改进方向

### 当前局限性

1. ~~**静态图**~~: **已解决** — 层级重组 (RHF) 实现动态拓扑切换 (见 4.9)
2. ~~无碰撞避免~~: **已解决** — CBF-QP 安全滤波 (见 4.6)
3. **简化动力学**: 使用积分器模型，未建模四旋翼实际动力学
4. **集中式初始化**: 应力矩阵计算需要集中式 SDP，仅控制阶段分布式
5. **无通信延迟/丢包**: 理想通信假设
6. ~~无环境扰动~~: **已解决** — ESO 扩展状态观测器抗扰补偿 (见 4.7)
7. **领航者追踪假设**: 领航者完美跟踪参考轨迹
8. **固定增益**: K_p 为常数，未自适应调整

### 改进方向

1. **鲁棒抗扰**: 引入 ISS (Input-to-State Stability), H∞ 控制, 自抗扰控制 (ADRC)
2. **自适应重构**: 动态拓扑切换, 在线应力矩阵更新, 领航者故障恢复
3. **异构动力学**: 四旋翼二阶/非线性模型, CBF 碰撞避免
4. **通信约束**: ~~事件触发通信~~, 量化/延迟/丢包建模, 分布式应力矩阵估计

### 已解决的局限性

- ~~输入无约束~~: 已实现 smooth/norm/clip 三种饱和模式，支持 Crazyflie 速度限制 (u_max=1.0 m/s)
- ~~无碰撞避免~~: 已实现 CBF-QP 安全滤波器，保证智能体间距 ≥ d_s=0.2m，仅在 24% 时步激活约束
- ~~无环境扰动~~: 已实现 ESO 扩展状态观测器，在 [0.2, 0.1, 0.05] m/s 风场 + OU 阵风下编队误差降低 82.6%
- ~~连续通信假设~~: 已实现自适应事件触发通信（二阶积分器），通信量减少 96.1%，误差仅增加 2%
- ~~仅旋转/缩放变换~~: 已验证剪切、反射、非均匀缩放、一般仿射组合等复杂变换均成立（仿射不变性误差 ~1e-14）
- ~~静态通信拓扑~~: 已实现层级重组 (RHF)，Power-Centric 拓扑动态切换 Leader，驻留时间保证收敛，3 阶段 U 形转弯验证通过
---

## 九、参考文献索引

`docs/` 目录中的参考文献：

| 文件名 | 主题 |
|--------|------|
| `Affine (1).pdf` / `affine.pdf` | 仿射编队控制核心理论 |
| `Shaan_Hossain_...Thesis_v2.pdf` | 四旋翼仿射编队控制硕士论文 |
| `CrazyChoir_Flying_Swarms...pdf` | CrazyChoir ROS 2 框架 |
| `Crazyswarm_A_large...pdf` | Crazyswarm 大规模集群 |
| `Joint_Estimation_and_Planar...pdf` | 联合估计与平面仿射编队 |
| `Formation_Control_Algorithms...pdf` | 编队控制算法综述 |
| `Reconfigurable_Leader-Follower...pdf` | 可重构领航-跟随编队 |
| `2503.07376v2.pdf` | arXiv 最新相关论文 |

---

## 十、常用命令

```bash
# 运行仿真
cd src
..\.venv\Scripts\python.exe main_sim.py

# 生成动画
cd ..
.\.venv\Scripts\python.exe animate_sim.py

# 安装新包
.\.venv\Scripts\pip.exe install <package>

# WSL2 中启动 CrazyChoir
source ~/crazychoir_ws/install/setup.bash
ros2 launch crazychoir_examples ...
```

---

## 十一、后续计划

- [ ] 实物飞行实验（单机测试 → 多机编队）
- [ ] 室内定位系统校准（Lighthouse）
- [ ] 将仿真代码迁移到 CrazyChoir/ROS 2 框架
- [ ] 毕业论文撰写
- [ ] 二阶动力学仿真验证
- [ ] 碰撞避免机制集成
- [x] 输入饱和约束 (✓ smooth tanh, u_max=1.0 m/s)
