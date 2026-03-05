# 项目文档：基于Crazyflie开发平台的微型无人机编队控制研究

> **最后更新**: 2025年7月 → 2026年3月05日  
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
| 虚拟环境 | `e:\crazyflie\.venv` |
| Python 路径 | `e:\crazyflie\.venv\Scripts\python.exe` |
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
e:\crazyflie\
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
    ├── main_sim.py             # 仿真主程序（~780行）
    ├── animate_sim.py          # 3D 动画生成（~270行）
    ├── fig1_formation_snapshots.png  # 编队快照
    ├── fig2_trajectories_3d.png     # 3D 轨迹
    ├── fig3_convergence.png         # 误差收敛
    ├── fig4_stress_matrix.png       # 应力矩阵热力图
    ├── fig5_communication_graph.png # 稀疏 vs 完全图对比
    ├── fig6_sparse_comparison.png   # 通信指标对比
    ├── fig7_saturation_analysis.png # 输入饱和分析
    ├── fig8_saturation_comparison.png # 不同饱和上限对比
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

### 4.4 稀疏通信图设计（4 阶段算法）

为适应 Crazyflie P2P 通信约束（最大邻居数=6，通信距离=10m），实现稀疏图设计：

**阶段 1**: 基于通信范围筛选可用边（距离 ≤ `comm_range`）  
**阶段 2**: 距离加权 ℓ1 SDP 求解最小边集（$\min \sum d_e |w_e|$ s.t. $\Omega_{ff} \succeq t_{\text{target}}I$）  
**阶段 3**: 贪心构造满足度约束的图（按边重要性排序，逐边添加，跳过超度数的边）  
**阶段 4**: 增量修复（若 $\lambda_{\min}$ 不足，从跳过的边中补充）

**当前仿真结果**:
- 完全图: 45 条边, $\lambda_{\min}(\Omega_{ff}) = 0.055$
- 稀疏图: 29 条边 (↓35.6%), $\lambda_{\min}(\Omega_{ff}) = 0.024$, 度数范围 5~7

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
| `simulate_first_order(controller, leader_traj, init_pos, T, dt)` | 一阶积分器仿真，返回 (t, pos_history, errors, control_inputs) |
| `simulate_second_order(controller, leader_traj, init_pos, T, dt)` | 二阶积分器仿真，返回 (t, pos_history, errors, control_inputs) |

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

---

## 八、已识别的局限性与改进方向

### 当前局限性

1. **静态图**: 通信拓扑固定，不支持动态切换/故障恢复
2. **无碰撞避免**: 仿真未考虑智能体间碰撞
3. **简化动力学**: 使用积分器模型，未建模四旋翼实际动力学
4. **集中式初始化**: 应力矩阵计算需要集中式 SDP，仅控制阶段分布式
5. **无通信延迟/丢包**: 理想通信假设
6. **无环境扰动**: 无风力等外部干扰
7. **领航者追踪假设**: 领航者完美跟踪参考轨迹
8. **固定增益**: K_p 为常数，未自适应调整

### 改进方向

1. **鲁棒抗扰**: 引入 ISS (Input-to-State Stability), H∞ 控制, 自抗扰控制 (ADRC)
2. **自适应重构**: 动态拓扑切换, 在线应力矩阵更新, 领航者故障恢复
3. **异构动力学**: 四旋翼二阶/非线性模型, CBF 碰撞避免
4. **通信约束**: 事件触发通信, 量化/延迟/丢包建模, 分布式应力矩阵估计

### 已解决的局限性

- ~~输入无约束~~: 已实现 smooth/norm/clip 三种饱和模式，支持 Crazyflie 速度限制 (u_max=1.0 m/s)

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
cd e:\crazyflie\src
e:\crazyflie\.venv\Scripts\python.exe main_sim.py

# 生成动画
e:\crazyflie\.venv\Scripts\python.exe animate_sim.py

# 安装新包
e:\crazyflie\.venv\Scripts\pip.exe install <package>

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
