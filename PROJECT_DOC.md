# 项目文档：基于 Crazyflie 开发平台的微型无人机编队控制研究

> 最后更新：2026年3月14日  
> 目的：作为当前仓库的总工作文档，统一记录研究目标、代码结构、仿真入口、模块能力、结果产物和最近重构状态，避免后续重复梳理代码。

---

## 一、课题与目标

- 题目：基于 Crazyflie 开发平台的微型无人机编队控制研究
- 英文：Research on Formation Control of Micro Unmanned Aerial Vehicles Based on the Crazyflie Development Platform
- 方向：飞行控制、集群控制、分布式编队控制
- 当前仿真规模：10 架无人机
- 实验平台规模：30 架 Crazyflie 2.1
- 定位系统：2 台 SteamVR Lighthouse
- 通信设备：多台 Crazyradio PA

### 毕设任务要求

| 序号 | 任务 | 指标要求 |
|------|------|----------|
| 1 | Crazyflie 平台搭建 | 完成硬件连接与软件安装 |
| 2 | 室内定位系统 | 定位误差 ≤ 0.5 m，更新频率 ≥ 2 Hz |
| 3 | 三维仿射编队控制算法 | 理论推导误差 ≤ 10% |
| 4 | 单机/多机飞行实验 | 多机编队位置偏差 ≤ 0.3 m |
| 5 | 毕业论文撰写 | 完整覆盖系统搭建、算法设计、实验验证 |

---

## 二、开发环境

### 2.1 Windows 算法开发环境

| 项目 | 配置 |
|------|------|
| Python | 3.12.6 |
| 虚拟环境 | .venv |
| 必用解释器 | .venv/Scripts/python.exe |
| 关键包 | cvxpy 1.8.1, numpy, scipy, matplotlib, clarabel, scs, osqp, highspy |

关键约束：

- 必须使用仓库内 .venv 的 Python。
- 全局 Python 不能保证装有 cvxpy、clarabel 等求解依赖。

### 2.2 WSL2 / ROS 2 侧环境

| 项目 | 配置 |
|------|------|
| 系统 | Ubuntu 20.04 (WSL2) |
| ROS 2 | Foxy |
| CrazyChoir | ~/crazychoir_ws/ |
| Webots | R2023a |
| qpSWIFT | 已编译安装 |

### 2.3 硬件信息

- Crazyflie 2.1：30 架
- 当前控制与仿真主案例：10 架
- 室内定位：2 台 Vive/SteamVR Lighthouse
- 无线通信：多台 Crazyradio PA

---

## 三、当前仓库结构

```text
crazyflie/
├── .venv/                       Python 虚拟环境
├── docs/                        开题材料与参考论文
├── outputs/
│   ├── figures/                 主仿真、随机测试、海报图输出
│   ├── videos/                  各动画场景 GIF / MP4
│   └── tuning/                  金字塔任务安全参数调优结果
├── src/
│   ├── afc_controller.py        AFC 控制器与输入饱和
│   ├── animate_sim.py           动画生成器
│   ├── archive.py               仿真归档工具
│   ├── collision_avoidance.py   CBF 安全滤波器
│   ├── disturbance_observer.py  风场模型与 ESO
│   ├── event_trigger.py         自适应事件触发通信
│   ├── formation.py             编队定义、仿射工具、RHF 辅助函数
│   ├── main_sim.py              主仿真与统一场景定义中心
│   ├── random_test.py           随机仿射压力测试 + 模块集成测试
│   ├── stress_matrix.py         应力矩阵与稀疏图/Power-Centric 设计
│   └── tune_pyramid_safety.py   金字塔任务安全参数自动调优
├── integration/
│   ├── config/
│   │   └── fleet_config.json    真机机群配置（URI、安全边界、坐标变换）
│   ├── logs/                    真机运行 CSV 日志（按时间戳命名）
│   └── scripts/
│       ├── pose_bridge.py       定位桥接（cflib stateEstimate → 算法坐标系）
│       ├── cf_command_bridge.py 命令下发桥接（速度/位置 setpoint）
│       ├── safety_guard.py      安全防护层（边界/间距/超时/限幅）
│       └── formation_runner.py  真机编队主控制循环（第一版接入入口）
├── animate_sim.py               根目录动画入口包装脚本
├── main_sim.py                  根目录主仿真入口包装脚本
├── random_test.py               根目录随机测试入口包装脚本
├── tune_pyramid_safety.py       根目录调参入口包装脚本
├── pyramid_mission_config.json  金字塔综合任务默认配置
└── PROJECT_DOC.md               当前总工作文档
```

### 结构变化说明

- src/main_sim.py 已成为“单一事实来源”，统一维护主仿真、动画场景定义、金字塔综合任务定义。
- src/animate_sim.py 不再各自复制场景逻辑，场景 1 到 6 都直接绑定 src/main_sim.py 中的场景构造或任务入口。
- src/random_test.py 已重构为统一测试入口，不再手工重复拼装老版本控制器和场景。
- 根目录的 main_sim.py、random_test.py、tune_pyramid_safety.py 只是包装脚本，便于在仓库根目录直接运行。

---

## 四、系统总览

当前仓库已经不只是“一个 AFC 基础仿真”，而是一个围绕 10 机三维仿射编队控制建立起来的完整仿真与验证框架，主要包含以下层次：

1. 基础 AFC 控制：应力矩阵、领航者-跟随者控制律、仿射不变性验证。
2. 工程约束建模：输入饱和、稀疏通信图。
3. 安全与鲁棒性：CBF 碰撞避免、ESO 抗扰补偿。
4. 通信优化：自适应事件触发通信。
5. 时变组织结构：RHF 层级重组、动态 Leader 切换、Power-Centric 拓扑。
6. 综合任务验证：地面起飞到空中金字塔的集成场景。
7. 支撑工具链：动画、参数调优、随机压力测试、模块集成测试、归档导出。

### 当前统一架构

- 场景定义中心：src/main_sim.py
- 渲染层：src/animate_sim.py
- 独立随机测试层：src/random_test.py
- 调参层：src/tune_pyramid_safety.py
- 配置层：pyramid_mission_config.json

这意味着：

- 动画和主仿真不再各写一套场景参数。
- random_test 现在也直接复用统一场景构造，而不是沿用旧的孤立脚本。
- 金字塔综合任务支持单独运行、单独调参、单独动画化、单独写回默认参数。

---

## 五、算法与理论基础

### 5.1 仿射编队控制 AFC

目标：只控制少量 Leader 的期望位置，其余 Follower 通过分布式协议自动收敛到同一仿射像。

对于标称构型 p0，期望构型可写为：

$$
p^* = (A \otimes I_d) p_0 + \mathbf{1}_n \otimes b
$$

其中 A 是仿射变换矩阵，b 是平移向量。

一阶积分器控制律：

$$
\dot{p}_f = -K_p(\Omega_{ff}p_f + \Omega_{fl}p_l)
$$

二阶积分器控制律：

$$
\ddot{p}_f = -K_p(\Omega_{ff}p_f + \Omega_{fl}p_l) - K_d\dot{p}_f
$$

稳态：

$$
p_f^* = -\Omega_{ff}^{-1}\Omega_{fl}p_l
$$

### 5.2 应力矩阵要求

应力矩阵 Ω 需要满足：

1. 对称性：Ω = Ω^T
2. 行和为零：Ω1 = 0
3. 应力平衡：Ωp0 = 0
4. rank(Ω) = n - d - 1
5. Ω_ff 正定

### 5.3 输入饱和

为匹配 Crazyflie 的速度/加速度约束，控制器支持三种饱和策略：

- smooth：tanh 平滑饱和
- norm：范数裁剪
- clip：逐分量裁剪

当前常用物理参数：

- 最大速度：1.0 m/s
- 最大加速度：5.0 m/s²

### 5.4 CBF 碰撞避免

对每一对智能体定义障碍函数：

$$
h_{ij}(p) = \|p_i - p_j\|^2 - d_s^2
$$

一阶系统安全条件：

$$
\dot{h}_{ij} \ge -\gamma h_{ij}
$$

实现方式是在标称 AFC 控制输入外再套一层 QP 安全滤波器，尽可能小地修改原控制输入。

### 5.5 ESO 抗扰补偿

考虑外部扰动 w(t)：

$$
\dot{p}_f = u_f + w_f(t)
$$

ESO 通过扩展状态观测器同时估计位置和总扰动，再把扰动估计前馈补偿到控制输入中。

### 5.6 自适应事件触发通信

二阶模型中使用广播位置 p_hat 替代连续实时邻居位置，并使用自适应阈值决定何时广播。

目标：在误差可接受的前提下，大幅降低通信负担。

### 5.7 RHF 层级重组

参考 Power-Centric 思路，在不同阶段切换 Leader 组合和拓扑，同时保持仿射可定位性与收敛性。

关键条件：

- Leader 必须仿射张成 R^d
- 各阶段拓扑要保证 Ω_ff 可逆/正定
- 切换间隔需满足驻留时间约束

---

## 六、核心代码模块

### 6.1 src/stress_matrix.py

职责：

- 常规 SDP 应力矩阵求解
- 稀疏图设计
- Power-Centric 拓扑对应力矩阵求解
- 性质验证

主要能力：

- compute_stress_matrix()
- compute_sparse_stress_matrix()
- compute_power_centric_stress_matrix()
- validate_stress_matrix()

### 6.2 src/afc_controller.py

职责：

- 一阶/二阶 AFC 控制器
- 输入饱和
- 稳态解计算
- 收敛速率评估
- 在线更新 Ω 和 Leader/Follower 角色

关键点：

- AFCController.update_omega() 已支持 RHF 场景中的动态切换。

### 6.3 src/formation.py

职责：

- 标称编队生成
- 仿射变换矩阵工具
- 轨迹生成
- RHF 相关辅助函数

当前关键编队：

- double_pentagon()：10 机反棱柱基线编队
- aerial_pyramid_10()：10 机空中金字塔编队

关键辅助函数：

- affine_transform()
- scale_matrix()
- rotation_matrix_z()
- rotation_matrix_axis()
- shear_matrix_3d()
- create_leader_trajectory()
- check_affine_span()
- build_power_centric_topology()
- select_leaders_for_direction()
- compute_dwell_time()

### 6.4 src/collision_avoidance.py

职责：

- CBF 安全滤波
- 最小距离与成对距离分析

关键类：

- CBFSafetyFilter

### 6.5 src/disturbance_observer.py

职责：

- 恒定风 + OU 阵风扰动建模
- ESO 状态更新与扰动估计

关键类：

- WindDisturbance
- ExtendedStateObserver

### 6.6 src/event_trigger.py

职责：

- 自适应事件触发广播管理
- 触发记录和通信率统计

关键类：

- EventTriggerManager

### 6.7 src/main_sim.py

这是当前代码架构的中心文件，职责包括：

1. 基础一阶、二阶、CBF、ESO、ET、RHF、综合任务仿真引擎。
2. 主论文图输出。
3. 统一动画场景定义。
4. 金字塔综合任务配置读写。
5. 单独场景运行入口。

当前新增的统一能力：

- build_base_animation_setup()
- build_baseline_animation_scenario()
- build_cbf_animation_scenario()
- build_eso_animation_scenario()
- build_et_animation_scenario()
- build_rhf_animation_scenario()
- run_pyramid_integrated_mission()
- load_pyramid_mission_config()
- save_pyramid_mission_config()

CLI：

- python main_sim.py --scenario all
- python main_sim.py --scenario pyramid

### 6.8 src/animate_sim.py

职责：

- 把统一场景渲染成 GIF / MP4 / poster
- 场景 1 到 6 均直接依赖 src/main_sim.py 中的场景定义或综合任务入口

当前场景编号：

| 编号 | 场景 | 来源 |
|------|------|------|
| 1 | baseline | build_baseline_animation_scenario() |
| 2 | cbf | build_cbf_animation_scenario() |
| 3 | eso | build_eso_animation_scenario() |
| 4 | et | build_et_animation_scenario() |
| 5 | rhf | build_rhf_animation_scenario() |
| 6 | pyramid mission | run_pyramid_integrated_mission() |

CLI：

- python animate_sim.py --scenario 0
- python animate_sim.py --scenario 1
- python animate_sim.py --scenario 6

其中 0 表示生成全部场景，1 到 6 分别生成单场景。

### 6.9 src/random_test.py

该文件已于 2026年3月6日彻底重构，不再是旧的“独立脚本”，而是统一测试入口。

当前职责分为两类：

1. 随机仿射压力测试
2. 模块集成测试

随机仿射测试覆盖：

- 均匀/非均匀缩放
- 绕 z 轴旋转
- 绕任意轴旋转
- 剪切
- 反射
- 一般仿射组合

模块集成测试覆盖：

- baseline
- cbf
- eso
- et
- rhf
- mission

CLI：

- python random_test.py --mode all
- python random_test.py --mode affine
- python random_test.py --mode modules
- python random_test.py --mode affine --trials 20 --no-show

### 6.10 src/tune_pyramid_safety.py

职责：

- 对金字塔综合任务的安全参数进行网格搜索
- 自动比较 d_safe、d_activate、cbf_gamma
- 输出 CSV / JSON 排名结果
- 自动把最佳参数写回 pyramid_mission_config.json

CLI：

- python tune_pyramid_safety.py
- python tune_pyramid_safety.py --quick
- python tune_pyramid_safety.py --no-write

### 6.11 integration/（真机接入层）

**第一版接入范围**：AFC + 速度限幅 + 基础安全保护（CBF/ESO/ET/RHF 留第二阶段）

| 文件 | 职责 |
|------|------|
| `pose_bridge.py` | 订阅 cflib stateEstimate 日志，整理成 (n,3) 位置/速度矩阵；线程安全缓存；坐标系变换 |
| `cf_command_bridge.py` | 复用 PoseBridge 连接句柄，下发速度 setpoint；起飞/悬停/降落/紧急停止；watchdog keepalive |
| `safety_guard.py` | 定位新鲜度、边界越界、最小间距、位置误差、速度限幅五项检查；返回 SafetyStatus（SAFE/HOVER/EMERGENCY）|
| `formation_runner.py` | 主控制循环：连接→等待定位→起飞→20Hz 闭环→安全关闭；CSV 日志；键盘紧急停止 |
| `fleet_config.json` | 机群 URI、控制频率、安全参数、坐标变换、标称编队位置 |

**运行方式**：
```
cd e:/crazyflie
python integration/scripts/formation_runner.py
```

**已知局限（第一版）**：
- 标称编队为 3 机等腰三角形（1 Leader + 2 Follower），10 机版需更新 fleet_config.json
- 使用完全图应力矩阵（保守策略），大机群可改为 compute_sparse_stress_matrix
- CBF/ESO/ET/RHF 尚未接入真机闭环

---

## 七、配置文件

### 7.1 pyramid_mission_config.json

该文件用于保存综合金字塔任务的默认参数，当前仓库中的值为：

```json
{
  "gain": 5.5,
  "d_safe": 0.18,
  "d_activate": 0.9,
  "cbf_gamma": 7.0,
  "total_time": 34.0,
  "init_radius": 1.9,
  "init_noise_std": 0.04,
  "rng_seed": 2026,
  "wind_const": [0.08, -0.03, 0.02],
  "ou_theta": 0.4,
  "ou_sigma": 0.04,
  "wind_seed": 321,
  "eso_omega": 8.0,
  "et_mu": 0.015,
  "et_varpi": 0.45,
  "et_phi_0": 1.0
}
```

说明：

- 该配置是 tune_pyramid_safety.py 自动写回后的当前默认值。
- main_sim.py 在运行金字塔任务时会优先读取此文件。

---

## 八、输出产物

### 8.1 outputs/figures

当前主要图像产物包括：

- fig1_formation_snapshots.png
- fig2_trajectories_3d.png
- fig3_convergence.png
- fig4_stress_matrix.png
- fig5_communication_graph.png
- fig6_sparse_comparison.png
- fig7_saturation_analysis.png
- fig8_saturation_comparison.png
- fig9_cbf_collision_avoidance.png
- fig10_cbf_analysis.png
- fig11_eso_disturbance_rejection.png
- fig12_eso_bandwidth_comparison.png
- fig13_et_error_comparison.png
- fig14_et_communication_analysis.png
- fig15_et_parameter_comparison.png
- fig16_rhf_3d_trajectory.png
- fig17_rhf_error_evolution.png
- fig18_rhf_topology.png
- fig19_pyramid_mission_snapshots.png
- fig20_pyramid_mission_metrics.png
- fig21_pyramid_rhf_topology.png
- fig_rt_single.png
- fig_rt_monte_carlo.png
- fig_rt_modules.png

另有部分动画 poster 图：

- afc_scene1_baseline_poster.png
- afc_scene3_eso_poster.png
- afc_scene6_pyramid_mission_poster.png

### 8.2 outputs/videos

当前已有动画文件：

- afc_animation.gif / afc_animation.mp4
- afc_scene1_baseline.gif / .mp4
- afc_scene2_cbf.gif / .mp4
- afc_scene3_eso.gif / .mp4
- afc_scene4_et.gif / .mp4
- afc_scene5_rhf.gif / .mp4
- afc_scene6_pyramid_mission.gif / .mp4

### 8.3 outputs/tuning

金字塔任务安全参数调优结果以时间戳形式存放，当前目录已有：

- pyramid_safety_tuning_20260306_172711.csv / .json
- pyramid_safety_tuning_20260306_172926.csv / .json

---

## 九、当前已实现能力清单

### 9.1 基础控制与图设计

- 10 机三维反棱柱仿射编队
- SDP 应力矩阵求解
- 应力矩阵 5 条性质验证
- 稀疏通信图设计

### 9.2 工程约束与安全

- 输入饱和约束
- CBF 碰撞避免
- Crazyflie 速度/加速度参数映射

### 9.3 鲁棒与通信优化

- ESO 风扰补偿
- 事件触发通信
- 触发率与累计通信统计

### 9.4 动态组织与复杂任务

- RHF 层级重组
- 动态 Leader 切换
- Power-Centric 拓扑
- 地面起飞到空中金字塔综合任务

### 9.5 工具链

- 主仿真论文图生成
- 多场景动画生成
- 综合任务单独运行
- 综合任务安全参数自动调优
- 随机仿射压力测试
- 模块集成测试

---

## 十、当前验证结果

### 10.1 稀疏图设计

既有主结果：

- 完全图 45 条边
- 稀疏图 29 条边
- 边数减少 35.6%
- rank(Ω) = 6
- λ_min(Ω_ff) 约为 0.0238

说明：

- 该结果仍是当前主仿真基线的重要工程约束来源。

### 10.2 CBF 结果

既有记录：

- 无 CBF 最小距离约 0.1072 m，存在碰撞风险
- 有 CBF 最小距离约 0.2250 m，达到安全要求
- 编队误差代价很小

### 10.3 ESO 结果

既有记录：

- 有扰动无 ESO 时误差显著增大
- 加入 ESO 后误差明显下降
- 典型记录中误差降低约 82.6%

### 10.4 ET 结果

既有记录：

- 连续通信：100%
- 事件触发平均通信率约 3.88%
- 通信节省约 96.1%
- 误差仅小幅增加

### 10.5 RHF 结果

既有记录：

- 3 阶段 U 形转弯层级切换可稳定完成
- 各阶段满足驻留时间条件
- Power-Centric 拓扑在切换后立即可用

### 10.6 金字塔综合任务最新结果

2026年3月最新已验证结果：

- 运行入口：python main_sim.py --scenario pyramid
- 当前默认安全参数：d_safe=0.18, d_activate=0.9, cbf_gamma=7.0
- 最终误差：0.3368
- 最小间距：0.1684 m
- 平均通信率：8.77%

说明：

- 该场景集成了 AFC + 输入饱和 + CBF + ESO + ET + RHF。
- 目前已支持单独运行、单独动画化、单独调参。

### 10.7 random_test 最新结果

2026年3月最新冒烟验证结果：

随机仿射模式：

- 单次复杂组合变换最终误差：0.4089
- 单次复杂组合最小间距：0.2484 m
- 仿射不变性误差：1.36e-14

Monte Carlo 小规模验证：

- 2 次试验最终误差均值：0.1845 ± 0.0002
- 平均最小间距：0.2510 m
- 全部收敛

模块集成模式：

| 模块 | final_err | d_min | mean_comm_rate |
|------|-----------|-------|----------------|
| baseline | 0.1237 | 0.6368 | NaN |
| cbf | 0.1231 | 0.2266 | NaN |
| eso | 0.1102 | 0.6368 | NaN |
| et | 0.4961 | 0.6368 | 3.98% |
| rhf | 0.0037 | 0.1535 | NaN |
| mission | 0.3368 | 0.1684 | 8.77% |

说明：

- 这组结果来自重构后的 random_test.py 实际运行。
- 说明 unified random test harness 已经和主场景结构正确接通。

---

## 十一、开发进展与最近重构

### 11.1 已完成的阶段

1. Crazyflie / WSL2 / ROS 2 / Webots 基础环境搭建。
2. AFC 主控制框架实现。
3. 稀疏图设计与应力矩阵验证实现。
4. 输入饱和、CBF、ESO、ET、RHF 各模块实现并接入主仿真。
5. 场景 1 到 6 的动画输出实现。
6. 金字塔综合任务实现。
7. 综合任务安全参数调优脚本实现。
8. random_test 彻底重构为统一测试入口。
9. **integration/ 真机接入层第一版实现**（2026年3月14日）。

### 11.2 2026年3月14日：真机接入层代码质量修复

对 `integration/scripts/` 进行了代码质量审查，修复以下问题：

| 优先级 | 文件 | 问题 | 修复 |
|--------|------|------|------|
| 高 | `formation_runner.py` | `_emergency_land()` 未设 `self._emergency=True`，导致 `_shutdown()` 可能二次降落 | 补加标志位 |
| 高 | `pose_bridge.py` | `get_latest_state()` 用 drone id 直接作数组行索引，id 不连续时越界 | 新增 `_id_to_row` 映射 |
| 高 | `cf_command_bridge.py` | MOCK 模式 `send_follower_velocities()` 静默返回，调试不可见 | 改为 `logger.debug` |
| 中 | `safety_guard.py` | `_check_min_distance()` 内层循环调用 `np.linalg.norm`，可向量化 | 改用广播预计算距离矩阵 |
| 中 | `pose_bridge.py` | `start()` 连接失败时已建立的连接未清理 | 失败时调用 `self.stop()` |
| 中 | `fleet_config.json` | `control.u_max` 与 `safety.max_velocity_mps` 重复，代码只读后者 | 废弃前者并加注释 |

### 11.3 2026年3月的关键重构

本轮重构的核心不是“再加几个功能”，而是统一代码架构：

1. 金字塔综合任务支持单独运行，不必每次全量跑 main_sim。
2. 新增 tune_pyramid_safety.py，对安全参数自动搜索并写回配置。
3. 新增 pyramid_mission_config.json，作为综合任务默认配置源。
4. animate_sim.py 场景 1 到 6 改为完全绑定 src/main_sim.py 的场景定义。
5. random_test.py 改为统一测试入口，接入主场景结构和综合任务。
6. 根目录主入口脚本补齐，所有常用功能都可以直接在仓库根目录运行。

### 11.3 当前代码状态判断

当前仓库已经从“单脚本实验代码”转向“可维护的研究型仿真框架”：

- 主场景定义集中化
- 动画/测试/调参共用同一套配置和场景来源
- 金字塔任务可独立复现实验
- 随机测试已从历史遗留脚本切换到现架构

---

## 十二、当前局限性

虽然功能已经较完整，但仍有以下问题没有解决：

1. 动力学仍以积分器近似为主，尚未接入更接近 Crazyflie 真机的完整飞行动力学。
2. 应力矩阵求解仍是集中式离线设计，不是完全分布式在线估计。
3. 通信模型仍未显式考虑延迟、丢包、带宽抖动和时钟偏差。
4. Leader 默认仍假设能较好跟踪参考轨迹，未建模高层轨迹跟踪误差。
5. 综合任务的最小间距虽然提升到了 0.1684 m，但仍低于 0.18 m 这一当前配置阈值，说明 CBF 仍有进一步调参空间。
6. 实物飞行验证尚未闭环接到当前这套仿真框架。

---

## 十三、开题与毕设要求对照

### 13.1 要求与当前实现不一致的地方

1. 开题与任务书要求最终落点是 Crazyflie 真机实验系统，但当前仓库的主实现仍以 Python 仿真框架为主，尚未形成可直接部署的真机控制工程。
2. 开题报告中强调了 ROS 2、Crazyswarm2 或 CrazyChoir、Webots、rosbag 等工程链路，但当前仓库中没有对应的 launch 文件、节点代码、实验记录或 rosbag 数据，说明这一部分至少没有在本仓库内闭环落地。
3. 课题要求里有室内定位系统精度与更新频率指标，但当前仓库没有 Lighthouse 或 LPS 标定脚本、定位误差统计、更新频率日志，无法证明该指标已经达成。
4. 课题要求的是“单机、多机实物飞行实验”，当前仓库虽然有完整仿真、动画、调参与随机测试，但仍缺少真机实验数据、实验记录和误差统计。
5. 开题报告将 RMSE、收敛时间、通信流量、CPU 负载作为性能评价维度，当前仓库已经覆盖误差曲线、通信率和部分收敛过程，但没有系统性的 RMSE 汇总和 CPU 负载测试。
6. 当前总文档中写了 WSL2、ROS 2、CrazyChoir、Webots 环境信息，但这些更多反映开发背景，不等于本仓库已经包含对应的运行工程或实验结果。

### 13.2 按任务书逐项判断

| 任务 | 当前状态 | 判断 |
|------|----------|------|
| Crazyflie 平台搭建 | 文档中有硬件、软件环境描述，但仓库内缺少平台联调记录与配置证据 | 部分满足 |
| 室内定位系统 | 仅有设备说明，无定位误差和频率测试结果 | 未证明满足 |
| 三维仿射编队控制算法 | AFC、稀疏图、CBF、ESO、ET、RHF、综合任务均已实现，且复杂仿射变换验证充分 | 基本满足 |
| 单机/多机飞行实验 | 当前仍以仿真为主，缺真机实验闭环 | 未满足 |
| 毕业论文撰写支撑 | 代码、图表、动画、调参和工作文档已具备较好素材基础 | 可支撑，但实验部分仍需补齐 |

### 13.3 当前算法能否满足课题要求

结论分两层看：

1. 如果只看“算法研究”这一层，当前实现已经明显超过最初的基础 AFC 目标。除三维仿射编队外，还加入了输入饱和、CBF、ESO、ET、RHF 和综合金字塔任务，算法仿真部分是成立的。
2. 如果按“整个毕设任务书”来判断，当前还不能说已经满足，因为定位系统指标和真机飞行实验指标都没有在仓库中被验证闭环。

### 13.4 对关键指标的保守判断

- 理论推导误差 ≤ 10%：从当前仿射不变性误差接近机器精度、多个场景稳定收敛来看，算法仿真层面可以认为满足，且结果强于该要求。
- 定位误差 ≤ 0.5 m、更新频率 ≥ 2 Hz：当前没有实测数据，不能认定满足。
- 多机编队位置偏差 ≤ 0.3 m：当前没有真机数据，而且现有综合任务的 final_error 为 0.3368，这个指标口径也和“真机编队位置偏差”不完全等价，因此不能认定满足。

### 13.5 更准确的当前结论

更稳妥的表述应该是：

- 当前已经完成了毕设中“算法设计、仿真验证、可视化分析、工程约束扩展”这一大块。
- 当前尚未完成或尚未提供证据的，是“定位系统定量验证”和“单机/多机真机飞行实验”。
- 所以现在最合适的结论不是“整题已满足”，而是“算法仿真部分基本完成，工程实验部分仍缺关键证据”。

---

## 十四、后续建议

优先级建议如下：

1. 继续调优金字塔综合任务，把最小间距再往上推，并验证对误差和通信率的影响。
2. 把 random_test 的模块集成结果导出为 CSV/JSON，方便论文表格直接引用。
3. 给主仿真与动画增加更规范的实验编号和批量输出清单，便于论文插图管理。
4. 推进 CrazyChoir / ROS 2 对接，把当前统一场景定义逐步迁移到真实实验链路。
5. 引入更真实的二阶或非线性四旋翼模型，用于验证当前控制逻辑在真实动力学下的鲁棒性。

---

## 十五、常用命令

在仓库根目录执行：

```powershell
# 主仿真：全流程
e:/crazyflie/.venv/Scripts/python.exe main_sim.py --scenario all

# 主仿真：只跑综合金字塔任务
e:/crazyflie/.venv/Scripts/python.exe main_sim.py --scenario pyramid

# 动画：只生成场景 6
e:/crazyflie/.venv/Scripts/python.exe animate_sim.py --scenario 6

# 动画：生成全部场景
e:/crazyflie/.venv/Scripts/python.exe animate_sim.py --scenario 0

# 随机测试：全部
e:/crazyflie/.venv/Scripts/python.exe random_test.py --mode all

# 随机测试：仅模块集成
e:/crazyflie/.venv/Scripts/python.exe random_test.py --mode modules --no-show

# 随机测试：仅随机仿射测试
e:/crazyflie/.venv/Scripts/python.exe random_test.py --mode affine --trials 20 --no-show

# 金字塔任务安全参数快速调优
e:/crazyflie/.venv/Scripts/python.exe tune_pyramid_safety.py --quick
```

---

## 十六、参考文献与资料目录

docs/ 目录下保存了当前课题的主要资料，包括：

- 毕设信息.txt
- 开题报告.docx
- 仿射编队控制基础论文 affine.pdf / Affine (1).pdf
- CrazyChoir、Crazyswarm 相关论文
- Joint Estimation and Planar Affine Formation Control 相关论文
- Reconfigurable Leader-Follower Swarm Formation 相关论文
- 多篇 2024 至 2025 年的最新 arXiv / 期刊文献

建议用途：

- 理论推导优先参考 affine.pdf 和相关硕士论文。
- 动态重组、复杂环境、通信约束优先参考较新的 arXiv 文献。
- 真机系统设计与 ROS 2 对接优先参考 CrazyChoir / Crazyswarm 文献。

---

## 十七、结论

截至 2026年3月6日，当前仓库已经形成了一个围绕 Crazyflie 三维仿射编队控制的较完整研究框架。基础 AFC、稀疏图设计、输入饱和、CBF、ESO、ET、RHF 以及金字塔综合任务均已实现，动画、调参、随机测试和模块集成测试也已经接入统一架构。当前最重要的后续工作，不再是继续“堆功能”，而是把这套统一仿真框架进一步整理成可复现实验链路，并逐步对接真机实验与论文写作。
