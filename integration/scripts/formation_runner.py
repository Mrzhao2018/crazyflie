"""
formation_runner.py - 真机编队主控制循环

概览：
  本脚本是第一版真机闭环的入口。它把 pose_bridge（定位输入）、
  afc_controller（编队算法）、safety_guard（安全检查）、
  cf_command_bridge（命令下发）串联成一个完整在线闭环。

第一版接入范围：
  AFC + 速度限幅 + 基础安全保护
  （CBF、ESO、ET、RHF 留在第二阶段接入）

运行方式：
  cd e:/crazyflie
  python integration/scripts/formation_runner.py

键盘控制：
  Enter  → 进入控制循环（起飞并启动编队控制）
  q      → 紧急降落并退出

日志文件：
  integration/logs/run_<timestamp>.csv
"""

import sys
import os
import time
import json
import logging
import threading
import csv
import numpy as np

# ─────────────────────────────────────────────────────────
# 把仓库根目录和 src/ 加入 Python 路径
# ─────────────────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
_SCRIPTS_DIR = os.path.dirname(__file__)

for _p in [_REPO_ROOT, _SRC_DIR, _SCRIPTS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from afc_controller import AFCController  # src/afc_controller.py
from stress_matrix import compute_stress_matrix  # src/stress_matrix.py
from formation import create_leader_trajectory, smoothstep  # src/formation.py
from pose_bridge import PoseBridge  # integration/scripts/pose_bridge.py
from cf_command_bridge import CommandBridge  # integration/scripts/cf_command_bridge.py
from safety_guard import (
    SafetyGuard,
    SafetyStatus,
)  # integration/scripts/safety_guard.py

# ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("formation_runner")


# ─────────────────────────────────────────────────────────
# 配置加载
# ─────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────
# 主类
# ─────────────────────────────────────────────────────────


class FormationRunner:
    """
    真机编队主控制器。

    Parameters
    ----------
    config_path : str
        fleet_config.json 的路径
    """

    def __init__(self, config_path: str):
        self._cfg = load_config(config_path)
        self._running = False
        self._emergency = False

        # ── 基础参数 ──
        ctrl = self._cfg["control"]
        self._freq = float(ctrl["freq_hz"])
        self._dt = 1.0 / self._freq
        self._gain = float(ctrl["gain"])
        self._u_max = float(self._cfg["safety"]["max_velocity_mps"])
        self._saturation_type = ctrl.get("saturation_type", "norm")
        algo_cfg = self._cfg.get("algorithm_integration", {})
        self._algorithm_mode = str(algo_cfg.get("mode", "dry_run")).lower()
        allowed_modes = tuple(
            algo_cfg.get(
                "allowed_modes", ["dry_run", "live_single_follower", "live_group"]
            )
        )
        if self._algorithm_mode not in allowed_modes:
            raise ValueError(
                f"Unsupported control.algorithm_mode={self._algorithm_mode!r}; "
                f"expected one of {allowed_modes}"
            )
        self._planar_only = bool(algo_cfg.get("planar_only", True))
        self._allow_vertical_velocity = bool(
            algo_cfg.get("allow_vertical_velocity", False)
        )
        self._allow_yaw_rate = bool(algo_cfg.get("allow_yaw_rate", False))
        self._max_xy_velocity = float(algo_cfg.get("max_xy_velocity_mps", 0.10))
        self._max_z_velocity = float(algo_cfg.get("max_z_velocity_mps", 0.0))
        self._max_live_duration = float(algo_cfg.get("max_live_duration_s", 20.0))

        # ── 编队参数 ──
        formation_cfg = self._cfg["formation"]
        self._n = formation_cfg["n_agents"]
        self._leader_indices = list(formation_cfg["leader_indices"])
        self._follower_indices = sorted(set(range(self._n)) - set(self._leader_indices))

        # ── 标称位置（用于应力矩阵计算和位置误差检查）──
        nominal_raw = self._cfg["nominal_formation"]["positions"]
        self._nominal_pos = np.array(nominal_raw, dtype=float)  # (n, 3)

        # ── 应力矩阵与 AFC 控制器 ──
        self._omega = self._build_stress_matrix()
        self._afc = AFCController(
            stress_matrix=self._omega,
            leader_indices=self._leader_indices,
            gain=self._gain,
            u_max=self._u_max,
            saturation_type=self._saturation_type,
        )

        # ── 安全层 ──
        self._guard = SafetyGuard(self._cfg)

        # ── 通信桥 ──
        self._pose_bridge = PoseBridge(self._cfg)
        # CommandBridge 通过共享连接句柄初始化（避免双重连接）
        self._cmd_bridge: CommandBridge | None = None

        # ── 日志 ──
        log_cfg = self._cfg.get("logging", {})
        self._log_enabled = log_cfg.get("enabled", True)
        self._log_dir = os.path.join(
            _REPO_ROOT, log_cfg.get("log_dir", "integration/logs")
        )
        self._log_interval = log_cfg.get("log_interval_steps", 1)
        os.makedirs(self._log_dir, exist_ok=True)
        self._log_file = None
        self._csv_writer = None

        # ── 紧急停止键盘监听线程 ──
        self._kbd_thread = threading.Thread(target=self._keyboard_listener, daemon=True)

    # ─────────────────────────────────────────
    # 应力矩阵构建
    # ─────────────────────────────────────────

    def _build_stress_matrix(self) -> np.ndarray:
        """
        从标称编队位置计算应力矩阵 Omega。

        使用完全图确保应力矩阵一定存在（第一版保守策略）。
        若编队较大（>6 机），可改为 compute_sparse_stress_matrix 以减少通信量。
        """
        n = self._n
        # 完全图邻接矩阵
        adj = np.ones((n, n), dtype=int) - np.eye(n, dtype=int)

        logger.info("[FormationRunner] 计算应力矩阵（完全图）...")
        try:
            omega, sm_info = compute_stress_matrix(
                self._nominal_pos, adj, self._leader_indices
            )
            logger.info(
                f"[FormationRunner] SDP info: null_dim={sm_info.get('null_dim')}, "
                f"n_leaders={sm_info.get('n_leaders')}, n_followers={sm_info.get('n_followers')}"
            )
            # 验证 Omega_ff 正定
            ff_idx = self._follower_indices
            omega_ff = omega[np.ix_(ff_idx, ff_idx)]
            eigvals = np.linalg.eigvalsh(omega_ff)
            logger.info(
                f"[FormationRunner] 应力矩阵计算成功，"
                f"Omega_ff min eigval = {eigvals.min():.4f}"
            )
        except Exception as e:
            logger.error(f"[FormationRunner] 应力矩阵计算失败: {e}")
            raise
        return omega

    def _build_leader_trajectory(self, leader_positions: np.ndarray):
        """
        基于 Leader 起飞后的实际位置，构建仿射变换演示轨迹。

        阶段：悬停 → 平移 → 缩放 → 旋转 → 复位

        Parameters
        ----------
        leader_positions : (n_l, 3) Leader 起飞后的实际位置
        """
        mission = self._cfg.get("mission", {})
        if mission.get("type") != "affine_demo":
            self._leader_traj = None
            return

        t_settle = mission.get("settle_time_s", 8.0)
        t_trans = mission.get("transition_time_s", 5.0)
        t_hold = mission.get("hold_time_s", 5.0)
        translate = np.array(mission.get("translate_m", [0.2, 0.0, 0.0]))
        scale = mission.get("scale_factor", 0.7)
        rotate_deg = mission.get("rotate_deg", 30.0)

        p0 = leader_positions.copy()  # (n_l, 3)
        center = p0.mean(axis=0)  # 编队中心

        # 阶段 1：平移
        p1 = p0 + translate

        # 阶段 2：以中心为原点缩放
        center1 = p1.mean(axis=0)
        p2 = center1 + scale * (p1 - center1)

        # 阶段 3：绕 z 轴旋转（保持缩放）
        theta = np.radians(rotate_deg)
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        center2 = p2.mean(axis=0)
        p3 = center2 + (p2 - center2) @ R.T

        # 阶段 4：复位
        p4 = p0.copy()

        t = 0.0
        phases = []

        # Phase 0: 悬停等收敛
        phases.append(
            {
                "start_positions": p0,
                "t_start": t,
                "t_end": t + 0.1,
                "positions": p0.copy(),
            }
        )
        t += t_settle

        # Phase 1: 平移
        phases.append({"t_start": t, "t_end": t + t_trans, "positions": p1})
        t += t_trans + t_hold

        # Phase 2: 缩放
        phases.append({"t_start": t, "t_end": t + t_trans, "positions": p2})
        t += t_trans + t_hold

        # Phase 3: 旋转
        phases.append({"t_start": t, "t_end": t + t_trans, "positions": p3})
        t += t_trans + t_hold

        # Phase 4: 复位
        phases.append({"t_start": t, "t_end": t + t_trans, "positions": p4})

        self._leader_traj = create_leader_trajectory(phases)

        # 安全检查：所有轨迹关键点是否在边界内
        boundary = self._cfg["safety"]["boundary"]
        for label, pts in [("平移", p1), ("缩放", p2), ("旋转", p3)]:
            for i, p in enumerate(pts):
                if (
                    p[0] < boundary["x_min"]
                    or p[0] > boundary["x_max"]
                    or p[1] < boundary["y_min"]
                    or p[1] > boundary["y_max"]
                    or p[2] < boundary["z_min"]
                    or p[2] > boundary["z_max"]
                ):
                    logger.warning(
                        f"[FormationRunner] 轨迹越界! {label} leader {i}: "
                        f"({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})"
                    )

        total_t = t + t_trans
        logger.info(f"[FormationRunner] 仿射演示轨迹已生成 (总时长 {total_t:.0f}s)")
        logger.info(
            f"  悬停 {t_settle}s → 平移 {translate} → "
            f"缩放 {scale}x → 旋转 {rotate_deg}° → 复位"
        )

    # ─────────────────────────────────────────
    # 控制循环
    # ─────────────────────────────────────────

    def run(self):
        """
        主入口：起飞 → 等待就绪 → 进入控制循环。
        按 Enter 开始，按 q 紧急退出。
        """
        logger.info("=" * 60)
        logger.info("  Crazyflie 编队控制器 — 第一版真机接入")
        logger.info(f"  编队: {self._n} 架  Leaders: {self._leader_indices}")
        logger.info(f"  控制频率: {self._freq} Hz   速度上限: {self._u_max} m/s")
        logger.info(f"  算法模式: {self._algorithm_mode}")
        logger.info("=" * 60)

        # 建立连接
        logger.info("[FormationRunner] 建立 Crazyflie 连接...")
        try:
            self._pose_bridge.start()
        except Exception as e:
            logger.error(f"[FormationRunner] 连接失败，请检查硬件: {e}")
            return

        # 共享连接句柄给 CommandBridge（通过公开接口，避免双重无线连接）
        self._cmd_bridge = CommandBridge(
            self._cfg, self._pose_bridge.get_cf_connections()
        )

        # 等待定位数据就绪
        logger.info("[FormationRunner] 等待定位系统就绪（最多 30s）...")
        if not self._pose_bridge.wait_until_fresh(timeout_s=30.0):
            logger.error("[FormationRunner] 定位数据获取失败，退出")
            self._pose_bridge.stop()
            return

        # 显示当前位置 & 预检查
        state = self._pose_bridge.get_latest_state()
        positions = state["positions"]
        logger.info("[FormationRunner] 当前飞机位置:")
        for did, d in state["per_drone"].items():
            p = d["pos"]
            logger.info(f"  drone {did}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")

        # 起飞前预检查：XY 边界和最小间距
        b = self._cfg["safety"]["boundary"]
        preflight_warnings = []
        for i in range(self._n):
            x, y, z = positions[i]
            if x < b["x_min"] or x > b["x_max"] or y < b["y_min"] or y > b["y_max"]:
                preflight_warnings.append(f"drone {i} XY 超出边界: ({x:.2f}, {y:.2f})")
        dists = np.linalg.norm(
            positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1
        )
        d_safe = self._cfg["safety"]["d_safe_m"]
        for i in range(self._n):
            for j in range(i + 1, self._n):
                if dists[i, j] < d_safe:
                    preflight_warnings.append(
                        f"drone {i}-{j} 间距 {dists[i, j]:.3f}m < d_safe {d_safe}m"
                    )
        if preflight_warnings:
            logger.warning("[FormationRunner] 起飞前预检查警告:")
            for w in preflight_warnings:
                logger.warning(f"  ⚠ {w}")
            logger.warning("  请确认飞机位置是否正确再继续")

        # 等待用户确认起飞
        input("\n按 Enter 键全机起飞并启动编队控制... (Ctrl+C 取消)\n")

        # 起飞
        logger.info("[FormationRunner] 起飞...")
        self._cmd_bridge.takeoff_all()

        # 验证起飞高度（轮询等待位置数据刷新）
        min_takeoff_z = 0.15  # 至少离地 15cm 才算起飞成功
        max_wait = 5.0  # 最多再等 5 秒
        poll_interval = 0.5
        elapsed = 0.0
        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval
            state = self._pose_bridge.get_latest_state()
            positions = state["positions"]
            failed_drones = []
            for i in range(self._n):
                z = positions[i, 2]
                if z < min_takeoff_z:
                    failed_drones.append((i, z))
            if not failed_drones:
                logger.info(f"[FormationRunner] 全部起飞确认 ({elapsed:.1f}s)")
                break
            logger.info(
                f"[FormationRunner] 等待起飞... ({elapsed:.1f}s) "
                f"未达标: {[(d, f'{z:.3f}m') for d, z in failed_drones]}"
            )
        if failed_drones:
            for did, z in failed_drones:
                logger.error(f"[FormationRunner] drone {did} 起飞失败! z={z:.3f}m")
            logger.error("[FormationRunner] 有飞机未成功起飞，执行降落")
            self._cmd_bridge.land_all()
            self._pose_bridge.stop()
            return

        # 锁定 Leader 位置（防止 takeoff 轨迹到期后漂移）
        state = self._pose_bridge.get_latest_state()
        self._cmd_bridge.lock_leader_positions(state["positions"])

        # 构建 Leader 仿射演示轨迹（如果配置了 mission）
        leader_pos = state["positions"][self._leader_indices]
        self._build_leader_trajectory(leader_pos)

        # 启动键盘监听
        self._kbd_thread.start()

        # 打开日志文件
        if self._log_enabled:
            self._open_log()

        # 进入控制循环
        self._running = True
        logger.info("[FormationRunner] 编队控制循环启动 (按 q 紧急降落)")
        try:
            self._control_loop()
        except KeyboardInterrupt:
            logger.info("[FormationRunner] 用户中断")
        finally:
            self._shutdown()

    def _control_loop(self):
        """固定频率控制循环。"""
        step = 0
        t_start = time.time()

        while self._running and not self._emergency:
            t_loop_start = time.time()

            # 1. 读取当前状态
            state = self._pose_bridge.get_latest_state()
            positions = state["positions"]  # (n, 3)

            # 1.5 更新 Leader 轨迹目标（如果有仿射演示轨迹）
            t_elapsed = time.time() - t_start
            if self._leader_traj is not None:
                leader_targets = self._leader_traj(t_elapsed)  # (n_l, 3)
                for i, lid in enumerate(self._leader_indices):
                    self._cmd_bridge.update_leader_target(
                        lid,
                        float(leader_targets[i, 0]),
                        float(leader_targets[i, 1]),
                        float(leader_targets[i, 2]),
                    )

            # 2. 计算 AFC 速度命令（仅 follower）
            #    leaders 的位置来自真实定位，不需要主动控制
            u_f = self._afc.all_follower_inputs(positions)  # (n_f, 3)

            # 2.5 叠加排斥速度（轻量碰撞避免）
            u_f = self._guard.repulsive_velocity(positions, u_f, self._follower_indices)

            # 3. 安全检查 + 限幅
            safety_status = self._guard.check(
                positions=positions,
                velocities_cmd=u_f,
                follower_ids=self._follower_indices,
                pose_state=state,
                nominal_positions=self._nominal_pos,
            )
            u_safe = safety_status.clipped_velocities  # 始终有值
            u_safe = self._apply_first_pass_limits(u_safe)

            # 4. 日志（在安全响应之前，确保 emergency 也能记录）
            if self._log_enabled and step % self._log_interval == 0:
                err, per_agent_err = self._afc.formation_error(
                    positions, positions[self._leader_indices]
                )
                self._log_step(
                    step, time.time() - t_start, positions, u_safe, err, safety_status
                )

            # 5. 根据安全状态决定发送什么
            t_elapsed = time.time() - t_start
            if safety_status.need_emergency:
                if t_elapsed < 5.0:
                    # 起飞稳定期：降级为悬停，不触发紧急降落
                    logger.warning(
                        f"[FormationRunner] 起飞稳定期 ({t_elapsed:.1f}s)，"
                        f"降级为悬停: {safety_status.reasons}"
                    )
                    self._cmd_bridge.hover_all()
                else:
                    logger.error(f"[FormationRunner] 紧急状态! {safety_status.reasons}")
                    self._emergency_land()
                    break
            elif safety_status.need_hover:
                logger.warning(f"[FormationRunner] 悬停保护: {safety_status.reasons}")
                self._cmd_bridge.hover_all()
            else:
                if self._algorithm_mode == "dry_run":
                    if step % self._log_interval == 0:
                        logger.info(
                            f"[FormationRunner] DRY-RUN step={step} "
                            f"followers={self._follower_indices} "
                            f"u_safe={np.array2string(u_safe, precision=3, suppress_small=True)}"
                        )
                    self._cmd_bridge.hover_all()
                elif self._algorithm_mode == "live_single_follower":
                    live_target = self._cfg["algorithm_integration"][
                        "single_live_follower"
                    ]
                    live_ids = [
                        d["id"]
                        for d in self._cfg["drones"]
                        if d.get("name") == live_target
                    ]
                    if not live_ids:
                        raise RuntimeError(
                            f"single_live_follower={live_target!r} not found in config"
                        )
                    did = live_ids[0]
                    idx = self._follower_indices.index(did)
                    vx, vy, vz = [float(x) for x in u_safe[idx]]
                    self._cmd_bridge.send_drone_velocity(did, vx, vy, vz)
                    self._cmd_bridge.hover_all()
                else:
                    # 正常下发速度
                    self._cmd_bridge.send_follower_velocities(
                        self._follower_indices, u_safe
                    )

            # Leader 持续发送 keepalive（防止 watchdog 超时）
            self._cmd_bridge.keepalive_hover(watchdog_interval_s=0.3)

            step += 1

            # 6. 等待下一控制步
            elapsed = time.time() - t_loop_start
            sleep_t = self._dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
            elif elapsed > self._dt * 1.5:
                logger.warning(
                    f"[FormationRunner] 控制步超时: {elapsed * 1000:.1f}ms "
                    f"(目标 {self._dt * 1000:.0f}ms)"
                )

    def _apply_first_pass_limits(self, velocities: np.ndarray) -> np.ndarray:
        """Apply conservative first-pass live limits from algorithm_integration config."""
        limited = velocities.copy()
        if self._planar_only:
            limited[:, 2] = 0.0
        else:
            limited[:, 2] = np.clip(
                limited[:, 2], -self._max_z_velocity, self._max_z_velocity
            )

        xy = limited[:, :2]
        xy_norm = np.linalg.norm(xy, axis=1, keepdims=True)
        xy_scale = np.where(
            xy_norm > self._max_xy_velocity,
            self._max_xy_velocity / np.maximum(xy_norm, 1e-12),
            1.0,
        )
        limited[:, :2] = xy * xy_scale
        return limited

    # ─────────────────────────────────────────
    # 安全关闭
    # ─────────────────────────────────────────

    def _emergency_land(self):
        logger.warning("[FormationRunner] !!! 紧急降落 !!!")
        self._running = False
        self._emergency = True  # 防止 _shutdown() 再次触发降落
        if self._cmd_bridge:
            self._cmd_bridge.land_all(duration_s=2.0)

    def _shutdown(self):
        """正常或紧急退出时的清理流程。"""
        logger.info("[FormationRunner] 开始关闭流程...")
        self._running = False

        if self._cmd_bridge and not self._emergency:
            logger.info("[FormationRunner] 全机降落...")
            self._cmd_bridge.land_all()

        self._pose_bridge.stop()

        if self._log_file:
            name = self._log_file.name
            self._log_file.close()
            # 统计实际数据行数（不含表头）
            try:
                with open(name, "r", encoding="utf-8") as f:
                    n_rows = sum(1 for _ in f) - 1  # 减去表头
                logger.info(f"[FormationRunner] 日志已保存: {name} ({n_rows} 行数据)")
            except Exception:
                logger.info(f"[FormationRunner] 日志已保存: {name}")

        logger.info("[FormationRunner] 关闭完成")

    # ─────────────────────────────────────────
    # 键盘紧急停止监听
    # ─────────────────────────────────────────

    def _keyboard_listener(self):
        """监听终端键盘输入（q = 紧急降落）。"""
        emergency_key = self._cfg["safety"].get("emergency_key", "q")
        while self._running:
            try:
                key = input()
                if key.strip().lower() == emergency_key:
                    logger.warning(
                        f"[FormationRunner] 收到紧急停止键 '{emergency_key}'，降落..."
                    )
                    self._emergency = True
                    self._emergency_land()
                    break
            except EOFError:
                break

    # ─────────────────────────────────────────
    # CSV 日志
    # ─────────────────────────────────────────

    def _open_log(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(self._log_dir, f"run_{ts}.csv")
        self._log_file = open(log_path, "w", newline="", encoding="utf-8")
        fields = ["step", "t_s", "formation_error_m"]
        for i in range(self._n):
            fields += [f"d{i}_x", f"d{i}_y", f"d{i}_z"]
        for i in self._follower_indices:
            fields += [f"d{i}_vx_cmd", f"d{i}_vy_cmd", f"d{i}_vz_cmd"]
        fields += ["safety_level"]
        self._csv_writer = csv.DictWriter(self._log_file, fieldnames=fields)
        self._csv_writer.writeheader()
        logger.info(f"[FormationRunner] 日志文件: {log_path}")

    def _log_step(self, step, t, positions, u_safe, err, status: SafetyStatus):
        if self._csv_writer is None:
            return
        row = {"step": step, "t_s": f"{t:.4f}", "formation_error_m": f"{err:.5f}"}
        for i in range(self._n):
            row[f"d{i}_x"] = f"{positions[i, 0]:.4f}"
            row[f"d{i}_y"] = f"{positions[i, 1]:.4f}"
            row[f"d{i}_z"] = f"{positions[i, 2]:.4f}"
        for k, i in enumerate(self._follower_indices):
            row[f"d{i}_vx_cmd"] = f"{u_safe[k, 0]:.4f}"
            row[f"d{i}_vy_cmd"] = f"{u_safe[k, 1]:.4f}"
            row[f"d{i}_vz_cmd"] = f"{u_safe[k, 2]:.4f}"
        row["safety_level"] = status.level
        self._csv_writer.writerow(row)
        self._log_file.flush()


# ─────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    _config_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "fleet_config.json"
    )
    runner = FormationRunner(_config_path)
    runner.run()
