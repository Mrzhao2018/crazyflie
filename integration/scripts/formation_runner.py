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
from stage_a_runtime import (
    CoordinatorCommandIntent,
    CoordinatorLeaderUpdate,
    CoordinatorTickPlan,
    FollowerIntent,
    StageARuntimeComponents,
    StageARuntimeSnapshot,
    _MissionCoordinator,
    _StateAggregator,
    _SubgroupControllerStateStore,
)
from stage_a_execution import (
    CommandExecutionResult,
    SafetyExecutionDecision,
    SubgroupExecutionResult,
    _RadioGroupExecutor,
    _SafetyArbiter,
)
from stage_a_startup import (
    StageAStartupComponents,
    _ControlLoopEntrypoint,
    _PreflightInspector,
    _TakeoffVerifier,
)

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
        self._dry_run_print_interval = int(
            algo_cfg.get("dry_run_print_interval_steps", 20)
        )
        self._leader_update_interval_s = float(
            algo_cfg.get("leader_update_interval_s", 0.2)
        )
        stage_b_cfg = self._cfg.get("stage_b", {})
        self._stage_b_enabled = bool(stage_b_cfg.get("enabled", False))
        self._single_follower_intent_refresh_s = float(
            stage_b_cfg.get("single_follower_intent_refresh_s", 0.5)
        )
        self._single_follower_target_change_threshold_m = float(
            stage_b_cfg.get("single_follower_target_change_threshold_m", 0.05)
        )
        self._single_follower_target_change_threshold_rad = float(
            stage_b_cfg.get("single_follower_target_change_threshold_rad", 0.0)
        )
        self._single_follower_use_velocity_fallback = bool(
            stage_b_cfg.get("single_follower_use_velocity_fallback", True)
        )
        self._single_follower_max_refresh_jitter_s = float(
            stage_b_cfg.get("single_follower_max_refresh_jitter_s", 0.1)
        )

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
        self._last_log_flush_step = -1
        self._log_flush_interval = max(1, int(log_cfg.get("flush_interval_steps", 10)))
        self._last_leader_update_t = -1.0
        self._leader_update_rr_index = 0

        # ── 低频分段 profile ──
        prof_cfg = self._cfg.get("profiling", {})
        self._profiling_enabled = bool(prof_cfg.get("enabled", False))
        self._profiling_interval = max(1, int(prof_cfg.get("interval_steps", 50)))
        self._profiling_warn_threshold_ms = float(
            prof_cfg.get("warn_threshold_ms", 150.0)
        )

        # ── Stage A runtime components ──
        self._stage_a = StageARuntimeComponents(
            state_aggregator=_StateAggregator(self),
            mission_coordinator=_MissionCoordinator(self),
            subgroup_controller_state=_SubgroupControllerStateStore(self),
            safety_arbiter=_SafetyArbiter(self),
            radio_group_executor=_RadioGroupExecutor(self),
        )
        self._stage_a_startup = StageAStartupComponents(
            preflight_inspector=_PreflightInspector(self),
            takeoff_verifier=_TakeoffVerifier(self),
            control_loop_entrypoint=_ControlLoopEntrypoint(self),
        )

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

    def _build_stage_a_runtime_snapshot(self, state: dict) -> StageARuntimeSnapshot:
        radio_groups: dict[tuple[int, int], list[int]] = {}
        for drone in self._cfg["drones"]:
            uri = str(drone["uri"])
            parts = uri.split("/")
            radio_idx = int(parts[2])
            channel = int(parts[3])
            radio_groups.setdefault((radio_idx, channel), []).append(int(drone["id"]))
        return StageARuntimeSnapshot(
            raw_state=state,
            positions=np.array(state["positions"], copy=True),
            disconnected_ids=tuple(sorted(state.get("disconnected_ids", []))),
            radio_groups={key: tuple(ids) for key, ids in radio_groups.items()},
        )

    def _resolve_single_live_follower(self) -> tuple[int, int, tuple[int, ...]]:
        live_target = self._cfg["algorithm_integration"]["single_live_follower"]
        live_ids = [
            d["id"] for d in self._cfg["drones"] if d.get("name") == live_target
        ]
        if not live_ids:
            raise RuntimeError(
                f"single_live_follower={live_target!r} not found in config"
            )
        did = int(live_ids[0])
        if did not in self._follower_indices:
            raise RuntimeError(
                f"single_live_follower={live_target!r} is not a follower"
            )
        follower_index = self._follower_indices.index(did)
        hover_ids = tuple(fid for fid in self._follower_indices if fid != did)
        return did, follower_index, hover_ids

    def _initialize_stage_b_single_follower_state(self, state: dict):
        if self._algorithm_mode != "live_single_follower":
            return
        did, follower_index, hover_ids = self._resolve_single_live_follower()
        self._stage_a.subgroup_controller_state.initialize_single_follower(
            active_follower_id=did,
            active_follower_index=follower_index,
            parked_follower_ids=hover_ids,
        )
        if self._cmd_bridge is not None:
            if self._stage_b_enabled:
                self._cmd_bridge.set_dynamic_followers([did])
            else:
                self._cmd_bridge.set_dynamic_followers([])

    def _build_coordinator_tick_plan(
        self, runtime_snapshot: StageARuntimeSnapshot, t_elapsed: float
    ) -> CoordinatorTickPlan:
        leader_update = CoordinatorLeaderUpdate(should_update=False)
        leader_traj = self._leader_traj
        should_update_leaders = leader_traj is not None and (
            self._last_leader_update_t < 0.0
            or (t_elapsed - self._last_leader_update_t)
            >= self._leader_update_interval_s
        )
        if should_update_leaders:
            if leader_traj is None:
                raise RuntimeError("Leader trajectory 未初始化")
            leader_targets = leader_traj(t_elapsed)
            n_leaders = len(self._leader_indices)
            rr_idx = self._leader_update_rr_index % max(1, n_leaders)
            lid = self._leader_indices[rr_idx]
            leader_update = CoordinatorLeaderUpdate(
                should_update=True,
                target_drone_id=lid,
                target_position=(
                    float(leader_targets[rr_idx, 0]),
                    float(leader_targets[rr_idx, 1]),
                    float(leader_targets[rr_idx, 2]),
                ),
            )

        desired_follower_velocities = self._afc.all_follower_inputs(
            runtime_snapshot.positions
        )
        desired_follower_velocities = self._guard.repulsive_velocity(
            runtime_snapshot.positions,
            desired_follower_velocities,
            self._follower_indices,
        )

        command_intent = CoordinatorCommandIntent(mode=self._algorithm_mode)
        if self._algorithm_mode == "live_single_follower":
            did, follower_index, hover_ids = self._resolve_single_live_follower()
            command_intent = CoordinatorCommandIntent(
                mode=self._algorithm_mode,
                target_drone_id=did,
                target_follower_index=follower_index,
                hover_ids=hover_ids,
            )

        return CoordinatorTickPlan(
            runtime_snapshot=runtime_snapshot,
            positions=runtime_snapshot.positions,
            disconnected_ids=runtime_snapshot.disconnected_ids,
            t_elapsed=t_elapsed,
            leader_update=leader_update,
            desired_follower_velocities=desired_follower_velocities,
            command_intent=command_intent,
        )

    def _apply_coordinator_leader_update(
        self, cmd_bridge: CommandBridge, leader_update: CoordinatorLeaderUpdate
    ):
        if not leader_update.should_update:
            return
        if (
            leader_update.target_drone_id is None
            or leader_update.target_position is None
        ):
            raise RuntimeError("Leader update intent 不完整")
        x, y, z = leader_update.target_position
        cmd_bridge.update_leader_target(leader_update.target_drone_id, x, y, z)
        self._leader_update_rr_index = (self._leader_update_rr_index + 1) % max(
            1, len(self._leader_indices)
        )

    def _finalize_command_intent(
        self,
        tick_plan: CoordinatorTickPlan,
        u_safe: np.ndarray,
    ) -> CoordinatorCommandIntent:
        command_intent = tick_plan.command_intent
        if (
            command_intent.mode != "live_single_follower"
            or not self._stage_b_enabled
            or command_intent.target_drone_id is None
            or command_intent.target_follower_index is None
        ):
            return command_intent

        state = self._stage_a.subgroup_controller_state.get_single_follower_state()
        now = tick_plan.t_elapsed
        target_index = command_intent.target_follower_index
        target_id = command_intent.target_drone_id
        velocity_world = (
            float(u_safe[target_index, 0]),
            float(u_safe[target_index, 1]),
            float(u_safe[target_index, 2]),
        )
        current_position = tick_plan.positions[target_id]
        target_position = (
            float(
                current_position[0]
                + velocity_world[0] * self._single_follower_intent_refresh_s
            ),
            float(
                current_position[1]
                + velocity_world[1] * self._single_follower_intent_refresh_s
            ),
            float(
                current_position[2]
                + velocity_world[2] * self._single_follower_intent_refresh_s
            ),
        )

        refresh_reason = "steady_state"
        requires_refresh = False
        retained_intent = state.retained_intent
        if retained_intent is None:
            refresh_reason = "startup"
            requires_refresh = True
        else:
            target_delta = np.linalg.norm(
                np.array(target_position, dtype=float)
                - np.array(retained_intent.target_position, dtype=float)
            )
            velocity_delta = np.linalg.norm(
                np.array(velocity_world, dtype=float)
                - np.array(retained_intent.velocity_world, dtype=float)
            )
            if target_delta >= self._single_follower_target_change_threshold_m:
                refresh_reason = "target_delta"
                requires_refresh = True
            elif velocity_delta >= self._single_follower_target_change_threshold_m:
                refresh_reason = "velocity_delta"
                requires_refresh = True
            elif (now - state.last_refresh_at) >= (
                self._single_follower_intent_refresh_s
                + self._single_follower_max_refresh_jitter_s
            ):
                refresh_reason = "refresh_timer"
                requires_refresh = True
            elif state.last_applied_mode != command_intent.mode:
                refresh_reason = "mode_switch"
                requires_refresh = True

        follower_intent = FollowerIntent(
            target_drone_id=target_id,
            target_follower_index=target_index,
            target_position=target_position,
            velocity_world=velocity_world,
            hover_ids=command_intent.hover_ids,
            refresh_reason=refresh_reason,
            requires_transport_refresh=requires_refresh,
            created_at=now,
            stale_after_s=self._single_follower_intent_refresh_s,
        )
        self._stage_a.subgroup_controller_state.retain_single_follower_intent(
            follower_intent
        )
        return CoordinatorCommandIntent(
            mode=command_intent.mode,
            target_drone_id=command_intent.target_drone_id,
            target_follower_index=command_intent.target_follower_index,
            hover_ids=command_intent.hover_ids,
            follower_intent=follower_intent,
        )

    def _execute_live_single_follower_intent(
        self,
        cmd_bridge: CommandBridge,
        runtime_snapshot: StageARuntimeSnapshot,
        command_intent: CoordinatorCommandIntent,
    ) -> CommandExecutionResult:
        follower_intent = command_intent.follower_intent
        if follower_intent is None:
            raise RuntimeError("live_single_follower 缺少 follower_intent")

        cmd_bridge.set_dynamic_followers([follower_intent.target_drone_id])
        subgroup_results: list[SubgroupExecutionResult] = []
        if follower_intent.requires_transport_refresh:
            vx, vy, vz = follower_intent.velocity_world
            if self._single_follower_use_velocity_fallback:
                cmd_bridge.update_dynamic_follower_velocity(
                    follower_intent.target_drone_id,
                    vx,
                    vy,
                    vz,
                )
                cmd_bridge.send_drone_velocity(
                    follower_intent.target_drone_id, vx, vy, vz
                )
                self._stage_a.subgroup_controller_state.mark_single_follower_refresh(
                    follower_intent.created_at,
                    command_intent.mode,
                )
                self._stage_a.subgroup_controller_state.mark_single_follower_transport_send(
                    follower_intent.created_at,
                )
                active_group = next(
                    (
                        radio_group
                        for radio_group, drone_ids in runtime_snapshot.radio_groups.items()
                        if follower_intent.target_drone_id in drone_ids
                    ),
                    None,
                )
                if active_group is not None:
                    subgroup_results.append(
                        SubgroupExecutionResult(
                            radio_group=active_group,
                            executed_ids=(follower_intent.target_drone_id,),
                            action=f"intent_refresh:{follower_intent.refresh_reason}",
                        )
                    )
        else:
            vx, vy, vz = follower_intent.velocity_world
            cmd_bridge.update_dynamic_follower_velocity(
                follower_intent.target_drone_id,
                vx,
                vy,
                vz,
            )
        refreshed_ids = cmd_bridge.hold_follower_positions_if_due(
            list(follower_intent.hover_ids),
            positions=runtime_snapshot.positions,
        )
        for radio_group, drone_ids in runtime_snapshot.radio_groups.items():
            executed_ids = tuple(did for did in drone_ids if did in refreshed_ids)
            if executed_ids:
                subgroup_results.append(
                    SubgroupExecutionResult(
                        radio_group=radio_group,
                        executed_ids=executed_ids,
                        action="hold_follower_anchor_if_due",
                    )
                )
        action = "live_single_follower_intent_refresh"
        if not follower_intent.requires_transport_refresh:
            action = "live_single_follower_intent_hold"
        return CommandExecutionResult(
            action=action,
            subgroup_results=tuple(subgroup_results),
        )

    def _apply_subgroup_hover_if_due(
        self,
        cmd_bridge: CommandBridge,
        radio_groups: dict[tuple[int, int], tuple[int, ...]],
        hover_ids: tuple[int, ...],
    ) -> tuple[SubgroupExecutionResult, ...]:
        hover_set = set(hover_ids)
        results: list[SubgroupExecutionResult] = []
        for drone_ids in radio_groups.values():
            subgroup_hover_ids = [did for did in drone_ids if did in hover_set]
            if subgroup_hover_ids:
                cmd_bridge.hover_followers_if_due(subgroup_hover_ids)
                results.append(
                    SubgroupExecutionResult(
                        radio_group=next(
                            key for key, ids in radio_groups.items() if ids == drone_ids
                        ),
                        executed_ids=tuple(subgroup_hover_ids),
                        action="hover_if_due",
                    )
                )
        return tuple(results)

    def _apply_subgroup_follower_velocities(
        self,
        cmd_bridge: CommandBridge,
        radio_groups: dict[tuple[int, int], tuple[int, ...]],
        follower_ids: list[int],
        u_safe: np.ndarray,
    ) -> tuple[SubgroupExecutionResult, ...]:
        follower_to_velocity = {
            did: u_safe[idx] for idx, did in enumerate(follower_ids)
        }
        results: list[SubgroupExecutionResult] = []
        for drone_ids in radio_groups.values():
            subgroup_follower_ids = [
                did for did in drone_ids if did in follower_to_velocity
            ]
            if not subgroup_follower_ids:
                continue
            subgroup_u = np.array(
                [follower_to_velocity[did] for did in subgroup_follower_ids],
                dtype=float,
            )
            cmd_bridge.send_follower_velocities(subgroup_follower_ids, subgroup_u)
            results.append(
                SubgroupExecutionResult(
                    radio_group=next(
                        key for key, ids in radio_groups.items() if ids == drone_ids
                    ),
                    executed_ids=tuple(subgroup_follower_ids),
                    action="send_follower_velocities",
                )
            )
        return tuple(results)

    def _apply_command_intent(
        self,
        cmd_bridge: CommandBridge,
        runtime_snapshot: StageARuntimeSnapshot,
        u_safe: np.ndarray,
        command_intent: CoordinatorCommandIntent,
        step: int,
    ) -> CommandExecutionResult:
        if command_intent.mode == "dry_run":
            if step % self._dry_run_print_interval == 0:
                logger.info(
                    f"[FormationRunner] DRY-RUN step={step} "
                    f"followers={self._follower_indices} "
                    f"u_safe={np.array2string(u_safe, precision=3, suppress_small=True)}"
                )
            refreshed_ids = cmd_bridge.hold_follower_positions_if_due(
                self._follower_indices,
                positions=runtime_snapshot.positions,
            )
            subgroup_results = tuple(
                SubgroupExecutionResult(
                    radio_group=radio_group,
                    executed_ids=tuple(
                        did for did in drone_ids if did in refreshed_ids
                    ),
                    action="hold_follower_anchor_if_due",
                )
                for radio_group, drone_ids in runtime_snapshot.radio_groups.items()
                if any(did in refreshed_ids for did in drone_ids)
            )
            return CommandExecutionResult(
                action="dry_run_hold_anchor_if_due",
                subgroup_results=subgroup_results,
            )

        if command_intent.mode == "live_single_follower":
            if self._stage_b_enabled:
                return self._execute_live_single_follower_intent(
                    cmd_bridge,
                    runtime_snapshot,
                    command_intent,
                )
            if (
                command_intent.target_drone_id is None
                or command_intent.target_follower_index is None
            ):
                raise RuntimeError("live_single_follower command intent 不完整")
            vx, vy, vz = [
                float(x) for x in u_safe[command_intent.target_follower_index]
            ]
            cmd_bridge.send_drone_velocity(command_intent.target_drone_id, vx, vy, vz)
            refreshed_ids = cmd_bridge.hold_follower_positions_if_due(
                list(command_intent.hover_ids),
                positions=runtime_snapshot.positions,
            )
            subgroup_results = tuple(
                SubgroupExecutionResult(
                    radio_group=radio_group,
                    executed_ids=tuple(
                        did for did in drone_ids if did in refreshed_ids
                    ),
                    action="hold_follower_anchor_if_due",
                )
                for radio_group, drone_ids in runtime_snapshot.radio_groups.items()
                if any(did in refreshed_ids for did in drone_ids)
            )
            return CommandExecutionResult(
                action="live_single_follower",
                subgroup_results=subgroup_results,
            )

        subgroup_results = self._apply_subgroup_follower_velocities(
            cmd_bridge,
            runtime_snapshot.radio_groups,
            self._follower_indices,
            u_safe,
        )
        return CommandExecutionResult(
            action="group_follower_velocity",
            subgroup_results=subgroup_results,
        )

    def _resolve_safety_execution_decision(
        self,
        safety_status: SafetyStatus,
        tick_plan: CoordinatorTickPlan,
    ) -> SafetyExecutionDecision:
        if safety_status.need_emergency:
            emergency_due_to_disconnect = any(
                reason.startswith("断链") for reason in safety_status.reasons
            )
            if tick_plan.t_elapsed < 5.0 and not emergency_due_to_disconnect:
                return SafetyExecutionDecision(
                    action="hover",
                    message=(
                        f"[FormationRunner] 起飞稳定期 ({tick_plan.t_elapsed:.1f}s)，"
                        f"降级为悬停: {safety_status.reasons}"
                    ),
                )
            return SafetyExecutionDecision(
                action="emergency_land",
                message=f"[FormationRunner] 紧急状态! {safety_status.reasons}",
            )

        if safety_status.need_hover:
            return SafetyExecutionDecision(
                action="hover",
                message=f"[FormationRunner] 悬停保护: {safety_status.reasons}",
            )

        return SafetyExecutionDecision(
            action="execute_plan",
            command_intent=tick_plan.command_intent,
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
        if self._cmd_bridge is None:
            raise RuntimeError("CommandBridge 初始化失败")

        # 等待定位数据就绪
        logger.info("[FormationRunner] 等待定位系统就绪（最多 30s）...")
        if not self._pose_bridge.wait_until_fresh(timeout_s=30.0):
            logger.error("[FormationRunner] 定位数据获取失败，退出")
            self._pose_bridge.stop()
            return

        # 显示当前位置 & 预检查
        self._stage_a_startup.preflight_inspector.run(
            self._pose_bridge.get_latest_state()
        )

        # 等待用户确认起飞
        input("\n按 Enter 键全机起飞并启动编队控制... (Ctrl+C 取消)\n")

        # 起飞
        logger.info("[FormationRunner] 起飞...")
        self._cmd_bridge.takeoff_all()

        # 验证起飞高度（轮询等待位置数据刷新）
        if not self._stage_a_startup.takeoff_verifier.verify():
            return

        # 锁定 Leader / 构建轨迹 / 打开日志 / 启动键盘监听
        if not self._stage_a_startup.control_loop_entrypoint.prepare():
            return
        self._cmd_bridge.lock_follower_positions(
            self._pose_bridge.get_latest_state()["positions"],
            self._follower_indices,
        )

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
        cmd_bridge = self._cmd_bridge
        if cmd_bridge is None:
            raise RuntimeError("CommandBridge 未初始化")

        step = 0
        t_start = time.time()

        while self._running and not self._emergency:
            t_loop_start = time.time()
            t_stage = t_loop_start
            profile_ms: dict[str, float] = {}

            # 1. 读取当前状态
            runtime_snapshot = self._stage_a.state_aggregator.build_snapshot(
                self._pose_bridge.get_latest_state()
            )
            profile_ms["state"] = (time.time() - t_stage) * 1000.0
            t_stage = time.time()
            tick_plan = self._stage_a.mission_coordinator.build_tick_plan(
                runtime_snapshot,
                t_elapsed=time.time() - t_start,
            )
            positions = tick_plan.positions  # (n, 3)
            disconnected_ids = tick_plan.disconnected_ids
            if disconnected_ids:
                logger.error(
                    "[FormationRunner] 检测到飞行中断链，立即紧急降落: "
                    f"drones {disconnected_ids}"
                )
                self._emergency_land()
                break

            # 1.5 更新 Leader 轨迹目标（如果有仿射演示轨迹）
            self._stage_a.mission_coordinator.apply_leader_update(
                cmd_bridge, tick_plan.leader_update
            )
            if tick_plan.leader_update.should_update:
                self._last_leader_update_t = tick_plan.t_elapsed
            profile_ms["leader"] = (time.time() - t_stage) * 1000.0
            t_stage = time.time()

            # 2. 计算 AFC 速度命令（仅 follower）
            #    leaders 的位置来自真实定位，不需要主动控制
            u_f = tick_plan.desired_follower_velocities
            profile_ms["control"] = (time.time() - t_stage) * 1000.0
            t_stage = time.time()

            # 3. 安全检查 + 限幅
            safety_status = self._guard.check(
                positions=positions,
                velocities_cmd=u_f,
                follower_ids=self._follower_indices,
                pose_state=tick_plan.runtime_snapshot.raw_state,
                nominal_positions=self._nominal_pos,
            )
            u_safe = safety_status.clipped_velocities  # 始终有值
            if u_safe is None:
                raise RuntimeError("SafetyGuard 未返回 clipped_velocities")
            u_safe = self._apply_first_pass_limits(u_safe)
            finalized_command_intent = (
                self._stage_a.mission_coordinator.finalize_command_intent(
                    tick_plan,
                    u_safe,
                )
            )
            tick_plan = CoordinatorTickPlan(
                runtime_snapshot=tick_plan.runtime_snapshot,
                positions=tick_plan.positions,
                disconnected_ids=tick_plan.disconnected_ids,
                t_elapsed=tick_plan.t_elapsed,
                leader_update=tick_plan.leader_update,
                desired_follower_velocities=tick_plan.desired_follower_velocities,
                command_intent=finalized_command_intent,
            )
            profile_ms["safety"] = (time.time() - t_stage) * 1000.0
            t_stage = time.time()

            # 4. 日志（在安全响应之前，确保 emergency 也能记录）
            if self._log_enabled and step % self._log_interval == 0:
                err, per_agent_err = self._afc.formation_error(
                    positions, positions[self._leader_indices]
                )
                self._log_step(
                    step, time.time() - t_start, positions, u_safe, err, safety_status
                )
            profile_ms["logging"] = (time.time() - t_stage) * 1000.0
            t_stage = time.time()

            # 5. 根据安全状态决定发送什么
            safety_decision = self._stage_a.safety_arbiter.resolve(
                safety_status,
                tick_plan,
            )
            if safety_decision.message:
                if safety_decision.action == "emergency_land":
                    logger.error(safety_decision.message)
                else:
                    logger.warning(safety_decision.message)

            command_execution_result = CommandExecutionResult(
                action="noop",
                subgroup_results=(),
            )
            if safety_decision.action == "emergency_land":
                self._emergency_land()
                break
            if safety_decision.action == "hover":
                cmd_bridge.hold_or_hover_followers_if_due(self._follower_indices)
                command_execution_result = CommandExecutionResult(
                    action="hover",
                    subgroup_results=tuple(
                        SubgroupExecutionResult(
                            radio_group=radio_group,
                            executed_ids=tuple(
                                did
                                for did in drone_ids
                                if did in self._follower_indices
                            ),
                            action="hold_or_hover_followers_if_due",
                        )
                        for radio_group, drone_ids in tick_plan.runtime_snapshot.radio_groups.items()
                        if any(did in self._follower_indices for did in drone_ids)
                    ),
                )
            elif safety_decision.action == "execute_plan":
                if safety_decision.command_intent is None:
                    raise RuntimeError("Safety decision 缺少 command_intent")
                command_execution_result = self._stage_a.radio_group_executor.execute(
                    cmd_bridge,
                    tick_plan.runtime_snapshot,
                    u_safe,
                    safety_decision.command_intent,
                    step,
                )
            else:
                raise RuntimeError(
                    f"Unknown safety execution action: {safety_decision.action}"
                )
            profile_ms["subgroups"] = float(
                len(command_execution_result.subgroup_results)
            )
            command_trace = ";".join(
                f"{result.action}:{list(result.executed_ids)}"
                for result in command_execution_result.subgroup_results
            )
            if not command_trace:
                command_trace = "none"
            profile_ms["command"] = (time.time() - t_stage) * 1000.0
            t_stage = time.time()

            # Leader 持续发送 keepalive（防止 watchdog 超时）
            cmd_bridge.keepalive_hover(watchdog_interval_s=0.3)
            profile_ms["keepalive"] = (time.time() - t_stage) * 1000.0

            step += 1

            # 6. 等待下一控制步
            elapsed = time.time() - t_loop_start
            total_ms = elapsed * 1000.0
            if self._profiling_enabled and (
                step % self._profiling_interval == 0
                or total_ms >= self._profiling_warn_threshold_ms
            ):
                logger.warning(
                    "[FormationRunner] loop profile "
                    f"step={step} total={total_ms:.1f}ms "
                    f"state={profile_ms.get('state', 0.0):.1f} "
                    f"leader={profile_ms.get('leader', 0.0):.1f} "
                    f"control={profile_ms.get('control', 0.0):.1f} "
                    f"safety={profile_ms.get('safety', 0.0):.1f} "
                    f"logging={profile_ms.get('logging', 0.0):.1f} "
                    f"subgroups={profile_ms.get('subgroups', 0.0):.0f} "
                    f"command={profile_ms.get('command', 0.0):.1f} "
                    f"keepalive={profile_ms.get('keepalive', 0.0):.1f} "
                    f"actions={command_execution_result.action} "
                    f"targets={command_trace}"
                )
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
            state = self._pose_bridge.get_latest_state()
            stale_ids = [
                did
                for did, info in state.get("per_drone", {}).items()
                if not info["fresh"]
            ]
            disconnected_ids = list(state.get("disconnected_ids", []))
            priority_ids = stale_ids + [
                did for did in disconnected_ids if did not in stale_ids
            ]
            self._cmd_bridge.land_all(duration_s=2.0, priority_ids=priority_ids)

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
        if (
            self._log_file is not None
            and (step - self._last_log_flush_step) >= self._log_flush_interval
        ):
            self._log_file.flush()
            self._last_log_flush_step = step


# ─────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    _config_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "fleet_config.json"
    )
    runner = FormationRunner(_config_path)
    runner.run()
