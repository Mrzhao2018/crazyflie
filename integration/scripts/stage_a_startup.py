from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cf_command_bridge import CommandBridge
    from formation_runner import FormationRunner


logger = logging.getLogger("formation_runner")


class _PreflightInspector:
    def __init__(self, runner: "FormationRunner"):
        self._runner = runner

    def run(self, state: dict):
        positions = state["positions"]
        logger.info("[FormationRunner] 当前飞机位置:")
        for did, d in state["per_drone"].items():
            p = d["pos"]
            logger.info(f"  drone {did}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")

        b = self._runner._cfg["safety"]["boundary"]
        preflight_warnings = []
        for i in range(self._runner._n):
            x, y, z = positions[i]
            if x < b["x_min"] or x > b["x_max"] or y < b["y_min"] or y > b["y_max"]:
                preflight_warnings.append(f"drone {i} XY 超出边界: ({x:.2f}, {y:.2f})")
        dists = np.linalg.norm(
            positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1
        )
        d_safe = self._runner._cfg["safety"]["d_safe_m"]
        for i in range(self._runner._n):
            for j in range(i + 1, self._runner._n):
                if dists[i, j] < d_safe:
                    preflight_warnings.append(
                        f"drone {i}-{j} 间距 {dists[i, j]:.3f}m < d_safe {d_safe}m"
                    )
        if preflight_warnings:
            logger.warning("[FormationRunner] 起飞前预检查警告:")
            for w in preflight_warnings:
                logger.warning(f"  ⚠ {w}")
            logger.warning("  请确认飞机位置是否正确再继续")


class _TakeoffVerifier:
    def __init__(self, runner: "FormationRunner"):
        self._runner = runner

    def verify(self) -> bool:
        min_takeoff_z = 0.15
        max_wait = 5.0
        poll_interval = 0.5
        elapsed = 0.0
        failed_drones: list[tuple[int, float]] = []
        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval
            state = self._runner._pose_bridge.get_latest_state()
            disconnected_ids = state.get("disconnected_ids", [])
            if disconnected_ids:
                logger.error(
                    "[FormationRunner] 起飞后检测到断链，立即中止任务: "
                    f"drones {disconnected_ids}"
                )
                self._runner._emergency_land()
                self._runner._pose_bridge.stop()
                return False
            positions = state["positions"]
            failed_drones = []
            for i in range(self._runner._n):
                z = positions[i, 2]
                if z < min_takeoff_z:
                    failed_drones.append((i, z))
            if not failed_drones:
                logger.info(f"[FormationRunner] 全部起飞确认 ({elapsed:.1f}s)")
                return True
            logger.info(
                f"[FormationRunner] 等待起飞... ({elapsed:.1f}s) "
                f"未达标: {[(d, f'{z:.3f}m') for d, z in failed_drones]}"
            )

        for did, z in failed_drones:
            logger.error(f"[FormationRunner] drone {did} 起飞失败! z={z:.3f}m")
        logger.error("[FormationRunner] 有飞机未成功起飞，执行降落")
        if self._runner._cmd_bridge is not None:
            self._runner._cmd_bridge.land_all()
        self._runner._pose_bridge.stop()
        return False


class _ControlLoopEntrypoint:
    def __init__(self, runner: "FormationRunner"):
        self._runner = runner

    def prepare(self) -> bool:
        state = self._runner._pose_bridge.get_latest_state()
        disconnected_ids = state.get("disconnected_ids", [])
        if disconnected_ids:
            logger.error(
                "[FormationRunner] 进入控制前检测到断链，立即中止任务: "
                f"drones {disconnected_ids}"
            )
            self._runner._emergency_land()
            self._runner._pose_bridge.stop()
            return False

        if self._runner._cmd_bridge is None:
            raise RuntimeError("CommandBridge 未初始化")

        self._runner._cmd_bridge.lock_leader_positions(state["positions"])
        leader_pos = state["positions"][self._runner._leader_indices]
        self._runner._build_leader_trajectory(leader_pos)

        self._runner._kbd_thread.start()
        if self._runner._log_enabled:
            self._runner._open_log()
        self._runner._initialize_stage_b_single_follower_state(state)
        return True


@dataclass(frozen=True)
class StageAStartupComponents:
    preflight_inspector: _PreflightInspector
    takeoff_verifier: _TakeoffVerifier
    control_loop_entrypoint: _ControlLoopEntrypoint
