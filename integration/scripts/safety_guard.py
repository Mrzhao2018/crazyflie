"""
safety_guard.py - 安全保护层

职责：
  以纯函数或轻量类的形式，对控制输出做最后一道安全检查。
  formation_runner.py 在每个控制步调用 SafetyGuard.check()，
  若返回 EMERGENCY 则立即触发降落。

安全检查项（按优先级）：
  1. 定位数据新鲜度检查（pose_timeout）
  2. 飞行边界越界检查（boundary）
  3. 最小飞机间距检查（min_distance）
  4. 速度幅值限幅（velocity clipping）
  5. 位置误差过大保护（position error）

所有检查结果汇总为 SafetyStatus，
  status.ok == True  可以继续下发控制命令
  status.ok == False 需要悬停或降落
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class SafetyStatus:
    """单步安全检查的汇总结果。"""

    SAFE = 'safe'
    HOVER = 'hover'           # 悬停，不终止任务
    EMERGENCY = 'emergency'   # 立即触发紧急降落

    def __init__(self):
        self.level = self.SAFE
        self.reasons: list[str] = []
        self.clipped_velocities: np.ndarray | None = None

    @property
    def ok(self) -> bool:
        return self.level == self.SAFE

    @property
    def need_hover(self) -> bool:
        return self.level in (self.HOVER, self.EMERGENCY)

    @property
    def need_emergency(self) -> bool:
        return self.level == self.EMERGENCY

    def _escalate(self, new_level: str, reason: str):
        levels = [self.SAFE, self.HOVER, self.EMERGENCY]
        if levels.index(new_level) > levels.index(self.level):
            self.level = new_level
        self.reasons.append(reason)

    def __repr__(self):
        return f"SafetyStatus({self.level}, reasons={self.reasons})"


class SafetyGuard:
    """
    编队控制安全防护层。

    Parameters
    ----------
    config : dict
        fleet_config.json 的完整解析结果
    """

    def __init__(self, config: dict):
        s = config['safety']
        self._u_max = float(s['max_velocity_mps'])
        self._d_safe = float(s['d_safe_m'])
        self._pose_timeout = float(s['pose_timeout_s'])
        self._max_pos_err = float(s['max_position_error_m'])

        b = s['boundary']
        self._x_min = float(b['x_min'])
        self._x_max = float(b['x_max'])
        self._y_min = float(b['y_min'])
        self._y_max = float(b['y_max'])
        self._z_min = float(b['z_min'])
        self._z_max = float(b['z_max'])

    # ─────────────────────────────────────────
    # 主检查入口
    # ─────────────────────────────────────────

    def check(self,
              positions: np.ndarray,
              velocities_cmd: np.ndarray,
              follower_ids: list,
              pose_state: dict,
              nominal_positions: np.ndarray = None) -> SafetyStatus:
        """
        执行全套安全检查并返回 SafetyStatus。

        Parameters
        ----------
        positions : ndarray (n, 3)
            所有飞机当前位置（来自 pose_bridge）
        velocities_cmd : ndarray (n_f, 3)
            AFC+CBF 输出的原始速度命令（对应 follower_ids）
        follower_ids : list of int
            follower 逻辑编号
        pose_state : dict
            pose_bridge.get_latest_state() 的完整返回
        nominal_positions : ndarray (n, 3) or None
            当前标称编队位置，用于位置误差检查

        Returns
        -------
        SafetyStatus
            clipped_velocities: 限幅后的速度命令（始终完成，不管是否安全）
        """
        status = SafetyStatus()

        # 1. 定位新鲜度
        self._check_pose_fresh(pose_state, status)

        # 2. 边界越界
        self._check_boundary(positions, status)

        # 3. 最小间距
        self._check_min_distance(positions, status)

        # 4. 位置误差过大
        if nominal_positions is not None:
            self._check_position_error(positions, nominal_positions, follower_ids, status)

        # 5. 速度限幅（始终执行，无论安全状态）
        clipped = self.clip_velocities(velocities_cmd)
        status.clipped_velocities = clipped

        if not status.ok:
            logger.warning(f"[SafetyGuard] {status}")

        return status

    # ─────────────────────────────────────────
    # 速度限幅（纯函数，可单独调用）
    # ─────────────────────────────────────────

    def clip_velocities(self, velocities: np.ndarray) -> np.ndarray:
        """
        按范数裁剪速度：方向保持，合速度 <= u_max。

        Parameters
        ----------
        velocities : ndarray (n_f, 3)

        Returns
        -------
        clipped : ndarray (n_f, 3)
        """
        clipped = velocities.copy()
        norms = np.linalg.norm(clipped, axis=1, keepdims=True)   # (n_f, 1)
        exceed = (norms > self._u_max).flatten()
        if np.any(exceed):
            scale = np.where(
                norms > self._u_max,
                self._u_max / np.maximum(norms, 1e-12),
                1.0
            )
            clipped = clipped * scale
        return clipped

    # ─────────────────────────────────────────
    # 各项检查
    # ─────────────────────────────────────────

    def _check_pose_fresh(self, pose_state: dict, status: SafetyStatus):
        per_drone = pose_state.get('per_drone', {})
        stale = [did for did, d in per_drone.items() if not d['fresh']]
        if stale:
            status._escalate(
                SafetyStatus.EMERGENCY,
                f"定位数据过期: drones {stale}"
            )

    def _check_boundary(self, positions: np.ndarray, status: SafetyStatus):
        n = positions.shape[0]
        for i in range(n):
            x, y, z = positions[i]
            if (x < self._x_min or x > self._x_max or
                    y < self._y_min or y > self._y_max or
                    z < self._z_min or z > self._z_max):
                status._escalate(
                    SafetyStatus.EMERGENCY,
                    f"drone {i} 越界: pos=({x:.2f},{y:.2f},{z:.2f})"
                )

    def _check_min_distance(self, positions: np.ndarray, status: SafetyStatus):
        # 向量化计算所有成对间距（O(n^2) 内存，但无 Python 循环）
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (n,n,3)
        dists = np.linalg.norm(diff, axis=-1)  # (n,n)
        n = positions.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                d = float(dists[i, j])
                if d < self._d_safe:
                    status._escalate(
                        SafetyStatus.EMERGENCY,
                        f"drone {i}-{j} 距离过近: {d:.3f}m < {self._d_safe:.3f}m"
                    )
                elif d < self._d_safe * 1.5:
                    status._escalate(
                        SafetyStatus.HOVER,
                        f"drone {i}-{j} 距离警告: {d:.3f}m"
                    )

    def _check_position_error(self,
                              positions: np.ndarray,
                              nominal: np.ndarray,
                              follower_ids: list,
                              status: SafetyStatus):
        for i, fid in enumerate(follower_ids):
            err = float(np.linalg.norm(positions[fid] - nominal[fid]))
            if err > self._max_pos_err:
                status._escalate(
                    SafetyStatus.HOVER,
                    f"drone {fid} 位置误差过大: {err:.2f}m"
                )
