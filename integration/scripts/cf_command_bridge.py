"""
cf_command_bridge.py - 命令下发桥接模块

职责：
  把算法输出的速度或位置参考转换成 Crazyflie 可接收的 setpoint 命令，
  通过 cflib 逐机下发，并提供起飞/悬停/降落的统一安全接口。

重要：
  CommandBridge 复用 PoseBridge 已建立的 SyncCrazyflie 连接（通过
  pose_bridge.get_cf_connections() 获取），每架 Crazyflie 只允许
  存在一条无线连接。

接口：
  bridge = CommandBridge(config, sc_dict)
  bridge.takeoff_all()                       # 全机起飞
  bridge.send_follower_velocities(ids, vels)  # 下发速度（n_f × 3）
  bridge.hover_all()                         # 全机悬停（零速度）
  bridge.land_all()                          # 全机降落

依赖：pip install cflib
"""

import time
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie  # noqa: F401

    _CFLIB_OK = True
except ImportError:
    _CFLIB_OK = False


class CommandBridge:
    """
    下发 setpoint 命令到 Crazyflie 机群。

    Parameters
    ----------
    config : dict
        fleet_config.json 的完整解析结果
    sc_dict : dict[int, SyncCrazyflie]
        由 PoseBridge.get_cf_connections() 提供的连接句柄字典
        key = drone id，value = SyncCrazyflie 对象
    """

    def __init__(self, config: dict, sc_dict: dict):
        self._cfg = config
        self._sc = sc_dict  # {id: SyncCrazyflie}
        self._drones = config["drones"]
        self._ctrl = config["control"]
        self._safety = config["safety"]
        self._comm = config.get("communication", {})

        self._cmd_type = self._ctrl.get("command_type", "velocity")
        self._u_max = float(self._safety["max_velocity_mps"])
        self._startup_enhighlevel_rounds = int(
            self._comm.get("startup_enhighlevel_rounds", 3)
        )
        self._startup_enhighlevel_retry_interval_s = float(
            self._comm.get("startup_enhighlevel_retry_interval_s", 0.15)
        )
        self._startup_enhighlevel_post_delay_s = float(
            self._comm.get("startup_enhighlevel_post_delay_s", 0.2)
        )
        self._takeoff_retries_default = int(self._comm.get("takeoff_retries", 3))
        self._takeoff_retry_interval_s = float(
            self._comm.get("takeoff_retry_interval_s", 0.3)
        )
        self._takeoff_inter_drone_delay_s = float(
            self._comm.get("takeoff_inter_drone_delay_s", 0.0)
        )
        self._hover_command_interval_s = float(
            self._comm.get("hover_command_interval_s", 0.3)
        )
        self._follower_anchor_refresh_error_m = float(
            self._comm.get("follower_anchor_refresh_error_m", 0.08)
        )
        self._leader_command_spacing_s = float(
            self._comm.get("leader_command_spacing_s", 0.0)
        )
        self._land_retries_default = int(self._comm.get("land_retries", 3))
        self._land_retry_interval_s = float(
            self._comm.get("land_retry_interval_s", 0.3)
        )
        self._land_inter_drone_delay_s = float(
            self._comm.get("land_inter_drone_delay_s", 0.0)
        )

        # Leader 列表（不应收到低级速度命令，以免覆盖高级控制器）
        self._leader_ids = set(config["formation"]["leader_indices"])

        # 记录每架飞机最后下发命令的时间（防止 commander watchdog 超时）
        self._last_cmd_t: dict[int, float] = {d["id"]: 0.0 for d in self._drones}

        # Leader 悬停目标位置（takeoff 后由 go_to 锁定）
        self._leader_hover_pos: dict[int, tuple] = {}
        # Follower dry-run 锁定位姿（用于位置保持 dry-run）
        self._follower_hold_pos: dict[int, tuple] = {}
        # 当前由外层 retained-intent / local pacing 驱动的 follower，不参与 anchor keepalive
        self._dynamic_follower_ids: set[int] = set()
        self._dynamic_follower_velocity: dict[int, tuple[float, float, float]] = {}

    # ─────────────────────────────────────────
    # 起飞 / 降落
    # ─────────────────────────────────────────

    def takeoff_all(
        self,
        height_m: float | None = None,
        duration_s: float = 3.0,
        max_retries: int | None = None,
    ):
        """
        全机起飞（使用 HighLevelCommander），带重试机制。

        Parameters
        ----------
        height_m : float or None
            起飞目标高度，None 则使用各飞机的 takeoff_height_m
        duration_s : float
            起飞动作持续时间
        max_retries : int
            发送起飞命令的次数（重复发送以对抗无线丢包）
        """
        if not _CFLIB_OK:
            logger.warning("[CommandBridge] MOCK 模式：takeoff_all() 空操作")
            return

        logger.info("[CommandBridge] 全机起飞开始...")
        retries = self._takeoff_retries_default if max_retries is None else max_retries

        # 起飞前多轮确认 enHighLevel = 1（对抗 radio 丢包）
        import struct
        from cflib.crtp.crtpstack import CRTPPacket, CRTPPort

        PARAM_WRITE_CH = 2
        for _round in range(self._startup_enhighlevel_rounds):
            for drone in self._drones:
                sc = self._sc.get(drone["id"])
                if sc is None:
                    continue
                try:
                    element = sc.cf.param.toc.get_element_by_complete_name(
                        "commander.enHighLevel"
                    )
                    if element:
                        pk = CRTPPacket()
                        pk.set_header(CRTPPort.PARAM, PARAM_WRITE_CH)
                        pk.data = struct.pack("<H", element.ident)
                        pk.data += struct.pack("<B", 1)
                        sc.cf.send_packet(pk)
                except Exception:
                    pass
                if self._takeoff_inter_drone_delay_s > 0.0:
                    time.sleep(self._takeoff_inter_drone_delay_s)
            time.sleep(self._startup_enhighlevel_retry_interval_s)
        time.sleep(self._startup_enhighlevel_post_delay_s)  # 等参数生效

        # 重复发送起飞命令以对抗单 radio 的包丢失
        for attempt in range(retries):
            for drone in self._drones:
                did = drone["id"]
                h = height_m if height_m is not None else drone["takeoff_height_m"]
                sc = self._sc.get(did)
                if sc is None:
                    if attempt == 0:
                        logger.warning(f"[CommandBridge] drone {did} 无连接，跳过起飞")
                    continue
                try:
                    sc.cf.high_level_commander.takeoff(h, duration_s)
                    if attempt == 0:
                        logger.info(f"[CommandBridge] drone {did} 起飞 -> {h:.2f}m")
                    self._last_cmd_t[did] = time.time()
                except Exception as e:
                    logger.warning(
                        f"[CommandBridge] drone {did} 起飞命令失败 (attempt {attempt}): {e}"
                    )
                if self._takeoff_inter_drone_delay_s > 0.0:
                    time.sleep(self._takeoff_inter_drone_delay_s)
            time.sleep(self._takeoff_retry_interval_s)  # 每轮间隔，让 radio 有时间处理

        time.sleep(duration_s)  # 等待起飞完成
        logger.info("[CommandBridge] 全机起飞完成")

    def lock_leader_positions(self, positions: np.ndarray):
        """
        起飞后调用：让每个 Leader 用 go_to 锁定当前位置。

        HighLevelCommander.takeoff 是有限时长轨迹，到期后飞机无 setpoint，
        会触发 commander watchdog 或漂移。go_to 设 duration 很长，保持位置。

        Parameters
        ----------
        positions : (n, 3) 所有飞机当前位置
        """
        if not _CFLIB_OK:
            return
        for drone in self._drones:
            did = drone["id"]
            if did not in self._leader_ids:
                continue
            sc = self._sc.get(did)
            if sc is None:
                continue
            x, y, z = (
                float(positions[did, 0]),
                float(positions[did, 1]),
                float(positions[did, 2]),
            )
            h = drone.get("takeoff_height_m", 0.4)
            # 用 takeoff_height 而非当前 z（起飞过程中 z 可能还没到位）
            target_z = max(z, h)
            try:
                sc.cf.high_level_commander.go_to(x, y, target_z, 0.0, 1.0)
                self._leader_hover_pos[did] = (x, y, target_z)
                logger.info(
                    f"[CommandBridge] leader {did} 锁定位置 ({x:.2f},{y:.2f},{target_z:.2f})"
                )
            except Exception as e:
                logger.warning(f"[CommandBridge] leader {did} go_to 失败: {e}")

    def update_leader_target(self, drone_id: int, x: float, y: float, z: float):
        """
        更新 Leader 的位置目标（用于轨迹跟随）。

        发送 go_to 命令并更新 keepalive 位置，使后续 keepalive
        维持新位置而非旧位置。
        """
        if not _CFLIB_OK or drone_id not in self._leader_ids:
            return
        sc = self._sc.get(drone_id)
        if sc is None:
            return
        try:
            sc.cf.high_level_commander.go_to(x, y, z, 0.0, 1.0)
            self._leader_hover_pos[drone_id] = (x, y, z)
            self._last_cmd_t[drone_id] = time.time()
            if self._leader_command_spacing_s > 0.0:
                time.sleep(self._leader_command_spacing_s)
        except Exception:
            pass

    def lock_follower_positions(self, positions: np.ndarray, follower_ids: list[int]):
        """让指定 follower 使用 HLC go_to 锁定当前位置。"""
        if not _CFLIB_OK:
            return
        for did in follower_ids:
            sc = self._sc.get(did)
            if sc is None or did in self._leader_ids:
                continue
            x, y, z = (
                float(positions[did, 0]),
                float(positions[did, 1]),
                float(positions[did, 2]),
            )
            drone = next((d for d in self._drones if d["id"] == did), None)
            if drone is None:
                continue
            h = float(drone.get("takeoff_height_m", 0.4))
            target_z = max(z, h)
            try:
                sc.cf.high_level_commander.go_to(x, y, target_z, 0.0, 1.0)
                self._follower_hold_pos[did] = (x, y, target_z)
                self._last_cmd_t[did] = time.time()
                logger.info(
                    f"[CommandBridge] follower {did} 锁定位置 ({x:.2f},{y:.2f},{target_z:.2f})"
                )
            except Exception as e:
                logger.warning(f"[CommandBridge] follower {did} go_to 锁定失败: {e}")

    def set_dynamic_followers(self, follower_ids: list[int] | tuple[int, ...]):
        """标记当前由 retained-intent 驱动的 follower，避免被 anchor keepalive 覆盖。"""
        new_dynamic_ids = {
            int(did) for did in follower_ids if int(did) not in self._leader_ids
        }
        removed_ids = self._dynamic_follower_ids - new_dynamic_ids
        for did in removed_ids:
            self._dynamic_follower_velocity.pop(did, None)
        self._dynamic_follower_ids = new_dynamic_ids

    def update_dynamic_follower_velocity(
        self,
        drone_id: int,
        vx: float,
        vy: float,
        vz: float,
    ):
        if drone_id in self._dynamic_follower_ids:
            self._dynamic_follower_velocity[drone_id] = (
                float(vx),
                float(vy),
                float(vz),
            )

    def hold_follower_positions_if_due(
        self,
        follower_ids: list[int],
        positions: np.ndarray | None = None,
        min_interval_s: float | None = None,
    ) -> list[int]:
        """对指定 follower 按偏移阈值/最小间隔稀疏重发 HLC go_to。"""
        if not _CFLIB_OK:
            return []
        interval = (
            self._hover_command_interval_s if min_interval_s is None else min_interval_s
        )
        now = time.time()
        refreshed_ids: list[int] = []
        for did in follower_ids:
            if did in self._leader_ids:
                continue
            if did in self._dynamic_follower_ids:
                continue
            pos = self._follower_hold_pos.get(did)
            if pos is None:
                continue
            if positions is not None:
                current = positions[did]
                err = np.linalg.norm(
                    np.array(current, dtype=float) - np.array(pos, dtype=float)
                )
                if err < self._follower_anchor_refresh_error_m:
                    continue
            elif now - self._last_cmd_t.get(did, 0.0) < interval:
                continue
            sc = self._sc.get(did)
            if sc is None:
                continue
            try:
                sc.cf.high_level_commander.go_to(pos[0], pos[1], pos[2], 0.0, 1.0)
                self._last_cmd_t[did] = now
                refreshed_ids.append(did)
                break  # stagger: 每轮最多刷新一架 follower，避免双发 burst
            except Exception as e:
                logger.warning(f"[CommandBridge] follower {did} 锚点保持失败: {e}")
        return refreshed_ids

    def land_all(
        self,
        height_m: float = 0.05,
        duration_s: float = 3.0,
        priority_ids: list[int] | None = None,
        max_retries: int | None = None,
    ):
        """全机降落（使用 HighLevelCommander），重复发送以对抗丢包。"""
        if not _CFLIB_OK:
            logger.warning("[CommandBridge] MOCK 模式：land_all() 空操作")
            return

        logger.info("[CommandBridge] 全机降落...")
        retries = self._land_retries_default if max_retries is None else max_retries
        ordered_ids = []
        if priority_ids:
            for did in priority_ids:
                if did not in ordered_ids:
                    ordered_ids.append(did)
        for drone in self._drones:
            did = drone["id"]
            if did not in ordered_ids:
                ordered_ids.append(did)

        for attempt in range(retries):
            for did in ordered_ids:
                sc = self._sc.get(did)
                if sc is None:
                    continue
                try:
                    sc.cf.high_level_commander.land(height_m, duration_s)
                    if attempt == 0:
                        logger.info(f"[CommandBridge] drone {did} 降落")
                except Exception as e:
                    logger.warning(f"[CommandBridge] drone {did} 降落命令失败: {e}")
                if self._land_inter_drone_delay_s > 0.0:
                    time.sleep(self._land_inter_drone_delay_s)
            time.sleep(self._land_retry_interval_s)

        time.sleep(duration_s)
        logger.info("[CommandBridge] 全机降落完成")

    # ─────────────────────────────────────────
    # 速度 setpoint（主控制命令）
    # ─────────────────────────────────────────

    def send_follower_velocities(self, follower_ids: list, velocities: np.ndarray):
        """
        向所有 Follower 下发世界坐标速度 setpoint。

        Parameters
        ----------
        follower_ids : list of int
            Follower 的逻辑编号列表，与 velocities 行一一对应
        velocities : ndarray (n_f, 3)
            目标速度，单位 m/s，已经过安全限幅
        """
        if not _CFLIB_OK:
            logger.debug("[CommandBridge] MOCK 模式：send_follower_velocities() 空操作")
            return

        for i, did in enumerate(follower_ids):
            vx, vy, vz = (
                float(velocities[i, 0]),
                float(velocities[i, 1]),
                float(velocities[i, 2]),
            )
            # 保险限幅（safety_guard 已做过一次，这里再保底）
            speed = np.sqrt(vx**2 + vy**2 + vz**2)
            if speed > self._u_max:
                scale = self._u_max / speed
                vx, vy, vz = vx * scale, vy * scale, vz * scale

            sc = self._sc.get(did)
            if sc is None:
                logger.warning(f"[CommandBridge] drone {did} 无连接，跳过速度命令")
                continue
            try:
                sc.cf.commander.send_velocity_world_setpoint(vx, vy, vz, 0.0)
                self._last_cmd_t[did] = time.time()
            except Exception as e:
                logger.warning(f"[CommandBridge] drone {did} 速度命令失败: {e}")

    def send_drone_velocity(self, drone_id: int, vx: float, vy: float, vz: float):
        """向单架飞机下发速度 setpoint。"""
        if not _CFLIB_OK:
            return
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        if speed > self._u_max:
            s = self._u_max / speed
            vx, vy, vz = vx * s, vy * s, vz * s
        if drone_id in self._dynamic_follower_ids:
            self._dynamic_follower_velocity[drone_id] = (
                float(vx),
                float(vy),
                float(vz),
            )
        sc = self._sc.get(drone_id)
        if sc is None:
            return
        try:
            sc.cf.commander.send_velocity_world_setpoint(vx, vy, vz, 0.0)
            self._last_cmd_t[drone_id] = time.time()
        except Exception as e:
            logger.warning(f"[CommandBridge] drone {drone_id} 速度命令失败: {e}")

    # ─────────────────────────────────────────
    # 悬停 / 紧急停止
    # ─────────────────────────────────────────

    def _all_follower_ids(self) -> list[int]:
        return [
            drone["id"] for drone in self._drones if drone["id"] not in self._leader_ids
        ]

    def hover_all_if_due(self, min_interval_s: float | None = None):
        """仅当悬停命令到期时，才向所有 follower 发送零速度命令。"""
        self.hold_or_hover_followers_if_due(
            self._all_follower_ids(), min_interval_s=min_interval_s
        )

    def hold_or_hover_followers_if_due(
        self,
        follower_ids: list[int],
        positions: np.ndarray | None = None,
        min_interval_s: float | None = None,
    ):
        """若 follower 已有 anchor，则走 HLC hold；否则走低级零速度 hover。"""
        anchored_ids = [did for did in follower_ids if did in self._follower_hold_pos]
        plain_ids = [did for did in follower_ids if did not in self._follower_hold_pos]
        if anchored_ids:
            self.hold_follower_positions_if_due(
                anchored_ids, positions=positions, min_interval_s=min_interval_s
            )
        if plain_ids:
            self.hover_followers_if_due(plain_ids, min_interval_s=min_interval_s)

    def hover_followers_if_due(
        self, follower_ids: list[int], min_interval_s: float | None = None
    ):
        """仅对超过最小间隔未收到命令的 follower 发送零速度命令。"""
        interval = (
            self._hover_command_interval_s if min_interval_s is None else min_interval_s
        )
        now = time.time()
        due_ids = [
            did
            for did in follower_ids
            if (did not in self._leader_ids)
            and (now - self._last_cmd_t.get(did, 0.0) >= interval)
        ]
        if due_ids:
            self.hover_followers(due_ids)

    def hover_all(self):
        """向所有 Follower 发送零速度命令（就地悬停）。
        Leader 保持 HighLevelCommander 的自主位置保持，不发送低级命令。
        """
        self.hover_followers(self._all_follower_ids())

    def hover_followers(self, follower_ids: list[int]):
        """仅向指定 follower 发送零速度命令。"""
        if not _CFLIB_OK:
            logger.warning("[CommandBridge] MOCK 模式：hover_followers() 空操作")
            return

        target_ids = set(follower_ids)
        for did in target_ids:
            if did in self._leader_ids:
                continue
            sc = self._sc.get(did)
            if sc is None:
                continue
            try:
                sc.cf.commander.send_velocity_world_setpoint(0.0, 0.0, 0.0, 0.0)
                self._last_cmd_t[did] = time.time()
            except Exception as e:
                logger.warning(f"[CommandBridge] drone {did} 定向悬停命令失败: {e}")

    def stop_all(self):
        """
        停止全机推力（紧急制动，飞机将坠落）。
        仅在确认飞机接近地面时使用。
        """
        if not _CFLIB_OK:
            logger.warning("[CommandBridge] MOCK 模式：stop_all() 空操作")
            return

        logger.warning("[CommandBridge] !!! 紧急停止所有推力 !!!")
        for drone in self._drones:
            did = drone["id"]
            sc = self._sc.get(did)
            if sc is None:
                continue
            try:
                sc.cf.commander.send_stop_setpoint()
            except Exception as e:
                logger.warning(f"[CommandBridge] drone {did} 停止命令失败: {e}")

    # ─────────────────────────────────────────
    # watchdog：防止 commander 超时
    # ─────────────────────────────────────────

    def keepalive_hover(self, watchdog_interval_s: float = 0.3):
        """
        对超过 watchdog_interval_s 未收到命令的 Follower 补发零速度。
        对 Leader 定期重发 go_to 维持位置锁定。
        """
        now = time.time()
        leader_due: list[int] = []
        follower_anchor_due: list[int] = []
        follower_zero_due: list[int] = []
        for drone in self._drones:
            did = drone["id"]
            if did in self._leader_ids:
                # Leader: 每 2 秒重发一次 go_to 防止高级控制器超时
                if now - self._last_cmd_t.get(did, 0.0) > 2.0:
                    if self._leader_hover_pos.get(did) is not None:
                        leader_due.append(did)
            else:
                hold_pos = self._follower_hold_pos.get(did)
                if did in self._dynamic_follower_ids:
                    if now - self._last_cmd_t.get(did, 0.0) > watchdog_interval_s:
                        follower_zero_due.append(did)
                elif hold_pos is not None:
                    if now - self._last_cmd_t.get(did, 0.0) > 2.0:
                        follower_anchor_due.append(did)
                elif now - self._last_cmd_t.get(did, 0.0) > watchdog_interval_s:
                    follower_zero_due.append(did)

        if leader_due:
            did = leader_due[0]
            pos = self._leader_hover_pos.get(did)
            sc = self._sc.get(did)
            if pos is not None and sc is not None:
                try:
                    sc.cf.high_level_commander.go_to(pos[0], pos[1], pos[2], 0.0, 2.0)
                    self._last_cmd_t[did] = now
                except Exception:
                    pass

        if follower_anchor_due:
            self.hold_follower_positions_if_due(
                [follower_anchor_due[0]], positions=None, min_interval_s=0.0
            )

        if follower_zero_due:
            did = follower_zero_due[0]
            retained_velocity = self._dynamic_follower_velocity.get(did)
            if did in self._dynamic_follower_ids and retained_velocity is not None:
                self.send_drone_velocity(
                    did,
                    retained_velocity[0],
                    retained_velocity[1],
                    retained_velocity[2],
                )
            else:
                self.send_drone_velocity(did, 0.0, 0.0, 0.0)
