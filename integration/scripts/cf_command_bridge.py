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
        self._sc = sc_dict          # {id: SyncCrazyflie}
        self._drones = config['drones']
        self._ctrl = config['control']
        self._safety = config['safety']

        self._cmd_type = self._ctrl.get('command_type', 'velocity')
        self._u_max = float(self._safety['max_velocity_mps'])

        # Leader 列表（不应收到低级速度命令，以免覆盖高级控制器）
        self._leader_ids = set(config['formation']['leader_indices'])

        # 记录每架飞机最后下发命令的时间（防止 commander watchdog 超时）
        self._last_cmd_t: dict[int, float] = {d['id']: 0.0 for d in self._drones}

        # Leader 悬停目标位置（takeoff 后由 go_to 锁定）
        self._leader_hover_pos: dict[int, tuple] = {}

    # ─────────────────────────────────────────
    # 起飞 / 降落
    # ─────────────────────────────────────────

    def takeoff_all(self, height_m: float = None, duration_s: float = 3.0,
                    max_retries: int = 3):
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

        # 起飞前多轮确认 enHighLevel = 1（对抗 radio 丢包）
        import struct
        from cflib.crtp.crtpstack import CRTPPacket, CRTPPort
        PARAM_WRITE_CH = 2
        for _round in range(3):
            for drone in self._drones:
                sc = self._sc.get(drone['id'])
                if sc is None:
                    continue
                try:
                    element = sc.cf.param.toc.get_element_by_complete_name(
                        'commander.enHighLevel')
                    if element:
                        pk = CRTPPacket()
                        pk.set_header(CRTPPort.PARAM, PARAM_WRITE_CH)
                        pk.data = struct.pack('<H', element.ident)
                        pk.data += struct.pack('<B', 1)
                        sc.cf.send_packet(pk)
                except Exception:
                    pass
            time.sleep(0.15)
        time.sleep(0.2)  # 等参数生效

        # 重复发送起飞命令以对抗单 radio 的包丢失
        for attempt in range(max_retries):
            for drone in self._drones:
                did = drone['id']
                h = height_m if height_m is not None else drone['takeoff_height_m']
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
                    logger.warning(f"[CommandBridge] drone {did} 起飞命令失败 (attempt {attempt}): {e}")
            time.sleep(0.3)  # 每轮间隔，让 radio 有时间处理

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
            did = drone['id']
            if did not in self._leader_ids:
                continue
            sc = self._sc.get(did)
            if sc is None:
                continue
            x, y, z = float(positions[did, 0]), float(positions[did, 1]), float(positions[did, 2])
            h = drone.get('takeoff_height_m', 0.4)
            # 用 takeoff_height 而非当前 z（起飞过程中 z 可能还没到位）
            target_z = max(z, h)
            try:
                sc.cf.high_level_commander.go_to(x, y, target_z, 0.0, 1.0)
                self._leader_hover_pos[did] = (x, y, target_z)
                logger.info(f"[CommandBridge] leader {did} 锁定位置 ({x:.2f},{y:.2f},{target_z:.2f})")
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
        except Exception:
            pass

    def land_all(self, height_m: float = 0.05, duration_s: float = 3.0):
        """全机降落（使用 HighLevelCommander），重复发送以对抗丢包。"""
        if not _CFLIB_OK:
            logger.warning("[CommandBridge] MOCK 模式：land_all() 空操作")
            return

        logger.info("[CommandBridge] 全机降落...")
        for attempt in range(3):
            for drone in self._drones:
                did = drone['id']
                sc = self._sc.get(did)
                if sc is None:
                    continue
                try:
                    sc.cf.high_level_commander.land(height_m, duration_s)
                    if attempt == 0:
                        logger.info(f"[CommandBridge] drone {did} 降落")
                except Exception as e:
                    logger.warning(f"[CommandBridge] drone {did} 降落命令失败: {e}")
            time.sleep(0.3)

        time.sleep(duration_s)
        logger.info("[CommandBridge] 全机降落完成")

    # ─────────────────────────────────────────
    # 速度 setpoint（主控制命令）
    # ─────────────────────────────────────────

    def send_follower_velocities(self,
                                 follower_ids: list,
                                 velocities: np.ndarray):
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
            vx, vy, vz = float(velocities[i, 0]), float(velocities[i, 1]), float(velocities[i, 2])
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

    def hover_all(self):
        """向所有 Follower 发送零速度命令（就地悬停）。
        Leader 保持 HighLevelCommander 的自主位置保持，不发送低级命令。
        """
        if not _CFLIB_OK:
            logger.warning("[CommandBridge] MOCK 模式：hover_all() 空操作")
            return

        for drone in self._drones:
            did = drone['id']
            if did in self._leader_ids:
                continue  # Leader 由高级控制器自主保持位置
            sc = self._sc.get(did)
            if sc is None:
                continue
            try:
                sc.cf.commander.send_velocity_world_setpoint(0.0, 0.0, 0.0, 0.0)
                self._last_cmd_t[did] = time.time()
            except Exception as e:
                logger.warning(f"[CommandBridge] drone {did} 悬停命令失败: {e}")

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
            did = drone['id']
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
        for drone in self._drones:
            did = drone['id']
            if did in self._leader_ids:
                # Leader: 每 2 秒重发一次 go_to 防止高级控制器超时
                if now - self._last_cmd_t.get(did, 0.0) > 2.0:
                    pos = self._leader_hover_pos.get(did)
                    if pos is not None:
                        sc = self._sc.get(did)
                        if sc is not None:
                            try:
                                sc.cf.high_level_commander.go_to(
                                    pos[0], pos[1], pos[2], 0.0, 2.0)
                                self._last_cmd_t[did] = now
                            except Exception:
                                pass
            else:
                if now - self._last_cmd_t.get(did, 0.0) > watchdog_interval_s:
                    self.send_drone_velocity(did, 0.0, 0.0, 0.0)
