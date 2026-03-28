"""
pose_bridge.py - 定位桥接模块

职责：
  订阅每架 Crazyflie 的机载 stateEstimate 日志（Lighthouse/LPS 定位），
  整理成算法所需的 (n, 3) 位置/速度矩阵和统一时间戳。

接口：
  bridge = PoseBridge(config)
  bridge.start()          # 建立连接并开始接收数据
  state = bridge.get_latest_state()   # 获取最新状态
  bridge.stop()           # 关闭所有连接

state 字典格式：
  {
    'positions'  : np.ndarray (n, 3),  # 算法坐标系位置
    'velocities' : np.ndarray (n, 3),  # 算法坐标系速度
    'timestamp'  : float,              # 最后一次全量刷新时间 (time.time())
    'per_drone'  : {
        id: {'pos': (3,), 'vel': (3,), 't': float, 'fresh': bool}
    }
  }

依赖：
  pip install cflib
"""

import threading
import time
import logging
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 尝试导入 cflib（未安装时给出友好提示）
# ─────────────────────────────────────────────
try:
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.log import LogConfig
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    from cflib.crtp.crtpstack import CRTPPacket, CRTPPort

    _CFLIB_OK = True
except ImportError:
    _CFLIB_OK = False
    cflib = None
    Crazyflie = None
    LogConfig = None
    SyncCrazyflie = None
    CRTPPacket = None
    CRTPPort = None
    logger.warning("cflib 未安装，PoseBridge 将在 MOCK 模式下运行（仅供调试）。")


class _DroneState:
    """单架飞机的线程安全状态缓存。"""

    def __init__(self, drone_id: int, timeout_s: float = 0.5):
        self.id = drone_id
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._pos = np.zeros(3)
        self._vel = np.zeros(3)
        self._t = 0.0

    def update(self, x, y, z, vx=0.0, vy=0.0, vz=0.0):
        with self._lock:
            self._pos[:] = [x, y, z]
            self._vel[:] = [vx, vy, vz]
            self._t = time.time()

    def read(self):
        with self._lock:
            return (
                self._pos.copy(),
                self._vel.copy(),
                self._t,
                (time.time() - self._t) < self.timeout_s and self._t > 0,
            )


class PoseBridge:
    """
    从 Crazyflie 机载 stateEstimate 日志读取定位数据。

    Parameters
    ----------
    config : dict
        fleet_config.json 的完整解析结果
    """

    def __init__(self, config: dict):
        self._cfg = config
        self._n = config["formation"]["n_agents"]
        self._timeout_s = config["safety"]["pose_timeout_s"]
        self._drones_cfg = config["drones"]
        comm_cfg = config.get("communication", {})
        self._pose_log_period_ms = int(comm_cfg.get("pose_log_period_ms", 100))
        self._startup_subscribe_spacing_s = float(
            comm_cfg.get("startup_subscribe_spacing_s", 0.0)
        )

        # 坐标变换参数
        ct = config["coordinate_transform"]
        self._R = np.array(ct["R"], dtype=float)  # (3,3)
        self._t = np.array(ct["t"], dtype=float)  # (3,)

        # 每架飞机的状态缓存
        self._states: dict[int, _DroneState] = {
            d["id"]: _DroneState(d["id"], self._timeout_s) for d in self._drones_cfg
        }

        # drone id → 行索引映射（支持 id 不连续的情况）
        self._id_to_row: dict[int, int] = {
            d["id"]: i for i, d in enumerate(self._drones_cfg)
        }

        # cflib 连接句柄（id -> SyncCrazyflie）
        self._scs: dict[int, Any] = {}
        # 每架飞机的 LogConfig 句柄（必须持有引用，避免日志订阅生命周期异常）
        self._log_configs: dict[int, Any] = {}
        # 每架飞机 fully_connected / 首包日志事件
        self._fully_connected_events: dict[int, threading.Event] = {}
        self._first_log_events: dict[int, threading.Event] = {}
        self._link_ok: dict[int, bool] = {d["id"]: False for d in self._drones_cfg}
        self._running = False

    # ─────────────────────────────────────────
    # 公开接口
    # ─────────────────────────────────────────

    def start(self):
        """初始化驱动并建立所有飞机的连接。"""
        if not _CFLIB_OK:
            logger.warning("MOCK 模式：PoseBridge.start() 空操作，请安装 cflib。")
            self._running = True
            return

        assert cflib is not None
        assert Crazyflie is not None
        assert SyncCrazyflie is not None

        cflib.crtp.init_drivers()
        did = -1
        try:
            # Phase 1: 打开所有链路并注册 fully_connected 事件
            for drone in self._drones_cfg:
                did = drone["id"]
                uri = drone["uri"]
                logger.info(f"[PoseBridge] 连接 drone {did}: {uri}")
                sc = SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache"))

                fully_connected = threading.Event()
                self._fully_connected_events[did] = fully_connected

                def _on_fully_connected(link_uri, _did=did):
                    logger.info(
                        f"[PoseBridge] drone {_did} fully connected: {link_uri}"
                    )
                    self._link_ok[_did] = True
                    fully_connected.set()

                def _on_connection_lost(link_uri, msg, _did=did):
                    self._link_ok[_did] = False
                    logger.error(
                        f"[PoseBridge] drone {_did} 连接丢失: {link_uri} ({msg})"
                    )

                def _on_disconnected(link_uri, _did=did):
                    self._link_ok[_did] = False
                    logger.info(f"[PoseBridge] drone {_did} 已断开: {link_uri}")

                sc.cf.fully_connected.add_callback(_on_fully_connected)
                sc.cf.connection_lost.add_callback(_on_connection_lost)
                sc.cf.disconnected.add_callback(_on_disconnected)
                sc.open_link()
                self._scs[did] = sc

            # Phase 2: 等待所有飞机 fully_connected，再做参数写入
            for drone in self._drones_cfg:
                did = drone["id"]
                fully_connected = self._fully_connected_events[did]
                if not fully_connected.wait(timeout=15.0):
                    raise TimeoutError(f"drone {did} 等待 fully connected 超时")

                sc = self._scs[did]

                # 非阻塞写 commander.enHighLevel = 1
                # set_value() 会阻塞等全部参数读完（4架时要等1分钟+），
                # 改用直接发 CRTP 包，fire-and-forget
                self._set_param_nonblocking(sc, "commander.enHighLevel", 1)
                time.sleep(0.05)  # 给 radio 时间发包

            # Phase 3: 所有飞机统一启动日志订阅，避免前面飞机过早开始 freshness 计时
            for drone in self._drones_cfg:
                did = drone["id"]
                sc = self._scs[did]
                self._subscribe_log(sc, did)
                logger.info(f"[PoseBridge] drone {did} 连接成功")
                if self._startup_subscribe_spacing_s > 0.0:
                    time.sleep(self._startup_subscribe_spacing_s)

            # Phase 4: 显式等待每架飞机至少收到一帧 stateEstimate
            self._wait_for_first_logs(timeout_s=10.0)
        except Exception as e:
            logger.error(f"[PoseBridge] drone {did} 连接失败: {e}，清理已建立连接...")
            self.stop()  # 清理已成功建立的连接，防止泄漏
            raise
        self._running = True

    def stop(self):
        """关闭所有 cflib 连接。"""
        self._running = False

        for did, lgc in self._log_configs.items():
            try:
                lgc.stop()
                logger.info(f"[PoseBridge] drone {did} 日志订阅已停止")
            except Exception as e:
                logger.warning(f"[PoseBridge] 停止 drone {did} 日志订阅时出错: {e}")
        self._log_configs.clear()
        self._fully_connected_events.clear()
        self._first_log_events.clear()

        for did, sc in self._scs.items():
            try:
                sc.close_link()
                self._link_ok[did] = False
                logger.info(f"[PoseBridge] drone {did} 连接已关闭")
            except Exception as e:
                logger.warning(f"[PoseBridge] 关闭 drone {did} 时出错: {e}")
        self._scs.clear()

    def get_latest_state(self) -> dict:
        """
        返回所有飞机的最新状态，已转换到算法坐标系。

        Returns
        -------
        dict with keys: positions (n,3), velocities (n,3), timestamp, per_drone
        """
        positions = np.zeros((self._n, 3))
        velocities = np.zeros((self._n, 3))
        per_drone = {}

        for drone in self._drones_cfg:
            did = drone["id"]
            row = self._id_to_row[did]  # 使用映射索引，支持 id 不连续
            pos_w, vel_w, t, fresh = self._states[did].read()
            # 坐标系变换
            pos_a = self._R @ pos_w + self._t
            vel_a = self._R @ vel_w
            positions[row] = pos_a
            velocities[row] = vel_a
            per_drone[did] = {"pos": pos_a, "vel": vel_a, "t": t, "fresh": fresh}

        return {
            "positions": positions,
            "velocities": velocities,
            "timestamp": time.time(),
            "per_drone": per_drone,
            "disconnected_ids": self.get_disconnected_ids(),
        }

    def is_all_fresh(self) -> bool:
        """返回所有飞机定位数据是否在超时窗口内。"""
        for drone in self._drones_cfg:
            did = drone["id"]
            _, _, t, fresh = self._states[did].read()
            if not fresh:
                logger.warning(f"[PoseBridge] drone {did} 定位数据已过期 (t={t:.3f})")
                return False
        return True

    def get_cf_connections(self) -> dict:
        """
        返回已建立的 SyncCrazyflie 连接字典 {drone_id: SyncCrazyflie}。
        供 CommandBridge 共享使用，避免对同一架飞机建立双重无线连接。
        """
        return self._scs

    def get_disconnected_ids(self) -> list[int]:
        """返回当前已知断链的飞机 id 列表。"""
        return [did for did, ok in self._link_ok.items() if not ok]

    def wait_until_fresh(self, timeout_s: float = 10.0) -> bool:
        """
        阻塞等待所有飞机都有新鲜定位数据。

        Returns
        -------
        bool: 超时前是否全部就绪
        """
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.is_all_fresh():
                return True
            time.sleep(0.05)
        logger.error("[PoseBridge] 等待定位数据超时，请检查定位系统。")
        return False

    # ─────────────────────────────────────────
    # 内部：非阻塞参数写入
    # ─────────────────────────────────────────

    @staticmethod
    def _set_param_nonblocking(sc, param_name: str, value: int):
        """
        通过直接发 CRTP 包写参数，不等待 All parameters updated。
        仅适用于 UINT8 类型参数。重复发 3 次提高可靠性。
        """
        import struct

        assert CRTPPacket is not None
        assert CRTPPort is not None

        PARAM_WRITE_CH = 2
        try:
            element = sc.cf.param.toc.get_element_by_complete_name(param_name)
            if element is None:
                logger.warning(f"[PoseBridge] 参数 {param_name} 不在 TOC 中")
                return
            for _ in range(3):
                pk = CRTPPacket()
                pk.set_header(CRTPPort.PARAM, PARAM_WRITE_CH)
                pk.data = struct.pack("<H", element.ident)
                pk.data += struct.pack("<B", value)
                sc.cf.send_packet(pk)
                time.sleep(0.05)
            logger.info(f"[PoseBridge] {param_name} = {value} (non-blocking)")
        except Exception as e:
            logger.warning(f"[PoseBridge] 写参数 {param_name} 失败: {e}")

    # ─────────────────────────────────────────
    # 内部：订阅日志
    # ─────────────────────────────────────────

    def _subscribe_log(self, sc, drone_id: int):
        """为一架飞机注册 stateEstimate 日志回调。"""
        assert LogConfig is not None
        first_log_event = threading.Event()
        self._first_log_events[drone_id] = first_log_event
        lgc = LogConfig(
            name=f"StateEst_{drone_id}", period_in_ms=self._pose_log_period_ms
        )
        lgc.add_variable("stateEstimate.x", "float")
        lgc.add_variable("stateEstimate.y", "float")
        lgc.add_variable("stateEstimate.z", "float")

        def _callback(timestamp, data, logconf, did=drone_id):
            try:
                self._states[did].update(
                    x=data["stateEstimate.x"],
                    y=data["stateEstimate.y"],
                    z=data["stateEstimate.z"],
                    vx=0.0,
                    vy=0.0,
                    vz=0.0,
                )
                first_log_event.set()
            except KeyError as e:
                logger.warning(f"[PoseBridge] drone {did} 数据键缺失: {e}")

        sc.cf.log.add_config(lgc)
        lgc.data_received_cb.add_callback(_callback)
        lgc.error_cb.add_callback(
            lambda lc, msg: logger.error(
                f"[PoseBridge] drone {drone_id} 日志错误: {msg}"
            )
        )
        self._log_configs[drone_id] = lgc
        lgc.start()
        logger.info(
            f"[PoseBridge] drone {drone_id} 日志订阅成功 "
            f"({1000.0 / self._pose_log_period_ms:.1f}Hz, position-only)"
        )

    def _wait_for_first_logs(self, timeout_s: float = 10.0):
        """等待每架飞机至少收到一帧 stateEstimate。"""
        for drone in self._drones_cfg:
            did = drone["id"]
            first_log_event = self._first_log_events.get(did)
            if first_log_event is None:
                raise RuntimeError(f"drone {did} 缺少首包日志事件")
            if not first_log_event.wait(timeout=timeout_s):
                raise TimeoutError(
                    f"drone {did} 在 {timeout_s:.1f}s 内未收到 stateEstimate 首包"
                )
            logger.info(f"[PoseBridge] drone {did} 已收到首帧 stateEstimate")
