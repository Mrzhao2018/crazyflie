"""
test_three_radios.py - 三个 Crazyradio / Crazyflie 的官方 Swarm 稳定性测试

用途：
  使用 Bitcraze 官方推荐的 Swarm + CachedCfFactory 方式，同时测试
  3 条 Crazyradio → Crazyflie 链路。测试重点是：

  1. 3 条链路能否都被 Swarm 正常建立
  2. 每架飞机能否收到至少一帧 stateEstimate 日志
  3. 在保持连接期间是否出现异常掉线
  4. 最终断开是脚本主动结束，还是链路异常中断

默认测试目标：
  radio://0/40/2M/E7E7E7E701
  radio://1/60/2M/E7E7E7E701
  radio://2/80/2M/E7E7E7E701

运行方式：
  cd e:/crazyflie
  python integration/scripts/test_three_radios.py --hold-time 20

说明：
  - URI 第一段 0/1/2 是 Crazyradio dongle index，不是飞机编号。
  - 本脚本使用 Swarm 统一管理连接，避免手写多线程 open_link()。
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

logger = logging.getLogger("test_three_radios")

if TYPE_CHECKING:
    import cflib.crtp as cflib_crtp_module
    from cflib.crazyflie.log import LogConfig as LogConfigType
    from cflib.crazyflie.swarm import CachedCfFactory as CachedCfFactoryType
    from cflib.crazyflie.swarm import Swarm as SwarmType
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie as SyncCrazyflieType

try:
    import cflib.crtp as cflib_crtp
    from cflib.crazyflie.log import LogConfig
    from cflib.crazyflie.swarm import CachedCfFactory, Swarm

    _CFLIB_OK = True
except ImportError:
    cflib_crtp = None
    LogConfig = None
    CachedCfFactory = None
    Swarm = None
    _CFLIB_OK = False


DEFAULT_URIS = [
    "radio://0/40/2M/E7E7E7E701",
    "radio://1/60/2M/E7E7E7E701",
    "radio://2/80/2M/E7E7E7E701",
]


@dataclass(slots=True)
class LinkTestResult:
    """单条 Crazyradio → Crazyflie 链路测试结果。"""

    name: str
    uri: str
    open_ok: bool = False
    log_ok: bool = False
    stable_ok: bool = False
    connected_event: bool = False
    fully_connected_event: bool = False
    disconnected_event: bool = False
    connection_failed_event: bool = False
    connection_lost_event: bool = False
    disconnected_after_intentional_close: bool = False
    was_closed_by_script: bool = False
    last_sample: dict[str, float] = field(default_factory=dict)
    hold_duration_s: float = 0.0
    event_log: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def passed(self) -> bool:
        return (
            self.open_ok
            and self.log_ok
            and self.stable_ok
            and not self.connection_lost_event
            and not self.connection_failed_event
        )


class RadioLinkTester:
    """使用官方 Swarm + CachedCfFactory 进行多链路稳定性测试。"""

    def __init__(
        self,
        uris: list[str],
        log_timeout_s: float = 5.0,
        hold_time_s: float = 20.0,
        status_interval_s: float = 1.0,
    ):
        self._uris = uris
        self._log_timeout_s = log_timeout_s
        self._hold_time_s = hold_time_s
        self._status_interval_s = status_interval_s
        self._results_by_uri: dict[str, LinkTestResult] = {
            uri: LinkTestResult(name=f"radio{index}", uri=uri)
            for index, uri in enumerate(uris)
        }

    def run(self) -> list[LinkTestResult]:
        if cflib_crtp is None or CachedCfFactory is None or Swarm is None:
            raise RuntimeError("cflib Swarm 组件不可用")

        cflib_crtp.init_drivers()
        factory = CachedCfFactory(rw_cache="./cache")

        logger.info("使用 Swarm + CachedCfFactory 建立 3 条链路")
        with Swarm(self._uris, factory=factory) as swarm:
            logger.info("Swarm 已建立所有链路，开始逐机安装回调")
            swarm.sequential(self._prepare_link)

            logger.info("开始并行等待首包日志")
            swarm.parallel_safe(self._wait_for_first_log)

            logger.info("开始并行稳定性保持测试")
            swarm.parallel_safe(self._hold_connection)

            logger.info("标记脚本即将主动关闭 Swarm 链路")
            swarm.sequential(self._mark_intentional_close)

        for result in self._results_by_uri.values():
            result.was_closed_by_script = True
            if result.disconnected_event and not result.connection_lost_event:
                result.disconnected_after_intentional_close = True

        return [self._results_by_uri[uri] for uri in self._uris]

    def _prepare_link(self, scf: SyncCrazyflieType) -> None:
        if LogConfig is None:
            raise RuntimeError("LogConfig 不可用")

        uri = scf.cf.link_uri
        result = self._results_by_uri[uri]
        state: dict[str, object] = {}
        setattr(scf, "_radio_test_state", state)

        connected_event = threading.Event()
        fully_connected_event = threading.Event()
        log_event = threading.Event()
        lost_event = threading.Event()
        state["connected_event"] = connected_event
        state["fully_connected_event"] = fully_connected_event
        state["log_event"] = log_event
        state["lost_event"] = lost_event
        state["intentional_close"] = False

        result.open_ok = True
        result.connected_event = True
        result.fully_connected_event = True

        def _record_event(message: str) -> None:
            result.event_log.append(message)
            logger.info("[%s] %s", result.name, message)

        def _on_connected(link_uri: str) -> None:
            result.connected_event = True
            connected_event.set()
            _record_event(f"CONNECTED {link_uri}")

        def _on_fully_connected(link_uri: str) -> None:
            result.fully_connected_event = True
            result.open_ok = True
            fully_connected_event.set()
            _record_event(f"FULLY_CONNECTED {link_uri}")

        def _on_connection_failed(link_uri: str, msg: str) -> None:
            result.connection_failed_event = True
            if not result.error:
                result.error = f"connection_failed: {msg}"
            lost_event.set()
            fully_connected_event.set()
            _record_event(f"CONNECTION_FAILED {link_uri} — {msg}")

        def _on_connection_lost(link_uri: str, msg: str) -> None:
            result.connection_lost_event = True
            if not result.error:
                result.error = f"connection_lost: {msg}"
            lost_event.set()
            _record_event(f"CONNECTION_LOST {link_uri} — {msg}")

        def _on_disconnected(link_uri: str) -> None:
            result.disconnected_event = True
            intentional_close = bool(state["intentional_close"])
            result.disconnected_after_intentional_close = intentional_close
            suffix = (
                "CLOSED_BY_SCRIPT" if intentional_close else "DISCONNECTED_UNEXPECTEDLY"
            )
            _record_event(f"DISCONNECTED {link_uri} — {suffix}")
            if not intentional_close:
                lost_event.set()

        def _on_data(
            _timestamp: int,
            data: dict[str, float],
            _logconf: LogConfigType,
        ) -> None:
            result.last_sample = {
                "x": float(data["stateEstimate.x"]),
                "y": float(data["stateEstimate.y"]),
                "z": float(data["stateEstimate.z"]),
            }
            if not result.log_ok:
                _record_event(
                    "FIRST_LOG "
                    f"x={result.last_sample['x']:.3f}, "
                    f"y={result.last_sample['y']:.3f}, "
                    f"z={result.last_sample['z']:.3f}"
                )
            log_event.set()

        def _on_log_error(_logconf: LogConfigType, msg: str) -> None:
            if not result.error:
                result.error = f"日志订阅失败: {msg}"
            log_event.set()

        scf.cf.connected.add_callback(_on_connected)
        if hasattr(scf.cf, "fully_connected"):
            scf.cf.fully_connected.add_callback(_on_fully_connected)
        else:
            result.open_ok = True
            result.fully_connected_event = True
            fully_connected_event.set()
        scf.cf.connection_failed.add_callback(_on_connection_failed)
        scf.cf.connection_lost.add_callback(_on_connection_lost)
        scf.cf.disconnected.add_callback(_on_disconnected)

        logconf = LogConfig(name=f"HealthCheck_{result.name}", period_in_ms=100)
        logconf.add_variable("stateEstimate.x", "float")
        logconf.add_variable("stateEstimate.y", "float")
        logconf.add_variable("stateEstimate.z", "float")
        scf.cf.log.add_config(logconf)
        logconf.data_received_cb.add_callback(_on_data)
        logconf.error_cb.add_callback(_on_log_error)
        state["logconf"] = logconf

        connected_event.set()
        fully_connected_event.set()

        _record_event(f"SWARM_MEMBER_READY {uri}")

    def _wait_for_first_log(self, scf: SyncCrazyflieType) -> None:
        uri = scf.cf.link_uri
        result = self._results_by_uri[uri]
        state = self._get_state(scf)
        fully_connected_event = self._get_event(state, "fully_connected_event")
        log_event = self._get_event(state, "log_event")
        lost_event = self._get_event(state, "lost_event")
        logconf = self._get_logconf(state)

        if not fully_connected_event.wait(timeout=self._log_timeout_s):
            if not result.error:
                result.error = (
                    f"在 {self._log_timeout_s:.1f}s 内未等到 fully_connected，"
                    "链路初始化可能卡住"
                )
            return

        if result.connection_failed_event or result.connection_lost_event:
            return

        logger.info("[%s] FULLY_CONNECTED 后开始等待 stateEstimate", result.name)
        logconf.start()

        if lost_event.wait(timeout=0.05):
            return

        if log_event.wait(timeout=self._log_timeout_s):
            result.log_ok = bool(result.last_sample)
            if result.log_ok:
                logger.info(
                    "[%s] 收到 stateEstimate: x=%.3f y=%.3f z=%.3f",
                    result.name,
                    result.last_sample["x"],
                    result.last_sample["y"],
                    result.last_sample["z"],
                )
            elif not result.error:
                result.error = "收到了日志事件，但没有有效 stateEstimate 数据"
        else:
            result.error = (
                f"在 {self._log_timeout_s:.1f}s 内未收到 stateEstimate 日志，"
                "请检查飞机是否开机、地址/信道是否匹配、定位日志是否可用"
            )

    def _hold_connection(self, scf: SyncCrazyflieType) -> None:
        uri = scf.cf.link_uri
        result = self._results_by_uri[uri]
        state = self._get_state(scf)
        lost_event = self._get_event(state, "lost_event")

        if (
            not result.log_ok
            or result.connection_lost_event
            or result.connection_failed_event
        ):
            return

        self._record(result, f"HOLD_START duration={self._hold_time_s:.1f}s")
        hold_start = time.time()
        next_status_time = hold_start + self._status_interval_s
        hold_ok = True

        while time.time() - hold_start < self._hold_time_s:
            if lost_event.wait(timeout=0.1):
                hold_ok = False
                break

            now = time.time()
            if now >= next_status_time:
                elapsed = now - hold_start
                logger.info(
                    "[%s] HOLDING %.1fs / %.1fs",
                    result.name,
                    elapsed,
                    self._hold_time_s,
                )
                next_status_time += self._status_interval_s

        result.hold_duration_s = time.time() - hold_start
        result.stable_ok = (
            hold_ok
            and not result.connection_lost_event
            and not result.connection_failed_event
        )

        if result.stable_ok:
            self._record(result, f"HOLD_OK duration={result.hold_duration_s:.1f}s")
        elif not result.error:
            result.error = (
                f"保持连接阶段在 {result.hold_duration_s:.1f}s 时中断，"
                "更像是异常掉线而不是脚本主动关闭"
            )

    def _mark_intentional_close(self, scf: SyncCrazyflieType) -> None:
        state = self._get_state(scf)
        state["intentional_close"] = True

    @staticmethod
    def _get_state(scf: SyncCrazyflieType) -> dict[str, object]:
        state = getattr(scf, "_radio_test_state", None)
        if not isinstance(state, dict):
            raise RuntimeError(f"{scf.cf.link_uri} 缺少测试状态")
        return state

    @staticmethod
    def _get_event(state: dict[str, object], key: str) -> threading.Event:
        event = state.get(key)
        if not isinstance(event, threading.Event):
            raise RuntimeError(f"测试状态缺少事件: {key}")
        return event

    @staticmethod
    def _get_logconf(state: dict[str, object]) -> LogConfigType:
        if LogConfig is None:
            raise RuntimeError("LogConfig 不可用")
        logconf = state.get("logconf")
        if not isinstance(logconf, LogConfig):
            raise RuntimeError("测试状态缺少 LogConfig")
        return logconf

    @staticmethod
    def _record(result: LinkTestResult, message: str) -> None:
        result.event_log.append(message)
        logger.info("[%s] %s", result.name, message)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 Swarm + CachedCfFactory 测试 1 到 3 个 Crazyradio 是否稳定",
    )
    parser.add_argument(
        "--uri",
        action="append",
        dest="uris",
        help=(
            "要测试的 radio URI。可重复传 1 到 3 次；不传时使用默认的 3 条 URI 配置。"
        ),
    )
    parser.add_argument(
        "--log-timeout",
        type=float,
        default=5.0,
        help="等待 fully_connected / stateEstimate 首包的超时时间（秒）",
    )
    parser.add_argument(
        "--hold-time",
        type=float,
        default=20.0,
        help="首包日志收到后继续保持连接的测试时长（秒）",
    )
    parser.add_argument(
        "--status-interval",
        type=float,
        default=1.0,
        help="保持连接阶段的状态打印间隔（秒）",
    )
    return parser


def _validate_uris(uris: list[str]) -> list[str]:
    if not 1 <= len(uris) <= 3:
        raise ValueError(f"需要 1 到 3 个 URI，当前收到 {len(uris)} 个")
    return uris


def _print_summary(results: list[LinkTestResult], elapsed_s: float) -> int:
    total_count = len(results)
    print("\n" + "=" * 72)
    print(f"Crazyradio {total_count} 链路 Swarm 测试结果")
    print("=" * 72)

    passed_count = 0
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.uri}")
        print(f"       - swarm_open: {'OK' if result.open_ok else 'FAIL'}")
        print(f"       - stateEstimate: {'OK' if result.log_ok else 'FAIL'}")
        print(
            f"       - stable_hold: {'OK' if result.stable_ok else 'FAIL'} ({result.hold_duration_s:.1f}s)"
        )
        print(f"       - connected_event: {'YES' if result.connected_event else 'NO'}")
        print(
            f"       - fully_connected: {'YES' if result.fully_connected_event else 'NO'}"
        )
        print(
            f"       - connection_failed: {'YES' if result.connection_failed_event else 'NO'}"
        )
        print(
            f"       - connection_lost: {'YES' if result.connection_lost_event else 'NO'}"
        )
        print(
            "       - close_type: "
            f"{'SCRIPT_CLOSED' if result.disconnected_after_intentional_close else 'NOT_CONFIRMED_SCRIPT_CLOSE'}"
        )
        if result.last_sample:
            print(
                "       - sample: "
                f"x={result.last_sample['x']:.3f}, "
                f"y={result.last_sample['y']:.3f}, "
                f"z={result.last_sample['z']:.3f}"
            )
        if result.event_log:
            print("       - events:")
            for event in result.event_log:
                print(f"         * {event}")
        if result.error:
            print(f"       - error: {result.error}")
        if result.passed:
            passed_count += 1

    print("-" * 72)
    print(f"总耗时: {elapsed_s:.2f}s")
    print(f"通过数量: {passed_count}/{total_count}")

    if passed_count == total_count:
        print(
            f"结论: 这 {total_count} 条 Crazyradio 链路都能通过官方 Swarm 方式稳定建立链路并保持连接。"
        )
        return 0

    print(
        "结论: 至少有 1 个 Crazyradio / URI 组合在官方 Swarm 测试下未通过，请按 FAIL 项排查。"
    )
    return 1


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not _CFLIB_OK:
        logger.error("未检测到 cflib 或 Swarm 组件。请先执行: pip install cflib")
        return 2

    parser = _build_argparser()
    args = parser.parse_args()
    uris = _validate_uris(args.uris or DEFAULT_URIS)

    logger.info("开始三链路 Swarm 测试，目标 URI:")
    for uri in uris:
        logger.info("  %s", uri)

    t0 = time.time()
    tester = RadioLinkTester(
        uris=uris,
        log_timeout_s=args.log_timeout,
        hold_time_s=args.hold_time,
        status_interval_s=args.status_interval,
    )

    try:
        results = tester.run()
    except Exception as exc:
        logger.error("Swarm 测试执行失败: %s", exc)
        return 1

    elapsed_s = time.time() - t0
    return _print_summary(results, elapsed_s)


if __name__ == "__main__":
    raise SystemExit(main())
