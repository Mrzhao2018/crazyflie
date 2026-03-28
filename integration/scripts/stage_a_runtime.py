from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from stage_a_execution import _RadioGroupExecutor, _SafetyArbiter

if TYPE_CHECKING:
    from cf_command_bridge import CommandBridge
    from formation_runner import FormationRunner
    from safety_guard import SafetyStatus


@dataclass(frozen=True)
class StageARuntimeSnapshot:
    raw_state: dict
    positions: np.ndarray
    disconnected_ids: tuple[int, ...]
    radio_groups: dict[tuple[int, int], tuple[int, ...]]


@dataclass(frozen=True)
class CoordinatorLeaderUpdate:
    should_update: bool
    target_drone_id: int | None = None
    target_position: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class CoordinatorCommandIntent:
    mode: str
    target_drone_id: int | None = None
    target_follower_index: int | None = None
    hover_ids: tuple[int, ...] = ()
    follower_intent: "FollowerIntent | None" = None


@dataclass(frozen=True)
class FollowerIntent:
    target_drone_id: int
    target_follower_index: int
    target_position: tuple[float, float, float]
    velocity_world: tuple[float, float, float]
    hover_ids: tuple[int, ...]
    refresh_reason: str
    requires_transport_refresh: bool
    created_at: float
    stale_after_s: float


@dataclass
class SubgroupControllerState:
    active_follower_id: int | None = None
    active_follower_index: int | None = None
    parked_follower_ids: tuple[int, ...] = ()
    retained_intent: FollowerIntent | None = None
    last_refresh_at: float = 0.0
    last_transport_send_at: float = 0.0
    last_applied_mode: str | None = None
    last_refresh_reason: str | None = None
    initialized: bool = False


@dataclass(frozen=True)
class CoordinatorTickPlan:
    runtime_snapshot: StageARuntimeSnapshot
    positions: np.ndarray
    disconnected_ids: tuple[int, ...]
    t_elapsed: float
    leader_update: CoordinatorLeaderUpdate
    desired_follower_velocities: np.ndarray
    command_intent: CoordinatorCommandIntent


class _StateAggregator:
    def __init__(self, runner: "FormationRunner"):
        self._runner = runner

    def build_snapshot(self, state: dict) -> StageARuntimeSnapshot:
        return self._runner._build_stage_a_runtime_snapshot(state)


class _MissionCoordinator:
    def __init__(self, runner: "FormationRunner"):
        self._runner = runner

    def build_tick_plan(
        self, runtime_snapshot: StageARuntimeSnapshot, t_elapsed: float
    ) -> CoordinatorTickPlan:
        return self._runner._build_coordinator_tick_plan(runtime_snapshot, t_elapsed)

    def apply_leader_update(
        self, cmd_bridge: "CommandBridge", leader_update: CoordinatorLeaderUpdate
    ):
        self._runner._apply_coordinator_leader_update(cmd_bridge, leader_update)

    def finalize_command_intent(
        self,
        tick_plan: CoordinatorTickPlan,
        u_safe: np.ndarray,
    ) -> CoordinatorCommandIntent:
        return self._runner._finalize_command_intent(tick_plan, u_safe)


class _SubgroupControllerStateStore:
    def __init__(self, runner: "FormationRunner"):
        self._runner = runner
        self._single_follower_state = SubgroupControllerState()

    def initialize_single_follower(
        self,
        active_follower_id: int,
        active_follower_index: int,
        parked_follower_ids: tuple[int, ...],
    ):
        self._single_follower_state = SubgroupControllerState(
            active_follower_id=active_follower_id,
            active_follower_index=active_follower_index,
            parked_follower_ids=parked_follower_ids,
            retained_intent=None,
            last_refresh_at=0.0,
            last_transport_send_at=0.0,
            last_applied_mode=None,
            last_refresh_reason="startup",
            initialized=True,
        )

    def get_single_follower_state(self) -> SubgroupControllerState:
        return self._single_follower_state

    def retain_single_follower_intent(self, intent: FollowerIntent):
        self._single_follower_state.retained_intent = intent
        self._single_follower_state.last_refresh_reason = intent.refresh_reason

    def mark_single_follower_refresh(self, applied_at: float, mode: str):
        self._single_follower_state.last_refresh_at = applied_at
        self._single_follower_state.last_applied_mode = mode

    def mark_single_follower_transport_send(self, sent_at: float):
        self._single_follower_state.last_transport_send_at = sent_at


@dataclass(frozen=True)
class StageARuntimeComponents:
    state_aggregator: _StateAggregator
    mission_coordinator: _MissionCoordinator
    subgroup_controller_state: _SubgroupControllerStateStore
    safety_arbiter: _SafetyArbiter
    radio_group_executor: _RadioGroupExecutor
