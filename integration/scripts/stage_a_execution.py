from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cf_command_bridge import CommandBridge
    from formation_runner import FormationRunner
    from safety_guard import SafetyStatus
    from stage_a_runtime import (
        CoordinatorCommandIntent,
        CoordinatorTickPlan,
        StageARuntimeSnapshot,
    )


@dataclass(frozen=True)
class SafetyExecutionDecision:
    action: str
    command_intent: CoordinatorCommandIntent | None = None
    message: str | None = None


@dataclass(frozen=True)
class SubgroupExecutionResult:
    radio_group: tuple[int, int]
    executed_ids: tuple[int, ...]
    action: str


@dataclass(frozen=True)
class CommandExecutionResult:
    action: str
    subgroup_results: tuple[SubgroupExecutionResult, ...]


class _SafetyArbiter:
    def __init__(self, runner: "FormationRunner"):
        self._runner = runner

    def resolve(
        self,
        safety_status: "SafetyStatus",
        tick_plan: "CoordinatorTickPlan",
    ) -> SafetyExecutionDecision:
        return self._runner._resolve_safety_execution_decision(safety_status, tick_plan)


class _RadioGroupExecutor:
    def __init__(self, runner: "FormationRunner"):
        self._runner = runner

    def execute(
        self,
        cmd_bridge: "CommandBridge",
        runtime_snapshot: "StageARuntimeSnapshot",
        u_safe: np.ndarray,
        command_intent: "CoordinatorCommandIntent",
        step: int,
    ) -> CommandExecutionResult:
        return self._runner._apply_command_intent(
            cmd_bridge,
            runtime_snapshot,
            u_safe,
            command_intent,
            step,
        )

    def finalize_command_intent(
        self,
        tick_plan: "CoordinatorTickPlan",
        u_safe: np.ndarray,
    ):
        return self._runner._finalize_command_intent(tick_plan, u_safe)
