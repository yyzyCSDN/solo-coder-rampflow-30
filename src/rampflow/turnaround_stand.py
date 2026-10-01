from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Iterable

from .models import Decision
from .stand_orchestration import (
    StandAssignment,
    StandConflict,
    StandDemand,
    StandOrchestrator,
)
from .turnaround import TurnPlan, TurnTask, TurnaroundPlanner


@dataclass(frozen=True)
class TurnaroundStandPackage:
    flight_id: str
    demand: StandDemand
    tasks: tuple[TurnTask, ...]
    stand: StandAssignment
    turn_plan: TurnPlan


class TurnaroundStandCoordinator:
    """Keep a turnaround stand and its task-resource reservations in one package."""

    def __init__(self, stands: StandOrchestrator, turnaround: TurnaroundPlanner):
        self.stands = stands
        self.turnaround = turnaround
        self.packages: dict[str, TurnaroundStandPackage] = {}

    def _validate_package(self, package: TurnaroundStandPackage) -> None:
        if package.flight_id != package.demand.flight_id:
            raise ValueError("package flight ids differ")
        if package.flight_id != package.turn_plan.flight_id:
            raise ValueError("package flight ids differ")

    def plan(
        self,
        demand: StandDemand,
        tasks: Iterable[TurnTask],
        anchor: datetime,
        departure: datetime,
    ) -> TurnaroundStandPackage:
        task_tuple = tuple(tasks)
        turn_plan = self.turnaround.plan(
            demand.flight_id,
            task_tuple,
            anchor,
            departure,
        )
        stand = self.stands.plan(demand)
        package = TurnaroundStandPackage(
            demand.flight_id,
            demand,
            task_tuple,
            stand,
            turn_plan,
        )
        self._validate_package(package)
        return package

    def _snapshot(self) -> tuple[object, ...]:
        return (
            dict(self.stands.assignments),
            {sid: list(items) for sid, items in self.stands.by_stand.items()},
            self.stands.generation,
            dict(self.turnaround.plans),
            *self.turnaround.ledger.export_state(),
        )

    def _restore(self, snapshot: tuple[object, ...]) -> None:
        (
            assignments,
            by_stand,
            stand_generation,
            plans,
            ledger_rows,
            ledger_generation,
            next_token,
        ) = snapshot
        self.stands.assignments = dict(assignments)
        self.stands.by_stand = {sid: list(items) for sid, items in by_stand.items()}
        self.stands.generation = stand_generation
        self.turnaround.plans = dict(plans)
        self.turnaround.ledger.restore_state(
            (ledger_rows, ledger_generation, next_token)
        )

    def _stale(self, package: TurnaroundStandPackage) -> Decision | None:
        if package.stand.generation != self.stands.generation:
            return Decision(
                False,
                "stand-generation",
                "stand plan changed since planning; reassess target stand",
            )
        if package.turn_plan.resource_generation != self.turnaround.ledger.generation:
            return Decision(
                False,
                "resource-generation",
                "service resources changed since planning; full replan required",
            )
        return None

    def commit(self, package: TurnaroundStandPackage) -> Decision:
        self._validate_package(package)
        existing = self.packages.get(package.flight_id)
        if existing is not None:
            if existing == package:
                return Decision(True, "ok", "package already committed")
            return Decision(False, "duplicate", "flight already has a committed package")

        stale = self._stale(package)
        if stale is not None:
            return stale

        snapshot = self._snapshot()
        try:
            turn_decision = self.turnaround.commit(package.turn_plan)
            if not turn_decision.accepted:
                self._restore(snapshot)
                return turn_decision
            stand_assignment = replace(
                package.stand,
                generation=self.stands.generation,
            )
            try:
                self.stands.commit(stand_assignment)
            except StandConflict as error:
                self._restore(snapshot)
                return Decision(False, "stand-conflict", str(error))
            committed = replace(
                package,
                stand=stand_assignment,
                turn_plan=self.turnaround.plans[package.flight_id],
            )
            self.packages[package.flight_id] = committed
            return Decision(
                True,
                "ok",
                "stand and service resources committed",
                {
                    "stand_id": committed.stand.stand_id,
                    "resource_generation": committed.turn_plan.resource_generation,
                },
            )
        except Exception:
            self._restore(snapshot)
            raise

    def reassess(
        self,
        flight_id: str,
        demand: StandDemand | None = None,
    ) -> Decision:
        package = self.packages.get(flight_id)
        if package is None:
            return Decision(False, "unknown-flight", "flight has no committed package")
        demand = demand or package.demand
        current_plan = self.turnaround.plans[flight_id]
        if current_plan.resource_generation != self.turnaround.ledger.generation:
            return Decision(
                False,
                "resource-generation",
                "service resources changed; full package replan required",
                {"required": "full-replan"},
            )
        current = self.stands.assignments[flight_id]
        options = self.stands.candidates(demand, ignore=flight_id)
        if any(option.stand_id == current.stand_id for option in options):
            return Decision(
                True,
                "ok",
                "current stand remains feasible",
                {"stand_id": current.stand_id, "required": "none"},
            )
        if options:
            return Decision(
                True,
                "stand-change-required",
                "move the aircraft to another stand; task resources remain valid",
                {
                    "from_stand": current.stand_id,
                    "to_stand": options[0].stand_id,
                    "required": "stand-only",
                },
            )
        return Decision(
            False,
            "full-replan-required",
            "no compatible stand is available; release and rebuild the whole package",
            {"from_stand": current.stand_id, "required": "full-replan"},
        )

    def change_stand(
        self,
        flight_id: str,
        new_stand_id: str | None = None,
        demand: StandDemand | None = None,
    ) -> TurnaroundStandPackage:
        assessment = self.reassess(flight_id, demand)
        if not assessment.accepted:
            raise StandConflict(assessment.reason)
        current_package = self.packages[flight_id]
        current = self.stands.assignments[flight_id]
        demand = demand or current_package.demand
        if new_stand_id is None:
            if assessment.code == "ok":
                return current_package
            new_stand_id = str(assessment.details["to_stand"])
        if new_stand_id == current.stand_id and demand == current_package.demand:
            return current_package
        if new_stand_id not in self.stands.profiles:
            raise KeyError(new_stand_id)
        if new_stand_id == current.stand_id:
            return current_package

        options = {
            option.stand_id: option
            for option in self.stands.candidates(demand, ignore=flight_id)
        }
        assignment = options.get(new_stand_id)
        if assignment is None:
            raise StandConflict(f"stand {new_stand_id} unavailable")

        snapshot = self._snapshot()
        try:
            self.stands.replace(flight_id, assignment)
            updated = replace(current_package, demand=demand, stand=assignment)
            self.packages[flight_id] = updated
            return updated
        except Exception:
            self._restore(snapshot)
            raise

    def replan(
        self,
        flight_id: str,
        demand: StandDemand | None = None,
        tasks: Iterable[TurnTask] | None = None,
        anchor: datetime | None = None,
        departure: datetime | None = None,
    ) -> TurnaroundStandPackage:
        current = self.packages.get(flight_id)
        if current is None:
            raise KeyError(flight_id)
        demand = demand or current.demand
        task_tuple = tuple(tasks if tasks is not None else current.tasks)
        anchor = anchor or current.turn_plan.anchor
        departure = departure or current.turn_plan.departure

        turn_plan = self.turnaround.plan(
            flight_id,
            task_tuple,
            anchor,
            departure,
        )
        stand = self.stands.plan(demand, flight_id)
        snapshot = self._snapshot()
        try:
            decision = self.turnaround.replace_plan(turn_plan)
            if not decision.accepted:
                self._restore(snapshot)
                raise StandConflict(decision.reason)
            self.stands.replace(
                flight_id,
                replace(stand, generation=self.stands.generation),
            )
            updated = TurnaroundStandPackage(
                flight_id,
                demand,
                task_tuple,
                self.stands.assignments[flight_id],
                self.turnaround.plans[flight_id],
            )
            self.packages[flight_id] = updated
            return updated
        except Exception:
            self._restore(snapshot)
            raise

    def release(self, flight_id: str) -> None:
        package = self.packages.get(flight_id)
        if package is None and flight_id not in self.stands.assignments:
            return
        snapshot = self._snapshot()
        try:
            self.turnaround.release(flight_id)
            self.stands.release(flight_id)
            self.packages.pop(flight_id, None)
        except Exception:
            self._restore(snapshot)
            raise
