from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping

from .ledger import Reservation
from .models import Aircraft, Flight
from .stands import StandAssignment, StandPlanner, StandUse
from .turnaround import TurnPlan, TurnTask, TurnaroundPlanner


@dataclass(frozen=True)
class StandPackage:
    """The stand reservation and support-resource reservations of one flight."""
    flight_id: str
    stand: Reservation[StandUse] | None
    turn_plan: TurnPlan | None
    turn_reservations: tuple[Reservation[str], ...]


@dataclass(frozen=True)
class PackageAssessment:
    """Decision for a package built against a stale generation or a blocked stand.

    action "swap" means the turnaround plan is stand-agnostic and an alternative
    stand is feasible for the same occupancy window, so the stand reservation can
    move on its own.  action "replan" means the support resources are coupled to
    the current stand (or no alternative stand exists), so the whole package must
    be released and planned again.
    """
    flight_id: str
    action: str
    reasons: tuple[str, ...]
    current_stand: str | None
    candidate_stand: str | None
    pinned_resources: tuple[str, ...]


@dataclass(frozen=True)
class PackageCommit:
    flight_id: str
    plan: TurnPlan
    stand: StandAssignment


class StandPackageError(RuntimeError):
    pass


class PackageRollbackError(StandPackageError):
    """Replan failed and the original package could not be fully restored."""

    def __init__(self, message: str, failures: Iterable[str]):
        self.failures = tuple(failures)
        super().__init__(message)


class StandPackageCoordinator:
    """Keeps a flight's stand and its support resources committed or released as one package.

    The stand ledger and the turnaround resource ledger are independent, so the
    coordinator sequences their updates: a stand-only swap is allowed only when no
    committed support resource is bound to the current stand, and a full replan
    releases the old package from both ledgers before committing the replacement,
    restoring the original package if the replacement cannot be committed.
    """

    def __init__(
        self,
        stands: StandPlanner,
        turns: TurnaroundPlanner,
        stand_resources: Mapping[str, Iterable[str]] | None = None,
    ):
        self.stands = stands
        self.turns = turns
        self.stand_resources = {
            stand_id: frozenset(resource_ids)
            for stand_id, resource_ids in (stand_resources or {}).items()
        }

    def package_for(self, flight_id: str) -> StandPackage:
        return StandPackage(
            flight_id,
            self.stands.reservation_for_flight(flight_id),
            self.turns.plans.get(flight_id),
            self.turns.reservations_for(flight_id),
        )

    def _pinned_resources(self, package: StandPackage) -> tuple[str, ...]:
        if package.stand is None or package.turn_plan is None:
            return ()
        bound = self.stand_resources.get(package.stand.resource_id, frozenset())
        if not bound:
            return ()
        used = {rid for run in package.turn_plan.runs.values() for rid in run.resources}
        return tuple(sorted(used & bound))

    def _flight_for(self, package: StandPackage, flight: Flight | None) -> Flight:
        if flight is not None:
            return flight
        if package.stand is None:
            raise StandPackageError("flight details required when no stand reservation exists")
        use = package.stand.payload
        wingspan = float(getattr(use, "_aircraft_wingspan_m", 36.0))
        aircraft = Aircraft(use.aircraft_reg, "UNKNOWN", wingspan, 20.0, 1.0, 1)
        return Flight(
            use.flight_id,
            aircraft,
            "",
            "",
            use.actual_window.start,
            use.actual_window.end,
            0,
            0,
            international=use.international,
        )

    def _best_alternative(self, package: StandPackage, flight: Flight | None) -> str | None:
        assert package.stand is not None
        use = package.stand.payload
        candidate_flight = self._flight_for(package, flight)
        ranked = self.stands.rank(
            candidate_flight,
            None,
            use.tow_in,
            use.tow_out,
            ignore_token=package.stand.token,
        )
        for option in ranked:
            if option.stand_id != package.stand.resource_id:
                return option.stand_id
        return None

    def assess(self, flight_id: str, flight: Flight | None = None) -> PackageAssessment:
        package = self.package_for(flight_id)
        if package.stand is None and package.turn_plan is None:
            raise KeyError(flight_id)
        current_stand = package.stand.resource_id if package.stand is not None else None
        pinned = self._pinned_resources(package)
        if pinned:
            return PackageAssessment(
                flight_id, "replan", ("stand-bound-resources",), current_stand, None, pinned
            )
        if package.stand is None:
            return PackageAssessment(
                flight_id, "replan", ("no-stand-reservation",), None, None, ()
            )
        candidate = self._best_alternative(package, flight)
        if candidate is not None:
            return PackageAssessment(
                flight_id, "swap", ("stand-only-move-feasible",), current_stand, candidate, ()
            )
        return PackageAssessment(
            flight_id, "replan", ("no-alternative-stand",), current_stand, None, ()
        )

    def swap(
        self,
        flight_id: str,
        new_stand_id: str | None = None,
        *,
        flight: Flight | None = None,
        expected_generation: int | None = None,
    ) -> Reservation[StandUse]:
        """Move only the stand reservation; support reservations stay untouched."""
        package = self.package_for(flight_id)
        if package.stand is None:
            raise KeyError(flight_id)
        pinned = self._pinned_resources(package)
        if pinned:
            raise StandPackageError(
                f"turnaround plan is pinned to {package.stand.resource_id} "
                f"by stand-bound resources {list(pinned)}; replan required"
            )
        target = new_stand_id or self._best_alternative(package, flight)
        if target is None:
            raise StandPackageError("no alternative stand available")
        generation = (
            expected_generation
            if expected_generation is not None
            else self.stands.ledger.generation
        )
        return self.stands.move(package.stand.token, target, generation)

    def release_package(self, flight_id: str) -> StandPackage:
        """Release the stand and all support reservations of a flight together."""
        package = self.package_for(flight_id)
        if package.stand is not None:
            self.stands.ledger.release(package.stand.token)
        self.turns.release(flight_id)
        return package

    def replan(
        self,
        flight_id: str,
        *,
        flight: Flight,
        tasks: Iterable[TurnTask],
        anchor: datetime,
        departure: datetime,
        stand_id: str | None = None,
        tow_in: bool = False,
        tow_out: bool = False,
    ) -> PackageCommit:
        """Replace the whole package: release stand and support reservations
        together, then commit the new plan and new stand together.

        If the replacement cannot be committed, the original package is restored
        so the flight is never left with a stand but no support resources, or
        support reservations but no stand.
        """
        old = self.package_for(flight_id)
        if old.stand is None and old.turn_plan is None:
            raise KeyError(flight_id)
        ignore = old.stand.token if old.stand is not None else None
        if stand_id is None:
            ranked = self.stands.rank(flight, None, tow_in, tow_out, ignore_token=ignore)
            if not ranked:
                raise StandPackageError("no feasible stand for replacement package")
            stand_id = ranked[0].stand_id
        decision = self.stands.evaluate(flight, stand_id, tow_in, tow_out, ignore_token=ignore)
        if not decision.accepted:
            raise StandPackageError(f"replacement stand infeasible: {decision.reason}")
        self.release_package(flight_id)
        try:
            plan = self.turns.plan(flight_id, tasks, anchor, departure)
            committed = self.turns.commit(plan)
            if not committed.accepted:
                raise StandPackageError(
                    f"replacement turnaround plan rejected: {committed.reason}"
                )
            assignment = self.stands.assign(flight, stand_id, tow_in, tow_out)
            return PackageCommit(flight_id, plan, assignment)
        except Exception as exc:
            failures = self._restore(flight_id, old)
            if failures:
                raise PackageRollbackError(
                    f"replan failed and rollback incomplete: {'; '.join(failures)}",
                    failures,
                ) from exc
            raise StandPackageError("replan failed; original package restored") from exc

    def _restore(self, flight_id: str, old: StandPackage) -> tuple[str, ...]:
        failures: list[str] = []
        self.turns.release(flight_id)
        dangling = self.stands.reservation_for_flight(flight_id)
        if dangling is not None and (
            old.stand is None or dangling.token != old.stand.token
        ):
            try:
                self.stands.ledger.release(dangling.token)
            except Exception as error:
                failures.append(f"stand cleanup: {error}")
        if old.stand is not None:
            try:
                self.stands.ledger.reserve(
                    old.stand.resource_id, old.stand.window, old.stand.payload
                )
            except Exception as error:
                failures.append(f"stand restore: {error}")
        if old.turn_plan is not None:
            try:
                restored = self.turns.commit(old.turn_plan)
                if not restored.accepted:
                    failures.append(f"turnaround restore: {restored.reason}")
            except Exception as error:
                failures.append(f"turnaround restore: {error}")
        return tuple(failures)
