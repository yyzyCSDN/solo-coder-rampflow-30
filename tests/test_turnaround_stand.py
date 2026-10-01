import unittest
from datetime import datetime, timezone

from rampflow.models import TaskSpec
from rampflow.stand_orchestration import (
    StandAssignment,
    StandDemand,
    StandOrchestrator,
    StandProfile,
)
from rampflow.timeutil import Window
from rampflow.turnaround import ResourceRequirement, TurnaroundPlanner, TurnTask
from rampflow.turnaround_stand import TurnaroundStandCoordinator

UTC = timezone.utc
D = lambda h, m=0: datetime(2026, 1, 1, h, m, tzinfo=UTC)
W = lambda start, end: Window(D(*start), D(*end))


class FailingStandOrchestrator(StandOrchestrator):
    def commit(self, assignment):
        raise RuntimeError("boom")


class TurnaroundStandCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.stands = StandOrchestrator([
            StandProfile("S1", "T", 60, True, 1, incompatible_simultaneous=frozenset({"S2"})),
            StandProfile("S2", "T", 60, True, 1),
            StandProfile("S3", "T", 60, True, 1),
        ])
        self.turn = TurnaroundPlanner({"crew": ["C1", "C2"], "gpu": ["G1", "G2"]})
        self.coordinator = TurnaroundStandCoordinator(self.stands, self.turn)
        self.tasks = (
            TurnTask(TaskSpec("clean", 20), requirements=(ResourceRequirement("crew"),)),
            TurnTask(TaskSpec("power", 20), requirements=(ResourceRequirement("gpu"),)),
        )
        self.demand = StandDemand("F1", W((10,), (11,)), 35, False)

    def commit_package(self):
        package = self.coordinator.plan(self.demand, self.tasks, D(10), D(11))
        decision = self.coordinator.commit(package)
        self.assertTrue(decision.accepted)
        return package

    def test_package_commits_and_releases_together(self):
        self.commit_package()
        self.assertEqual(self.stands.occupancy("S1")[0].flight_id, "F1")
        self.assertEqual(len(self.turn.ledger.snapshot()[1]), 2)

        self.coordinator.release("F1")

        self.assertEqual(self.stands.occupancy("S1"), ())
        self.assertEqual(self.turn.ledger.snapshot()[1], ())
        self.assertNotIn("F1", self.coordinator.packages)

    def test_stale_resource_generation_requires_full_replan_and_commits_nothing(self):
        package = self.coordinator.plan(self.demand, self.tasks, D(10), D(11))
        self.turn.ledger.reserve("gpu", W((9,), (9, 30)), "other")

        decision = self.coordinator.commit(package)

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, "resource-generation")
        self.assertEqual(len(self.turn.ledger.reservations("gpu")), 1)
        self.assertEqual(self.stands.occupancy("S1"), ())

    def test_adjacency_conflict_allows_stand_only_change(self):
        self.commit_package()
        other_demand = StandDemand("F2", W((10, 10), (11, 10)), 35, False)
        other = self.stands.plan(other_demand, ignore="F2")
        self.stands.assignments["F2"] = StandAssignment(
            "F2", "S2", other_demand.window, other.protected_window,
            self.stands.generation, other.score,
        )
        self.stands.by_stand["S2"].append(self.stands.assignments["F2"])
        self.stands.generation += 1

        assessment = self.coordinator.reassess("F1")
        self.assertTrue(assessment.accepted)
        self.assertEqual(assessment.code, "stand-change-required")
        self.assertEqual(assessment.details["required"], "stand-only")

        moved = self.coordinator.change_stand("F1")
        self.assertEqual(moved.stand.stand_id, "S3")
        self.assertEqual(len(self.turn.ledger.snapshot()[1]), 2)
        self.assertEqual(self.stands.occupancy("S1"), ())
        self.assertEqual(self.stands.occupancy("S3")[0].flight_id, "F1")

    def test_failed_stand_commit_rolls_back_resource_reservations(self):
        package = self.coordinator.plan(self.demand, self.tasks, D(10), D(11))
        failing = FailingStandOrchestrator(list(self.stands.profiles.values()))
        failing.generation = self.stands.generation
        self.coordinator.stands = failing

        with self.assertRaises(RuntimeError):
            self.coordinator.commit(package)

        self.assertEqual(self.turn.ledger.snapshot()[1], ())

    def test_full_replan_replaces_old_reservations_without_orphans(self):
        self.commit_package()
        self.turn.ledger.reserve("R9", W((9,), (9, 30)), "other")

        assessment = self.coordinator.reassess("F1")
        self.assertEqual(assessment.code, "resource-generation")
        updated = self.coordinator.replan("F1")

        resources = {(r.resource_id, r.payload) for r in self.turn.ledger.snapshot()[1]}
        self.assertIn(("R9", "other"), resources)
        self.assertEqual(
            {resource for resource, payload in resources if isinstance(payload, str) and payload.startswith("F1:")},
            {"C1", "G1"},
        )
        self.assertIsNotNone(updated.stand)


if __name__ == "__main__":
    unittest.main()
