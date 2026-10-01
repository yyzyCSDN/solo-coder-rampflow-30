import unittest
from datetime import datetime, timezone

from rampflow.ledger import ConflictError
from rampflow.models import Aircraft, Flight, TaskSpec
from rampflow.stand_package import (
    StandPackageCoordinator,
    StandPackageError,
)
from rampflow.stands import AdjacencyRule, Stand, StandPlanner
from rampflow.timeutil import Window
from rampflow.turnaround import ResourceRequirement, TurnTask, TurnaroundPlanner

UTC = timezone.utc


def D(h, m=0):
    return datetime(2026, 1, 1, h, m, tzinfo=UTC)


def make_flight(fid="F1", reg="REG1", wingspan=36.0, start=None, end=None):
    start = start or D(10)
    end = end or D(11)
    jet = Aircraft(reg, "A320", wingspan, 38.0, 70000.0, 180)
    return Flight(fid, jet, "AAA", "BBB", start, end, 120, 60)


def three_stands():
    return StandPlanner(
        [
            Stand("S1", 40.0, 45.0, "T1", adjacency_group="G"),
            Stand("S2", 40.0, 45.0, "T1", adjacency_group="G"),
            Stand("S3", 40.0, 45.0, "T1"),
        ],
        [AdjacencyRule("G", 70.0)],
    )


def fuel_task():
    return TurnTask(
        TaskSpec("fuel", 20, earliest_offset_min=0),
        (ResourceRequirement("fuel"),),
    )


def clean_task():
    return TurnTask(
        TaskSpec("clean", 15, earliest_offset_min=5),
        (ResourceRequirement("clean"),),
    )


def bridge_task():
    return TurnTask(
        TaskSpec("bridge", 10, earliest_offset_min=0),
        (ResourceRequirement("bridge"),),
    )


def mobile_planner():
    return TurnaroundPlanner({"fuel": ["FT1"], "clean": ["CL1"]})


class TurnaroundCommitTests(unittest.TestCase):
    def test_stale_plan_is_revalidated_and_actually_reserved(self):
        planner = TurnaroundPlanner({"fuel": ["FT1"]})
        p1 = planner.plan("F1", [fuel_task()], D(10), D(12))
        self.assertTrue(planner.commit(p1).accepted)
        p2 = planner.plan("F2", [fuel_task()], D(12), D(14))
        p3 = planner.plan("F3", [fuel_task()], D(14), D(16))
        planner.commit(p3)  # bumps the generation after p2 was planned

        decision = planner.commit(p2)

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.code, "reconciled")
        windows = [
            r.window
            for r in planner.ledger.reservations("FT1")
            if r.payload == "F2:fuel"
        ]
        self.assertEqual(windows, [Window(D(12), D(12, 20))])

    def test_stale_plan_with_conflicting_windows_is_rejected_cleanly(self):
        planner = TurnaroundPlanner({"fuel": ["FT1"]})
        p1 = planner.plan("F1", [fuel_task()], D(10), D(12))
        p2 = planner.plan("F2", [fuel_task()], D(10), D(12))
        planner.commit(p1)

        decision = planner.commit(p2)

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, "resource-conflict")
        self.assertNotIn("F2", planner.plans)
        self.assertEqual(len(planner.ledger.reservations("FT1")), 1)

    def test_release_removes_every_reservation_of_the_flight(self):
        planner = mobile_planner()
        plan = planner.plan("F1", [fuel_task(), clean_task()], D(10), D(11))
        planner.commit(plan)
        self.assertEqual(len(planner.reservations_for("F1")), 2)

        released = planner.release("F1")

        self.assertEqual(len(released), 2)
        self.assertEqual(planner.ledger.reservations("FT1"), ())
        self.assertEqual(planner.ledger.reservations("CL1"), ())
        self.assertEqual(planner.plans, {})
        self.assertEqual(planner.release("F1"), ())

    def test_conflicting_duplicate_commit_is_rejected(self):
        planner = TurnaroundPlanner({"fuel": ["FT1"]})
        p1 = planner.plan("F1", [fuel_task()], D(10), D(12))
        planner.commit(p1)
        again = planner.plan("F1", [fuel_task()], D(10), D(12))

        decision = planner.commit(again)

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, "duplicate-plan")
        self.assertEqual(len(planner.ledger.reservations("FT1")), 1)

    def test_identical_recommit_is_a_noop(self):
        planner = TurnaroundPlanner({"fuel": ["FT1"]})
        p1 = planner.plan("F1", [fuel_task()], D(10), D(12))
        planner.commit(p1)

        decision = planner.commit(p1)

        self.assertTrue(decision.accepted)
        self.assertEqual(len(planner.ledger.reservations("FT1")), 1)


class StandMoveTests(unittest.TestCase):
    def test_move_to_adjacent_stand_ignores_own_reservation(self):
        planner = three_stands()
        assignment = planner.assign(make_flight(), "S1")
        token = assignment.reservation.token

        moved = planner.move(token, "S2", planner.ledger.generation)

        self.assertEqual(moved.resource_id, "S2")
        self.assertEqual(planner.ledger.reservations("S1"), ())
        self.assertEqual(planner.reservation_for_flight("F1").token, moved.token)

    def test_failed_move_keeps_original_reservation(self):
        planner = three_stands()
        planner.assign(make_flight("F2", "REG2", 28.0), "S2")
        assignment = planner.assign(make_flight(), "S1")
        token = assignment.reservation.token

        with self.assertRaises(ConflictError):
            planner.move(token, "S2", planner.ledger.generation)

        current = planner.reservation_for_flight("F1")
        self.assertIsNotNone(current)
        self.assertEqual(current.resource_id, "S1")
        self.assertEqual(current.window, Window(D(10), D(11)))

    def test_adjacent_occupied_stand_is_mutually_exclusive(self):
        planner = three_stands()
        planner.assign(make_flight(), "S1")

        with self.assertRaises(ConflictError):
            planner.assign(make_flight("F2", "REG2"), "S2")


class CoordinatorScenarioTests(unittest.TestCase):
    def setUp(self):
        self.stands = three_stands()
        self.turns = mobile_planner()

    def commit_package(self, tasks=None, fid="F1"):
        tasks = tasks if tasks is not None else [fuel_task(), clean_task()]
        self.stands.assign(make_flight(fid), "S1")
        plan = self.turns.plan(fid, tasks, D(10), D(11))
        self.assertTrue(self.turns.commit(plan).accepted)
        return plan

    def test_swap_path_moves_only_the_stand(self):
        self.commit_package()
        with self.assertRaises(ConflictError):
            self.stands.assign(make_flight("F2", "REG2"), "S2")
        coordinator = StandPackageCoordinator(self.stands, self.turns, {"S1": {"BR1"}})

        assessment = coordinator.assess("F1")

        self.assertEqual(assessment.action, "swap")
        self.assertEqual(assessment.current_stand, "S1")
        self.assertIsNotNone(assessment.candidate_stand)
        tokens_before = {r.token for r in self.turns.reservations_for("F1")}

        coordinator.swap("F1", "S3")

        self.assertEqual(self.stands.reservation_for_flight("F1").resource_id, "S3")
        self.assertEqual({r.token for r in self.turns.reservations_for("F1")}, tokens_before)
        self.assertEqual(
            [r.window for r in self.turns.ledger.reservations("FT1")],
            [Window(D(10), D(10, 20))],
        )
        self.stands.assign(make_flight("F2", "REG2"), "S2")

    def test_stand_bound_resources_force_replan(self):
        planner = TurnaroundPlanner({"fuel": ["FT1"], "bridge": ["BR1"]})
        self.turns = planner
        self.stands.assign(make_flight(), "S1")
        plan = planner.plan("F1", [fuel_task(), bridge_task()], D(10), D(11))
        planner.commit(plan)
        coordinator = StandPackageCoordinator(self.stands, planner, {"S1": {"BR1"}})

        assessment = coordinator.assess("F1")

        self.assertEqual(assessment.action, "replan")
        self.assertIn("stand-bound-resources", assessment.reasons)
        self.assertEqual(assessment.pinned_resources, ("BR1",))
        with self.assertRaises(StandPackageError):
            coordinator.swap("F1", "S3")

    def test_no_alternative_stand_forces_replan(self):
        single = StandPlanner([Stand("S1", 40.0, 45.0, "T1")])
        single.assign(make_flight(), "S1")
        plan = self.turns.plan("F1", [fuel_task()], D(10), D(11))
        self.turns.commit(plan)
        coordinator = StandPackageCoordinator(single, self.turns)

        assessment = coordinator.assess("F1")

        self.assertEqual(assessment.action, "replan")
        self.assertIn("no-alternative-stand", assessment.reasons)

    def test_replan_replaces_the_whole_package(self):
        old_plan = self.commit_package()
        coordinator = StandPackageCoordinator(self.stands, self.turns)
        new_flight = make_flight(start=D(12), end=D(13))

        result = coordinator.replan(
            "F1",
            flight=new_flight,
            tasks=[fuel_task(), clean_task()],
            anchor=D(12),
            departure=D(13),
            stand_id="S3",
        )

        self.assertEqual(result.stand.stand_id, "S3")
        self.assertEqual(self.stands.ledger.reservations("S1"), ())
        self.assertEqual(self.stands.reservation_for_flight("F1").window, Window(D(12), D(13)))
        self.assertEqual(
            [r.window for r in self.turns.ledger.reservations("FT1")],
            [Window(D(12), D(12, 20))],
        )
        self.assertEqual(
            [r.window for r in self.turns.ledger.reservations("CL1")],
            [Window(D(12, 5), D(12, 20))],
        )
        self.assertNotEqual(result.plan, old_plan)
        _, rows = self.stands.ledger.snapshot()
        self.assertEqual(len(rows), 1)

    def test_failed_replan_restores_original_package(self):
        old_plan = self.commit_package()
        coordinator = StandPackageCoordinator(self.stands, self.turns)
        bad_task = TurnTask(
            TaskSpec("bad", 30, latest_finish_offset_min=10),
            (ResourceRequirement("fuel"),),
        )

        with self.assertRaises(StandPackageError):
            coordinator.replan(
                "F1",
                flight=make_flight(start=D(12), end=D(13)),
                tasks=[bad_task],
                anchor=D(12),
                departure=D(13),
                stand_id="S3",
            )

        restored = self.stands.reservation_for_flight("F1")
        self.assertIsNotNone(restored)
        self.assertEqual(restored.resource_id, "S1")
        self.assertEqual(restored.window, Window(D(10), D(11)))
        self.assertEqual(self.turns.plans["F1"], old_plan)
        ft1 = self.turns.ledger.reservations("FT1")
        self.assertEqual([r.window for r in ft1], [Window(D(10), D(10, 20))])
        self.assertEqual(
            [r.window for r in self.turns.ledger.reservations("CL1")],
            [Window(D(10, 5), D(10, 20))],
        )
        _, rows = self.stands.ledger.snapshot()
        self.assertEqual(len(rows), 1)

    def test_infeasible_replacement_stand_aborts_before_release(self):
        self.commit_package()
        self.stands.assign(make_flight("F2", "REG2", 28.0), "S2")
        coordinator = StandPackageCoordinator(self.stands, self.turns)
        old_token = self.stands.reservation_for_flight("F1").token
        old_turn_tokens = {r.token for r in self.turns.reservations_for("F1")}

        with self.assertRaises(StandPackageError):
            coordinator.replan(
                "F1",
                flight=make_flight(),
                tasks=[fuel_task()],
                anchor=D(10),
                departure=D(11),
                stand_id="S2",
            )

        self.assertEqual(self.stands.reservation_for_flight("F1").token, old_token)
        self.assertEqual(
            {r.token for r in self.turns.reservations_for("F1")}, old_turn_tokens
        )

    def test_release_package_releases_both_ledgers(self):
        self.commit_package()
        coordinator = StandPackageCoordinator(self.stands, self.turns)

        package = coordinator.release_package("F1")

        self.assertIsNotNone(package.stand)
        self.assertEqual(len(package.turn_reservations), 2)
        self.assertIsNone(self.stands.reservation_for_flight("F1"))
        self.assertEqual(self.turns.plans, {})
        _, rows = self.stands.ledger.snapshot()
        self.assertEqual(rows, ())
        self.assertEqual(self.turns.ledger.reservations("FT1"), ())
        self.assertEqual(self.turns.ledger.reservations("CL1"), ())

    def test_unknown_flight_is_rejected(self):
        coordinator = StandPackageCoordinator(self.stands, self.turns)
        with self.assertRaises(KeyError):
            coordinator.assess("NOPE")
        with self.assertRaises(KeyError):
            coordinator.swap("NOPE", "S3")
        with self.assertRaises(KeyError):
            coordinator.replan(
                "NOPE",
                flight=make_flight("NOPE"),
                tasks=[fuel_task()],
                anchor=D(10),
                departure=D(11),
            )


if __name__ == "__main__":
    unittest.main()
