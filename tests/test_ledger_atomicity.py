import unittest
from datetime import datetime, timezone

from rampflow.ledger import ConflictError, ReservationLedger
from rampflow.timeutil import Window

D = lambda h, m=0: datetime(2026, 1, 1, h, m, tzinfo=timezone.utc)


class ReservationLedgerAtomicityTests(unittest.TestCase):
    def test_move_does_not_release_before_validating_destination(self):
        ledger = ReservationLedger[str]()
        moved = ledger.reserve("OLD", Window(D(10), D(11)), "F1")
        ledger.reserve("NEW", Window(D(10, 30), D(11, 30)), "F2")

        with self.assertRaises(ConflictError):
            ledger.move(
                moved.token,
                "NEW",
                Window(D(11), D(12)),
                "F1-moved",
                ledger.generation,
            )

        self.assertEqual(ledger.reservations("OLD")[0].token, moved.token)
        self.assertEqual(len(ledger.reservations("NEW")), 1)

    def test_bulk_replace_is_all_or_nothing(self):
        ledger = ReservationLedger[str]()
        first = ledger.reserve("A1", Window(D(10), D(11)), "F1:a")
        ledger.reserve("B1", Window(D(12), D(13)), "other")

        with self.assertRaises(ConflictError):
            ledger.bulk_replace(
                (first.token,),
                (
                    ("A2", Window(D(10), D(11)), "F1:a"),
                    ("B1", Window(D(12, 30), D(13, 30)), "F1:b"),
                ),
                ledger.generation,
            )

        self.assertEqual(ledger.reservations("A1")[0].token, first.token)
        self.assertEqual(ledger.reservations("B1")[0].payload, "other")


if __name__ == "__main__":
    unittest.main()
