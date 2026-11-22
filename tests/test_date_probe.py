import copy
import unittest

from experiments.date_probe import audit, host


def feed(ident, channel, dates, gold, *, notice=False, hint="mdy"):
    payload = ("\n".join(["feed_id=" + ident, *dates]) + "\n").encode()
    return host.Feed(ident, channel, notice, hint, payload, tuple(gold), (dates[0], gold[0]))


class ProbeAuditTests(unittest.TestCase):
    def setUp(self):
        self.ambiguous = feed("x", "alpha", ["04/07/2025", "05/05/2025"],
                              ["2025-07-04", "2025-05-05"])
        self.resolved = feed("y", "beta", ["16/07/2025", "05/06/2025"],
                             ["2025-07-16", "2025-06-05"])
        self.scores = {(n, p): {"empirical": 0.5, "tree": 0.5, "bert": 0.5}
                       for n in (True, False) for p in ("wrong", "right", "mixed", "unseen")}

    def test_valid_parse_can_be_wrong_and_one_witness_recovers_only_disputed_rows(self):
        choice = audit.inspect(self.ambiguous)
        self.assertEqual(host.observe(self.ambiguous), (("2025-04-07", "2025-05-05"), 1, False))
        self.assertTrue(choice.wrong)
        self.assertEqual(choice.gain, 1)
        self.assertIsNone(choice.free_bit)
        self.assertEqual(host.probe(self.ambiguous), (self.ambiguous.expected, True))
        self.assertTrue(audit.inspect(self.resolved).free_bit)
        self.assertEqual(audit.inspect(self.resolved).gain, 0)
        with self.assertRaises(ValueError):
            host.probe(self.resolved)

    def test_a_witness_for_the_wrong_source_convention_fails_grading(self):
        corrupted = host.Feed(self.ambiguous.id, self.ambiguous.channel, self.ambiguous.notice,
                              self.ambiguous.hint, self.ambiguous.payload, self.ambiguous.expected,
                              ("04/07/2025", "2025-04-07"))
        with self.assertRaisesRegex(ValueError, "witness disagrees"):
            audit.inspect(corrupted)

    def test_training_prior_counts_only_visible_peer_outcomes(self):
        # The unresolved alpha feed is wrong, but its own witness is never a feature.
        batch = host.Batch("fit-example", "coupled", (self.ambiguous, self.resolved))
        states = [{"notice": n, "peer_outcome": p, "train_rows": int(not n and p == "unseen"),
                   "train_prefer": int(not n and p == "unseen"), "scores": {"empirical": 2 / 3},
                   "empirical_source": "global" if n else ("state" if p == "unseen" else "notice")}
                  for n, p in self.scores]
        receipt = {"training_rows": 1, "groups": 1, "states": states}
        self.assertEqual(audit.verify_training([batch], receipt), 1)
        seen = next(row for row in states if not row["notice"] and row["peer_outcome"] == "unseen")
        seen["train_prefer"] = 0
        with self.assertRaisesRegex(ValueError, "training counts"):
            audit.verify_training([batch], receipt)

    def test_rows_reject_duplicates_missing_swapped_selections_bad_budget_and_fictitious_gains(self):
        batch = host.Batch("batch", "coupled", (self.ambiguous, self.resolved))
        rows, tables = [], []
        for budget in audit.BUDGETS:
            for policy in audit.POLICIES:
                paid = int(policy != "none")
                fields = {"split": "test", "family": "coupled", "budget": budget, "policy": policy}
                metrics = {"rows": 4, "native_exact": 3, "final_exact": 3 + paid,
                           "remaining_wrong": 1 - paid, "oracle_exact": 4}
                steps = [{"feed_id": "x", "wrong": True, "gain": 1}] if paid else []
                rows.append({**fields, **metrics, "batch_id": "batch", "ambiguous_feeds": 1, "steps": steps})
                tables.append({**fields, **metrics, "batches": 1, "probes": paid, "helpful_probes": paid,
                               "regret": 1 - paid})
        paired = {"batches": 1, "wins": 0, "ties": 1, "losses": 0, "additional_exact_rows": 0,
                  "mean_rows_per_batch": 0.0, "paired_95_interval": [0.0, 0.0]}
        summary = {"tables": tables, "primary_paired": {"coupled": paired}}

        def check(receipts, statistics=summary):
            return audit.check(receipts, statistics, {"test": [batch]}, self.scores)

        self.assertEqual(check(reversed(rows))["verified_receipts"], 30)
        alterations = [rows[:-1], [*rows, rows[-1]], [dict(rows[0], final_exact=4), *rows[1:]]]
        for steps in ([{"feed_id": "y", "wrong": True, "gain": 1}], rows[1]["steps"] * 2,
                      [{"feed_id": "x", "wrong": False, "gain": 1}],
                      [{"feed_id": "x", "wrong": True, "gain": 2}]):
            alterations.append([rows[0], dict(rows[1], steps=steps), *rows[2:]])
        for attempted in alterations:
            with self.subTest(attempt=attempted[:2]), self.assertRaises(ValueError):
                check(attempted)
        damaged = copy.deepcopy(summary)
        damaged["tables"][1]["probes"] = 0
        with self.assertRaisesRegex(ValueError, "summary probes"):
            check(rows, damaged)
        damaged = copy.deepcopy(summary)
        damaged["primary_paired"]["coupled"]["paired_95_interval"][0] = -2.0
        with self.assertRaisesRegex(ValueError, "paired"):
            check(rows, damaged)

    def test_lookahead_pays_for_information_only_if_a_future_choice_can_use_it(self):
        from experiments.date_probe.policy import Candidate, plan

        checked = [audit.CheckedFeed("b6", "B", False, 6, 6, False, None),
                   audit.CheckedFeed("a5", "A", True, 5, 5, False, None),
                   audit.CheckedFeed("a4", "A", True, 4, 4, False, None)]
        # Initially all chances are 1/2. An A witness predicts the other A exactly.
        self.scores[True, "right"]["empirical"] = 0.0
        self.scores[True, "wrong"]["empirical"] = 1.0
        self.assertEqual(audit.plan(checked, [], self.scores, 1).id, "b6")
        self.assertEqual(audit.plan(checked, [], self.scores, 2).id, "a5")
        roster = [Candidate(f.id, f.channel, f.notice, f.disputed) for f in checked]
        chance = lambda candidate, history: audit.probability(candidate, history, self.scores, "empirical")
        self.assertEqual(plan(roster, [], chance, 1).id, "b6")
        self.assertEqual(plan(roster, [], chance, 2).id, "a5")
        self.assertEqual(plan(roster, [], lambda candidate, history: 0.5, 2).id, "b6")
