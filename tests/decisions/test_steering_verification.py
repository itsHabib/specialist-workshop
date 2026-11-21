"""Check scheduling decisions against independent enumeration of complete job orders."""

from dataclasses import replace
from itertools import accumulate, permutations
import random
import unittest

from experiments.steering.scheduling import Job, optimal_first_costs, policy_for, schedule_cost, search


def _enumerate(jobs, elapsed=0, realized=False):
    """Calculate loss directly from completion times, with no production solver helpers."""
    results = {}
    for ordering in permutations(jobs):
        durations = (job.actual if realized else job.duration for job in ordering)
        completions = accumulate(durations, initial=elapsed)
        next(completions)  # initial time is before the first job finishes
        loss = sum(job.weight * max(0, finish - job.due) for job, finish in zip(ordering, completions))
        results[tuple(j.id for j in ordering)] = loss
    return results


def _first_costs(table):
    costs = {}
    for order, loss in table.items():
        if order[0] not in costs or loss < costs[order[0]]:
            costs[order[0]] = loss
    return costs


class SteeringVerificationTests(unittest.TestCase):
    def test_one_deadline_flips_the_unique_best_first_job(self):
        a = Job(id="A", duration=2, due=2, weight=3, actual=2)
        b = Job(id="B", duration=2, due=3, weight=1, actual=2)
        for jobs, expected in [((a, b), {("A", "B"): 1, ("B", "A"): 6}),
                               ((replace(a, due=4), b), {("A", "B"): 1, ("B", "A"): 0})]:
            with self.subTest(deadline=jobs[0].due):
                self.assertEqual(_enumerate(jobs), expected)
                self.assertEqual({o: schedule_cost(jobs, o) for o in expected}, expected)
                self.assertEqual(optimal_first_costs(jobs), _first_costs(expected))
                greedy = search(jobs, policy_for("edd"), budget=1)
                self.assertEqual(tuple(greedy["selected_order"]), min(expected, key=expected.get))

    def test_every_first_score_matches_full_enumeration_and_ignores_identifiers(self):
        rng = random.Random(11307)
        for i in range(12):
            jobs = [Job(id=f"key-{i}-{k}", duration=rng.randint(1, 6), due=rng.randint(1, 20),
                        weight=rng.randint(1, 4), actual=rng.randint(1, 7)) for k in range(5)]
            names = {job.id: f"new-{i}-{9-k}" for k, job in enumerate(jobs)}
            renamed = [replace(job, id=names[job.id]) for job in jobs]
            rng.shuffle(renamed)
            for realized in (False, True):
                for elapsed in (0, 7):
                    expected = _enumerate(jobs, elapsed, realized)
                    costs = optimal_first_costs(jobs, elapsed, realized=realized)
                    self.assertEqual(costs, _first_costs(expected))
                    self.assertEqual({names[k]: v for k, v in costs.items()},
                                     optimal_first_costs(renamed, elapsed, realized=realized))
                    if elapsed == 0:
                        for order, loss in expected.items():
                            self.assertEqual(schedule_cost(jobs, order, realized=realized), loss)

    def test_blocked_first_job_is_required_later_and_empty_root_is_unreachable(self):
        jobs = (Job(id="A", duration=2, due=2, weight=3, actual=2),
                Job(id="B", duration=2, due=3, weight=1, actual=2))
        result = search(jobs, policy_for("edd"), budget=1, blocked_first=("A",))
        self.assertEqual(len(result["attempts"]), 1)
        attempt = result["attempts"][0]
        self.assertEqual(list(attempt["order"]), ["B", "A"])
        self.assertEqual(attempt["estimated_cost"], 6)
        self.assertEqual(attempt["realized_cost"], 6)
        self.assertEqual(min(_enumerate(jobs).values()), 1)
        with self.assertRaises(ValueError):
            search(jobs, policy_for("edd"), budget=5, blocked_first=("A", "B"))

    def test_early_stop_observes_only_the_completed_prefix_and_preserves_its_best(self):
        jobs = [Job(id="a", duration=4, due=2, weight=2, actual=1),
                Job(id="b", duration=6, due=7, weight=3, actual=9),
                Job(id="c", duration=3, due=10, weight=4, actual=1),
                Job(id="d", duration=1, due=9, weight=1, actual=8),
                Job(id="e", duration=2, due=12, weight=5, actual=3)]
        predicted, realized = _enumerate(jobs), _enumerate(jobs, realized=True)
        policy = policy_for("edd")
        attempts = search(jobs, policy, budget=20)["attempts"]
        for attempt in attempts:
            order = tuple(attempt["order"])
            self.assertEqual(attempt["estimated_cost"], predicted[order])
            self.assertEqual(attempt["realized_cost"], realized[order])
        for limit in (1, 3, 5):
            observed = []

            def stop(prefix):
                observed.append(prefix)
                return len(prefix) == limit

            result = search(jobs, policy, budget=20, stop_after=stop)
            prefix = attempts[:limit]
            expected_costs = [a["estimated_cost"] for a in prefix]
            self.assertEqual(observed, [tuple(expected_costs[:i]) for i in range(1, limit+1)])
            self.assertEqual([a["order"] for a in result["attempts"]], [a["order"] for a in prefix])
            index = result["selected_attempt"] - 1
            self.assertEqual(expected_costs[index], min(expected_costs))
            self.assertEqual(result["selected_order"], prefix[index]["order"])

        # No true duration may change a choice or an observation available to a policy.
        altered = [replace(job, actual=job.duration) for job in jobs]
        original = search(jobs, policy, budget=20)
        alternative = search(altered, policy, budget=20)
        for key in ("order", "rank_code", "estimated_cost"):
            self.assertEqual([a[key] for a in original["attempts"]], [a[key] for a in alternative["attempts"]])
        self.assertEqual(original["selected_attempt"], alternative["selected_attempt"])
        self.assertNotEqual([a["realized_cost"] for a in original["attempts"]],
                            [a["realized_cost"] for a in alternative["attempts"]])

    def test_exhausted_search_visits_every_feasible_permutation_once(self):
        jobs = (Job("a", 3, 4, 2, 9), Job("b", 3, 4, 2, 1),
                Job("c", 2, 7, 1, 6), Job("d", 4, 9, 1, 3))
        for blocked in ((), ("a", "c")):
            table = {order: value for order, value in _enumerate(jobs).items() if order[0] not in blocked}
            result = search(jobs, policy_for("edd"), budget=100, blocked_first=blocked)
            orders = [tuple(a["order"]) for a in result["attempts"]]
            self.assertCountEqual(orders, list(table))
            self.assertEqual(table[tuple(result["selected_order"])], min(table.values()))
            for order, a in zip(orders, result["attempts"]):
                self.assertEqual(table[order], a["estimated_cost"])

    def test_invalid_complete_order_and_incomplete_ranking_are_rejected(self):
        jobs = (Job("one", 1, 3, 1, 1), Job("two", 1, 3, 1, 1))
        for bad in (("one",), ("one", "one"), ("two", "other"), ("one", "two", "one")):
            with self.assertRaises(ValueError):
                schedule_cost(jobs, bad)
        with self.assertRaises(ValueError):
            search(jobs, lambda candidates: [(candidates[0][0], 1.0)], budget=1)
        self.assertEqual(optimal_first_costs(jobs), {"one": 0, "two": 0})
