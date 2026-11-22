from itertools import permutations
import unittest

from experiments.steering.scheduling import (
    Job, aggregate, forbidden_optima, make_queue, measure, optimal_first_costs, policy_for, search,
)


class SchedulingTests(unittest.TestCase):
    def test_an_independent_hand_checked_tie_at_due_date(self):
        a = Job("A", 2, 2, 3, 2)
        b = Job("B", 2, 3, 1, 2)
        self.assertEqual(optimal_first_costs([a, b]), {"A": 1, "B": 6})
        self.assertEqual(optimal_first_costs([Job("A", 2, 4, 3, 2), b]), {"A": 1, "B": 0})

    def test_complete_search_agrees_with_independent_enumeration(self):
        jobs = make_queue("validator", 9, n=4)
        costs = {}
        for perm in permutations(jobs):
            elapsed = total = 0
            for j in perm:
                elapsed += j.duration
                total += j.weight * max(0, elapsed - j.due)
            costs[tuple(j.id for j in perm)] = total
        first = {j.id: min(v for p, v in costs.items() if p[0] == j.id) for j in jobs}
        self.assertEqual(optimal_first_costs(jobs), first)
        res = search(jobs, policy_for("edd"), budget=100)
        self.assertEqual(len(res["attempts"]), len(costs))
        self.assertEqual({tuple(r["order"]): r["estimated_cost"] for r in res["attempts"]}, costs)
        self.assertEqual(costs[tuple(res["selected_order"])], min(costs.values()))

    def test_ordering_is_invariant_under_renaming_input_shuffle_and_hidden_duration(self):
        jobs = make_queue("iid", 4)
        transformed = [Job(str(i+17), j.duration, j.due, j.weight, j.actual+9) for i,j in enumerate(jobs)]
        unrename = {new.id: old.id for old,new in zip(jobs,transformed)}
        for name in ("spt","edd","wspt","atc"):
            gold = search(jobs,policy_for(name))
            result = search(list(reversed(transformed)),policy_for(name))
            self.assertEqual([r["order"] for r in gold["attempts"]],
                             [[unrename[x] for x in r["order"]] for r in result["attempts"]])
            self.assertEqual(gold["selected_attempt"],result["selected_attempt"])

    def test_omission_is_initial_only_and_exhaustion_is_scored(self):
        jobs = make_queue("iid", 0, n=4)
        blocked = tuple(j.id for j in jobs[:3])
        res = search(jobs,policy_for("spt"),blocked_first=blocked)
        self.assertEqual(len(res["attempts"]),6)
        for x in res["attempts"]:
            self.assertEqual(x["order"][0], jobs[-1].id)
            self.assertEqual(set(x["order"]), {j.id for j in jobs})
        row = {"realized_optimum":0,"attainable_actual":0,"policies":{"spt":res}}
        self.assertEqual(aggregate([row])["spt"]["20"]["exhausted_before_budget"],1)
        self.assertEqual(aggregate([row])["spt"]["20"]["mean_cost"],min(x["realized_cost"] for x in res["attempts"]))
        with self.assertRaises(ValueError):
            search(jobs,policy_for("spt"),blocked_first=tuple(j.id for j in jobs))

    def test_stop_callback_only_gets_costs_of_completed_attempts(self):
        observed = []
        def stop(costs):
            observed.append(costs)
            return len(costs) >= 3
        base = search(make_queue("noisy",6), policy_for("edd"), stop_after=stop)
        self.assertEqual([len(o) for o in observed],[1,2,3])
        self.assertEqual(observed[-1],tuple(x["estimated_cost"] for x in base["attempts"]))
        self.assertEqual(list(base["checkpoints"]),["1"])


class NoisySelectionTests(unittest.TestCase):
    def test_hidden_durations_and_names_do_not_change_which_schedule_is_selected(self):
        from dataclasses import replace
        from experiments.steering.noisy_scoring import select_plans
        jobs = make_queue("invariance", 8)
        orders = [row["order"] for row in search(jobs, policy_for("atc"))["attempts"]]
        renamed = {j.id: f"arbitrary-name-{k+10}" for k, j in enumerate(jobs)}
        new_jobs = [replace(j, id=renamed[j.id], actual=1000) for j in reversed(jobs)]
        new_orders = [[renamed[x] for x in order] for order in orders]
        a = select_plans(jobs, orders, 3, robust=True)
        b = select_plans(new_jobs, new_orders, 3, robust=True)
        self.assertEqual(a["candidate_costs"], b["candidate_costs"])
        self.assertEqual(a["by_budget"]["20"]["selected_attempt"], b["by_budget"]["20"]["selected_attempt"])

    def test_same_visited_orders_in_all_worlds_and_selectors(self):
        from experiments.steering.noisy_scoring import one_queue
        row = one_queue(17)
        nominal = row["selectors"]["nominal"]
        self.assertEqual(nominal["candidate_costs"], row["worlds"]["no_noise"]["candidate_realized_costs"])
        self.assertEqual(len(row["selectors"]["robust"]["profiles"]), 32)
        for world in row["worlds"].values():
            self.assertEqual(len(world["candidate_realized_costs"]), 20)
            for budget in ("5", "20"):
                for kind, selector in row["selectors"].items():
                    idx = selector["by_budget"][budget]["selected_attempt"] - 1
                    self.assertEqual(world["selected_costs"][kind][budget], world["candidate_realized_costs"][idx])
