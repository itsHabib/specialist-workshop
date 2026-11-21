from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from decision_workshop import load_model, predict
from decision_workshop.core import train
from decision_workshop.sequence import compare_verified, outcome_labels, replay_alternatives, run_episode
from experiments.decision_limits.sequence_replay import compare


TASK = {"name": "counting", "version": "1", "features": {"count": "number"}, "labels": ["add", "finish"]}


class CountingEpisode:
    def __init__(self):
        self.state = {"count": 0}
        self.done = False

    def observe(self):
        return self.state, ["add", "finish"]

    def apply(self, action):
        if action == "finish":
            self.done = True
        else:
            self.state["count"] += 1

    def is_terminal(self):
        return self.done

    def verify(self, reason):
        if self.done:
            return {"status": "verified" if self.state["count"] == 1 else "incorrect"}
        return {"status": "review" if reason == "review" else "failed"}


def trial(identifier, add, finish, count=0):
    return {"id": identifier, "group": identifier, "state": {"count": count},
            "candidates": ["add", "finish"], "values": {"add": add, "finish": finish}}


class SequenceTests(unittest.TestCase):
    def test_verification_depends_on_result(self):
        trace = run_episode(CountingEpisode(), lambda s, a: "finish" if s["count"] else "add", max_steps=2)
        self.assertEqual([step["state"]["count"] for step in trace["steps"]], [0, 1])
        self.assertEqual(trace["outcome"]["status"], "verified")
        wrong = run_episode(CountingEpisode(), lambda s, a: "finish", max_steps=2)
        self.assertEqual(wrong["outcome"]["status"], "incorrect")

    def test_mutating_policy_cannot_add_an_action_or_change_the_observation(self):
        def mutate(state, actions):
            state["count"] = 5
            actions.append("extra")
            return "extra"
        episode = CountingEpisode()
        trace = run_episode(episode, mutate, max_steps=1)
        self.assertEqual(trace["stop_reason"], "unavailable_action")
        self.assertFalse(trace["steps"][0]["executed"])
        self.assertEqual(trace["steps"][0]["candidates"], ["add", "finish"])
        self.assertEqual(episode.state["count"], 0)

    def test_budgets_and_reviews(self):
        for budget in (0, 1, 4):
            trace = run_episode(CountingEpisode(), lambda s, a: "add", max_steps=budget)
            self.assertEqual((trace["stop_reason"], len(trace["steps"])), ("step_limit", budget))
        decision = {"status": "review", "action": None, "reasons": ["below_min_score"]}
        trace = run_episode(CountingEpisode(), lambda s, a: decision, max_steps=2)
        self.assertEqual(trace["outcome"]["status"], "review")
        self.assertEqual(trace["steps"][0]["decision"], decision)
        episode = CountingEpisode()
        episode.observe = lambda: ({}, [])
        self.assertEqual(run_episode(episode, None, max_steps=1)["stop_reason"], "review")

    def test_faults_remain_visible(self):
        class Broken(CountingEpisode):
            def apply(self, action):
                raise RuntimeError("bad executor")
        with self.assertRaisesRegex(RuntimeError, "bad executor"):
            run_episode(Broken(), lambda s, a: "add", max_steps=2)

    def test_paired_runs_grade_terminal_outcomes_and_restore_each_copy(self):
        branches = replay_alternatives(lambda: nullcontext(CountingEpisode()),
                                      lambda s, a: "finish", lambda t: int(t["outcome"]["status"] == "verified"),
                                      max_steps=2)
        self.assertEqual(branches["values"], {"add": 1, "finish": 0})
        self.assertEqual([s["action"] for s in branches["replays"]["add"]["steps"]], ["add", "finish"])

        opened = []
        closed = []

        @contextmanager
        def different_start():
            episode = CountingEpisode()
            episode.state["count"] = len(opened)
            opened.append(episode)
            try:
                yield episode
            finally:
                closed.append(episode)

        with self.assertRaisesRegex(ValueError, "different observation"):
            replay_alternatives(different_start, None, None, max_steps=1)
        self.assertEqual(opened, closed)

    def test_indistinguishable_worlds_share_one_preference_and_ties_use_priority(self):
        one = trial("a", 10, 0)
        two = trial("b", 0, 10)
        two["candidates"].reverse()
        two["state"]["count"] = 0.0
        records = outcome_labels(TASK, [one, two, trial("c", 0, 1, count=2)])
        self.assertEqual([r["target"] for r in records], ["add", "add", "finish"])
        rev = outcome_labels(TASK, [two, one], action_order=["finish", "add"])
        self.assertEqual([r["target"] for r in rev], ["finish", "finish"])
        restricted = {**one, "id": "d", "candidates": ["finish"], "values": {"finish": 0}}
        self.assertEqual(outcome_labels(TASK, [one, restricted])[1]["target"], "finish")

    def test_rejects_unpaired_and_held_out_evidence(self):
        original = trial("one", 1, 0)
        for modified in ({"split": "test"}, {"values": {"add": 0}}, {"values": {"add": float("nan"), "finish": 0}},
                         {"values": {"add": True, "finish": 0}}, {"candidates": []}):
            with self.subTest(modified=modified), self.assertRaises(ValueError):
                outcome_labels(TASK, [{**original, **modified}])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            outcome_labels(TASK, [original, original])

    def test_availability_can_be_supplied_as_an_explicit_decision_feature(self):
        left = trial("a", 1, 0)
        right = {**trial("b", 0, 1), "candidates": ["finish"], "values": {"finish": 1}}
        trials = [left, right]
        guarded = train(TASK, outcome_labels(TASK, trials), min_leaf=1)
        self.assertEqual(predict(guarded, right["state"], right["candidates"])["status"], "review")
        task = {**TASK, "features": {**TASK["features"], "can_add": "boolean"}}
        for row in trials:
            row["state"] = {**row["state"], "can_add": "add" in row["candidates"]}
        model = train(task, outcome_labels(task, trials), min_leaf=1)
        self.assertEqual([predict(model, r["state"], r["candidates"])["action"] for r in trials], ["add", "finish"])

    def test_train_cli_scores_saved_outcomes_through_predict(self):
        inputs = [trial("one", 1, 0), trial("two", 0, 1, count=1)]
        with TemporaryDirectory() as dirname:
            root = Path(dirname)
            for name, data in (("inputs", inputs), ("task", TASK)):
                (root / name).write_text(json.dumps(data))
            out = root / "tree.json"
            cmd = [sys.executable, "-m", "decision_workshop", "train", "--task", str(root / "task"),
                   "--outcomes", str(root / "inputs"), "--min-leaf", "1", "--model", str(out)]
            subprocess.run(cmd, check=True, capture_output=True)
            model = load_model(out)
            for row in inputs:
                preferred = predict(model, row["state"], row["candidates"])["action"]
                self.assertEqual(preferred, max(row["values"], key=row["values"].get))

    def test_comparison_pairs_jobs_and_resamples_entire_groups_of_unequal_sizes(self):
        first = {("large", str(i)): "verified" for i in range(4)}
        first["small", "one"] = "review"
        second = {key: "failed" for key in first}
        second["small", "one"] = "verified"
        report = compare_verified(first, second, replicates=500)
        self.assertEqual(report["difference"], 3 / 5)
        self.assertEqual(report["candidate_only_verified"], 4)
        self.assertEqual(report["baseline_only_verified"], 1)
        self.assertEqual(report["bootstrap95"], [-1, 1])
        self.assertEqual(compare_verified(first, first)["bootstrap95"], [0, 0])
        with self.assertRaisesRegex(ValueError, "identical"):
            compare_verified(first, {**second, ("unknown", "one"): "verified"})
        with self.assertRaisesRegex(ValueError, "unknown verifier"):
            compare_verified(first, {**second, ("small", "one"): "skipped"})

    def test_second_domain_matches_its_existing_rollout(self):
        self.assertEqual(compare(), {"coded_rule": 128, "expected_cost": 128})

    def test_a_good_first_move_can_fail_under_the_learned_continuation(self):
        # The job succeeds only after exactly one add followed by finish.
        for visible, expected in ((False, "failed"), (True, "verified")):
            def snapshot(count):
                episode = CountingEpisode()
                episode.state["count"] = count
                if not visible:
                    episode.observe = lambda: ({"count": 0}, ["add", "finish"])
                return episode

            trials = []
            for count in (0, 1):
                branches = replay_alternatives(lambda: nullcontext(snapshot(count)), lambda s, a: "finish",
                                              lambda t: int(t["outcome"]["status"] == "verified"),
                                              max_steps=2 - count)
                trials.append({"id": str(count), "group": "training", "state": branches["state"],
                               "candidates": branches["candidates"], "values": branches["values"]})
            model = train(TASK, outcome_labels(TASK, trials), min_leaf=1)
            receipt = run_episode(snapshot(0), lambda s, a: predict(model, s, a), max_steps=2)
            self.assertEqual(receipt["steps"][0]["action"], "add")
            self.assertEqual(receipt["outcome"]["status"], expected)
