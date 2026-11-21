import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from experiments.decision_recovery import worker
from experiments.decision_recovery.worker import Episode, Scenario, hand_demo, negative_controls


BATCHES = ((2, -1), (8, -3), (1, 6))


def _interrupted_child(path):
    episode = Episode(Scenario("restart", BATCHES, "checkpoint", 12), path)
    assert episode.timeouts == 1
    os._exit(0)  # Leave the file in the state a replacement worker actually inherits.


def _recovery_child(path, pipe, reuse_progress=False):
    episode = Episode(Scenario("restart", BATCHES, "checkpoint", 12), path,
                      already_interrupted=True, reuse_progress=reuse_progress)
    try:
        if reuse_progress:
            episode.apply("retry")
        else:
            episode.apply("lookup")
            episode.apply("resume")
        pipe.send((os.getpid(), episode.verify("terminal")))
    finally:
        episode.close()
        pipe.close()


def _racing_child(path, barrier, pipe, reuse_progress):
    case = Scenario("race", BATCHES, "before", 12)
    episode = Episode(case, path, already_interrupted=True)
    worked = 0

    def did_batch():
        nonlocal worked
        worked += 1
        if worked == len(BATCHES):
            # All children have done the same real work, none can publish yet.
            barrier.wait(timeout=10)

    def before_first_work(sql):
        if worked == 0 and sql.startswith("INSERT INTO work_log"):
            barrier.wait(timeout=10)
    episode.db.set_trace_callback(before_first_work)
    try:
        pipe.send(worker._run_worker(episode.db, case, "none", 0, 0, did_batch, reuse_progress=reuse_progress))
    finally:
        episode.close()
        pipe.close()


class RecoveryWorkerTests(unittest.TestCase):
    def episode(self, *args, **kwargs):
        episode = Episode(*args, **kwargs)
        self.addCleanup(episode.close)
        return episode

    def test_typed_faults_and_uninterrupted_control(self):
        states = {}
        for fault, action in (("none", None), ("before", "retry"), ("checkpoint", "resume"),
                              ("lost_reply", "return_result")):
            with self.subTest(fault=fault):
                episode = self.episode(Scenario(fault, BATCHES, fault, 12))
                if action:
                    states[fault] = episode.observe()
                    episode.apply("lookup")
                    self.assertIn(action, episode.observe()[1])
                    episode.apply(action)
                result = episode.verify("terminal")
                self.assertEqual(result["status"], "verified")
                self.assertEqual(result["caller"], {"request_id": fault, "result": 13})
                self.assertEqual((result["event_count"], result["batches_processed"]), (1, 3))
        self.assertEqual(states["before"], states["checkpoint"])
        self.assertEqual(states["before"], states["lost_reply"])

    def test_failed_lookup_hides_fault_and_wait_costs_real_simulated_time(self):
        for fault in ("before", "lost_reply"):
            e = self.episode(Scenario("twin", BATCHES, fault, 3, outage_until=3))
            e.apply("lookup")
            state, actions = e.observe()
            self.assertEqual((state["remaining"], state["known"], state["lookup_failures"]),
                             (2, "unknown", 1))
            self.assertEqual(set(actions), {"lookup", "retry", "wait"})
            e.apply("wait")
            e.apply("lookup")
            if fault == "before":
                self.assertEqual(e.verify("terminal")["status"], "failed")
            else:
                e.apply("return_result")
                self.assertEqual(e.verify("terminal")["status"], "verified")

    def test_late_reply_and_voluntary_review_are_distinct(self):
        e = self.episode(Scenario("late", BATCHES, "before", 1))
        self.assertEqual(e.verify("review")["status"], "review")
        e.apply("retry")
        result = e.verify("terminal")
        self.assertEqual((result["status"], result["on_time"], result["event_count"]),
                         ("failed", False, 1))
        self.assertEqual(result["caller"]["result"], 13)

    def test_second_timeout_is_typed_and_recovery_is_bounded(self):
        e = self.episode(Scenario("bounded", BATCHES, "before", 10, retry_fault="lost_reply"))
        e.apply("retry")
        self.assertEqual(e.observe()[0]["known"], "unknown")
        self.assertEqual(e.verify("step_limit")["status"], "failed")
        e.apply("lookup")
        e.apply("return_result")
        outcome = e.verify("terminal")
        self.assertEqual((outcome["status"], outcome["timeouts"], outcome["event_count"]),
                         ("verified", 2, 1))

    def test_rollback_between_result_and_event_preserves_checkpoint(self):
        e = self.episode(Scenario("rollback", BATCHES, "before", 12))
        e.db.execute("""CREATE TRIGGER abort_completion BEFORE INSERT ON completions
                        BEGIN SELECT RAISE(ABORT, 'injected publication failure'); END""")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "injected publication failure"):
            e.apply("retry")
        ledger = e.verify("unavailable_action")["ledger"]
        self.assertIsNone(ledger["requests"][0]["result"])
        self.assertEqual(ledger["requests"][0]["checkpoint_batches"], 3)
        self.assertEqual(ledger["completions"], [])
        e.db.execute("DROP TRIGGER abort_completion")
        e.apply("lookup")
        e.apply("resume")
        self.assertEqual(e.verify("terminal")["status"], "verified")

    def test_non_timeout_propagates_original_cause(self):
        def broken(*args, **kwargs):
            try:
                raise LookupError("fixture cause")
            except LookupError as cause:
                raise RuntimeError("unrelated worker error") from cause

        with patch.object(worker, "_run_worker", broken):
            with self.assertRaisesRegex(RuntimeError, "unrelated worker error") as exc:
                Episode(Scenario("error", BATCHES, "before", 10))
        self.assertIsInstance(exc.exception.__cause__, LookupError)

    def test_key_collision_never_acknowledges_another_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "store.sqlite"
            original = self.episode(Scenario("same", ((1, 2),), "none", 8), path)
            other = self.episode(Scenario("same", ((3,),), "before", 8), path)
            for action in ("lookup", "retry"):
                with self.assertRaisesRegex(ValueError, "different input"):
                    other.apply(action)
            self.assertEqual(original.verify("terminal")["status"], "verified")

    def test_demo_repeated_work_and_negative_controls(self):
        demo = hand_demo()
        first = demo["status_first"]["outcome"]
        retry = demo["same_id_retry"]["outcome"]
        self.assertEqual((first["status"], retry["status"]), ("verified", "verified"))
        self.assertEqual((first["repeated_batches"], retry["repeated_batches"]), (0, 2))
        self.assertEqual(first["ledger"]["completions"], retry["ledger"]["completions"])
        self.assertEqual({value["status"] for value in negative_controls().values()}, {"incorrect"})


    def test_improved_retry_preserves_outcome_and_avoids_only_already_saved_work(self):
        # Same input and same deadline. A late valid reply counts as failed.
        for fault, expected in {
            "before": (("failed", 6, 0), ("failed", 6, 0)),
            "checkpoint": (("failed", 6, 1), ("verified", 4, 0)),
            "lost_reply": (("failed", 6, 3), ("verified", 0, 0)),
        }.items():
            for improved, detail in zip((False, True), expected):
                with self.subTest(fault=fault, reuse_progress=improved):
                    e = self.episode(Scenario("pair", BATCHES, fault, deadline=4),
                                     reuse_progress=improved)
                    e.apply("retry")
                    out = e.verify("terminal")
                    self.assertEqual((out["status"], out["elapsed"], out["repeated_batches"]), detail)
                    self.assertEqual((out["event_count"], out["duplicate_effects"]), (1, 0))

    def test_zero_cost_retry_at_deadline_does_not_reveal_a_hidden_commit(self):
        # First retry takes the last two ticks. One twin committed; its matched twin
        # saved the complete prefix but has not published. Both visibly timed out.
        for fault in ("lost_reply", "checkpoint"):
            case = Scenario("at-deadline", ((4,),), "before", 2, retry_fault=fault)
            improved = self.episode(case, reuse_progress=True)
            improved.apply("retry")
            state, actions = improved.observe()
            self.assertEqual((state["remaining"], state["known"]), (0, "unknown"))
            self.assertIn("retry", actions)
            improved.apply("retry")
            self.assertEqual((improved.elapsed, improved.verify("terminal")["status"]), (2, "verified"))
            original = self.episode(case)
            original.apply("retry")
            self.assertTrue(original.is_terminal())
        empty = self.episode(Scenario("empty", ((4,),), "before", 0), reuse_progress=True)
        empty.apply("retry")
        self.assertEqual((empty.elapsed, empty.verify("terminal")["status"]), (2, "failed"))

    def test_visible_complete_checkpoint_can_publish_at_deadline(self):
        e = self.episode(Scenario("complete-prefix", ((2, 8),), "checkpoint", 1))
        e.apply("lookup")
        state, actions = e.observe()
        self.assertEqual((state["known"], state["remaining"], state["checkpoint_batches"]),
                         ("checkpoint", 0, 1))
        self.assertIn("resume", actions)
        e.apply("resume")
        out = e.verify("terminal")
        self.assertEqual((out["elapsed"], out["status"], out["repeated_batches"]), (1, "verified", 0))

    def test_cached_result_still_loses_reply_and_checks_matching_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "store.sqlite"
            e = self.episode(Scenario("lost", BATCHES, "lost_reply", 2, retry_fault="lost_reply"),
                             path, reuse_progress=True)
            e.apply("retry")
            first = e.verify("step_limit")
            self.assertEqual((first["status"], first["caller"], first["timeouts"], first["elapsed"]),
                             ("failed", None, 2, 0))
            e.apply("retry")
            self.assertEqual((e.verify("terminal")["status"], e.elapsed), ("verified", 0))
            other = self.episode(Scenario("lost", ((13,),), "before", 2), path, reuse_progress=True)
            with self.assertRaisesRegex(ValueError, "different input"):
                other.apply("retry")

    def test_restart_in_new_process_and_reopen_file_independently(self):
        for improved in (False, True):
            with self.subTest(reuse_progress=improved):
                self.check_process_restart(improved)

    def check_process_restart(self, improved):
        ctx = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recovery.sqlite"
            first = ctx.Process(target=_interrupted_child, args=(path,))
            first.start()
            first.join(15)
            self.assertEqual(first.exitcode, 0)
            reader, writer = ctx.Pipe(duplex=False)
            second = ctx.Process(target=_recovery_child, args=(path, writer, improved))
            second.start()
            writer.close()
            second.join(15)
            self.assertEqual(second.exitcode, 0)
            pid, result = reader.recv()
            reader.close()
            self.assertNotEqual(first.pid, pid)
            self.assertEqual((result["status"], result["repeated_batches"]), ("verified", 0))
            db = sqlite3.connect(path)
            try:
                self.assertEqual(db.execute("SELECT result FROM completions").fetchall(), [(13,)])
                raw = db.execute("SELECT input_json FROM requests").fetchone()[0]
                self.assertEqual(sum(sum(batch) for batch in json.loads(raw)), 13)
            finally:
                db.close()

    def test_concurrent_processes_publish_one_result_after_both_have_done_work(self):
        for improved in (False, True):
            with self.subTest(reuse_progress=improved):
                self.check_concurrent_processes(improved)

    def check_concurrent_processes(self, improved):
        ctx = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "race.sqlite"
            owner = self.episode(Scenario("race", BATCHES, "before", 12), path)
            barrier = ctx.Barrier(2)
            children = []
            for _ in range(2):
                reader, writer = ctx.Pipe(duplex=False)
                child = ctx.Process(target=_racing_child, args=(path, barrier, writer, improved))
                child.start()
                writer.close()
                children.append((child, reader))
            for child, reader in children:
                child.join(16)
                self.assertEqual(child.exitcode, 0)
                self.assertEqual(reader.recv(), {"request_id": "race", "result": 13})
                reader.close()
            owner.apply("lookup")
            owner.apply("return_result")
            final = owner.verify("terminal")
            self.assertEqual((final["status"], final["batches_processed"], final["repeated_batches"]),
                             ("verified", 6, 3))
            self.assertEqual((final["event_count"], final["duplicate_effects"]), (1, 0))
