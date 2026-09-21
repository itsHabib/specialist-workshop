import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

import rooms_backend


def config():
    return {
        "host": "rooms-host",
        "limactl": "limactl",
        "binary": "/opt/rooms",
        "snapshot": "/state/snapshot",
        "snapshot_sha256": {"snapshot.json": "a", "snapshot.mem": "b"},
        "image_path": "/images/python.ext4",
        "image_sha256": "image-sha",
        "toolstore": "/stores/python",
        "toolstore_sha256": "store-sha",
        "remote_root": "/root/bridge-runs",
        "state_root": "/root/.local/state/rooms",
        "identity": "python-neutral-v1",
    }


class FakeProcess:
    def __init__(self, stdout, stderr=b"", polls_before_exit=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = None
        self.polls_before_exit = polls_before_exit
        self.terminated = False

    def poll(self):
        if self.polls_before_exit > 0:
            self.polls_before_exit -= 1
            return None
        self.returncode = 0
        return 0

    def communicate(self, timeout=None):
        self.returncode = 0
        return self.stdout, self.stderr

    def terminate(self):
        self.terminated = True
        self.returncode = -15


class FakeLima:
    instance = None

    def __init__(self, _config):
        type(self).instance = self
        self.manifest = None
        self.cancelled = False
        self.cleanup_calls = 0
        self.public = None
        self.outputs = [{"answer": 2}, {"candidate_exception": "ValueError"}]
        self.clone = {
            "case_id": "candidate-batch",
            "command_sha256": "sha256:command",
            "room_id": "room-1",
            "snapshot_id": "snapshot-1",
            "slot": 1,
            "namespace": "rooms-c1",
            "host_veth": "veth-h1",
            "status": "exited",
            "exit_code": 0,
        }

    def prepare(self, _remote_dir, manifest):
        self.manifest = manifest

    def start(self, _rooms_argv, _remote_dir):
        case_id = json.loads(self.manifest)["cases"][0]["id"]
        self.clone["case_id"] = case_id
        matrix = {
            "schema": "rooms.matrix.result.v1",
            "matrix_sha256": "sha256:matrix",
            "status": "completed",
            "clones": [self.clone],
        }
        return FakeProcess((json.dumps(matrix) + "\n").encode())

    def cancel(self, _remote_dir):
        self.cancelled = True

    def read(self, path, limit=rooms_backend._OUTPUT_LIMIT):
        if path.endswith("outputs.json"):
            return json.dumps(self.outputs).encode()
        if path.endswith("witness.json"):
            return json.dumps(
                {
                    "capture_complete": True,
                    "egress_policy": "none",
                    "permitted": [],
                    "blocked": [],
                }
            ).encode()
        raise AssertionError(path)

    def teardown(self, _clone):
        return {"complete": True, "checks": {"process_absent": True}}

    def cleanup_owned(self, _case_id):
        self.cleanup_calls += 1
        return {"complete": True, "checks": {"owned_room_absent": True}}

    def write_public(self, _path, record):
        self.public = record


class RoomsBackendTests(unittest.TestCase):
    def test_mocked_rooms_batch_preserves_contract_and_evidence(self):
        source = "def evaluate(request):\n    raise ValueError('synthetic')\n"
        requests = [{"value": 1}, {"value": 2}]
        with mock.patch.object(rooms_backend, "_Lima", FakeLima):
            result = rooms_backend.evaluate(source, requests, image=config())
        self.assertIsNone(result["error"])
        self.assertEqual(result["outputs"], FakeLima.instance.outputs)
        self.assertEqual(result["witness"]["egress_policy"], "none")
        self.assertTrue(result["evidence_summary"]["teardown"]["complete"])
        manifest = json.loads(FakeLima.instance.manifest)
        command = manifest["cases"][0]["command"]
        self.assertLess(len(command.encode()), 16 * 1024)
        self.assertNotIn("synthetic", command)
        self.assertNotIn("expected", command)
        self.assertIn("ulimit -t 5", command)
        self.assertIn("ulimit -v 262144", command)
        self.assertIn("> /workspace/out/candidate.stdout", command)
        self.assertEqual(FakeLima.instance.public["snapshot_sha256"], config()["snapshot_sha256"])
        self.assertRegex(FakeLima.instance.public["host_command_sha256"], r"^[0-9a-f]{64}$")

    def test_source_load_exception_becomes_one_answer_per_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "candidate.py"
            requests = root / "requests.json"
            runner = root / "runner.py"
            outputs = root / "outputs.json"
            source.write_text("def (", encoding="utf-8")
            requests.write_text("[{},{}]", encoding="utf-8")
            runner.write_text(rooms_backend._RUNNER, encoding="utf-8")
            subprocess.run(
                [sys.executable, runner, source, requests, outputs],
                check=True,
                capture_output=True,
            )
            self.assertEqual(
                json.loads(outputs.read_text(encoding="utf-8")),
                [
                    {"candidate_exception": "SyntaxError"},
                    {"candidate_exception": "SyntaxError"},
                ],
            )

    def test_invalid_or_oversized_input_fails_before_host_access(self):
        with mock.patch.object(rooms_backend, "_Lima", side_effect=AssertionError("host used")):
            bad = rooms_backend.evaluate("def evaluate(x): return x", [float("nan")], image=config())
            huge = rooms_backend.evaluate("# " + os.urandom(30_000).hex(), [], image=config())
        self.assertIn("strict JSON", bad["error"])
        self.assertIn("16 KiB", huge["error"])

    def test_pre_set_stop_cancels_only_the_owned_driver(self):
        class SlowLima(FakeLima):
            def start(self, _rooms_argv, _remote_dir):
                return FakeProcess(b"", polls_before_exit=100)

        stop = threading.Event()
        stop.set()
        with mock.patch.object(rooms_backend, "_Lima", SlowLima):
            result = rooms_backend.evaluate("def evaluate(x): return x", [], image=config(), stop=stop)
        self.assertEqual(result["error"], "stopped")
        self.assertTrue(SlowLima.instance.cancelled)

    def test_noncompleted_matrix_runs_owned_cleanup(self):
        class FailedMatrixLima(FakeLima):
            def start(self, _rooms_argv, _remote_dir):
                matrix = {"status": "failed", "clones": []}
                return FakeProcess((json.dumps(matrix) + "\n").encode())

        with mock.patch.object(rooms_backend, "_Lima", FailedMatrixLima):
            result = rooms_backend.evaluate("def evaluate(x): return x", [], image=config())
        self.assertIn("did not complete", result["error"])
        self.assertEqual(FailedMatrixLima.instance.cleanup_calls, 1)
        self.assertTrue(result["evidence_summary"]["teardown"]["complete"])

    def test_roster_requires_rooms_list(self):
        lima = rooms_backend._Lima.__new__(rooms_backend._Lima)
        lima.config = config()
        completed = subprocess.CompletedProcess([], 0, b"{}", b"")
        lima.run = mock.Mock(return_value=completed)
        with self.assertRaisesRegex(rooms_backend._BackendError, "invalid roster"):
            lima._roster()

    def test_failed_teardown_probes_cannot_report_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            binary = root / "rooms"
            ip = root / "ip"
            binary.write_text("#!/bin/sh\nprintf not-json\nexit 1\n", encoding="utf-8")
            ip.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            binary.chmod(0o700)
            ip.chmod(0o700)
            env = dict(os.environ)
            env["PATH"] = f"{root}:{env['PATH']}"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    rooms_backend._TEARDOWN,
                    binary,
                    root / "state",
                    "room-test",
                    "rooms-c3",
                    "veth-h3",
                    "tap-fc3",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            probe = json.loads(completed.stdout)
        self.assertFalse(probe["complete"])
        self.assertFalse(probe["checks"]["roster_command_ok"])
        self.assertFalse(probe["checks"]["roster_json_valid"])
        self.assertFalse(probe["checks"]["netns_command_ok"])

    def test_cleanup_derives_network_names_from_owned_slot(self):
        class SlotLima(rooms_backend._Lima):
            def __init__(self):
                self.config = config()
                self.rosters = [
                    {
                        "rooms": [
                            {
                                "id": "room-3",
                                "label": "matrix:candidate-batch-test",
                                "state": "running",
                                "slot": {"index": 3},
                            }
                        ]
                    },
                    {"rooms": []},
                ]
                self.probed = None

            def _roster(self):
                return self.rosters.pop(0)

            def run(self, _argv, **_kwargs):
                return subprocess.CompletedProcess([], 0, b"", b"")

            def teardown(self, clone):
                self.probed = clone
                return {"complete": True}

        lima = SlotLima()
        self.assertTrue(lima.cleanup_owned("candidate-batch-test")["complete"])
        self.assertEqual(lima.probed["namespace"], "rooms-c3")
        self.assertEqual(lima.probed["host_veth"], "veth-h3")


class RoomsBackendLiveTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("BRIDGE_ROOMS_TESTS") == "1",
        "explicit local Rooms execution checks",
    )
    def test_live_python_batch_has_zero_egress_witness_and_cleanup(self):
        source = """def evaluate(request):
    if request.get("explode"):
        raise LookupError("synthetic")
    return {"double": request["value"] * 2}
"""
        result = rooms_backend.evaluate(
            source,
            [{"value": 3}, {"value": 9, "explode": True}],
            image=os.environ["BRIDGE_ROOMS_CONFIG"],
            timeout=50,
        )
        self.assertIsNone(result["error"], result)
        self.assertEqual(result["outputs"], [{"double": 6}, {"candidate_exception": "LookupError"}])
        self.assertEqual(result["witness"]["egress_policy"], "none")
        self.assertTrue(result["witness"]["capture_complete"])
        self.assertEqual(result["witness"]["permitted"], [])
        self.assertTrue(result["evidence_summary"]["teardown"]["complete"])


if __name__ == "__main__":
    unittest.main()
