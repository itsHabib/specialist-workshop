"""Narrow, zero-egress execution of one candidate batch in Rooms.

The guest receives only candidate source and request data.  Expected answers and
model credentials stay on the host.  Configuration is supplied as a mapping, a
JSON path through ``image``, or ``BRIDGE_ROOMS_CONFIG``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import pathlib
import shlex
import subprocess
import time
import uuid
import zlib
from collections.abc import Mapping


_COMMAND_LIMIT = 16 * 1024
_MANIFEST_LIMIT = 128 * 1024
_OUTPUT_LIMIT = 64 * 1024
_STDERR_LIMIT = 64 * 1024
_CASE_PREFIX = "candidate-batch-"
_RUNNER = """import json
import sys

namespace = {}
with open(sys.argv[2], encoding="utf-8") as request_file:
    requests = json.load(request_file)
try:
    with open(sys.argv[1], encoding="utf-8") as source_file:
        source = source_file.read()
    exec(compile(source, "candidate.py", "exec"), namespace)
    candidate = namespace.get("evaluate")
    if not callable(candidate):
        raise TypeError("candidate source must define evaluate(request)")
except BaseException as error:
    candidate = None
    load_error = type(error).__name__
outputs = []
for request in requests:
    try:
        if candidate is None:
            answer = {"candidate_exception": load_error}
        else:
            answer = candidate(request)
        json.dumps(answer, allow_nan=False)
    except BaseException as error:
        answer = {"candidate_exception": type(error).__name__}
    outputs.append(answer)
with open(sys.argv[3], "w", encoding="utf-8") as output_file:
    json.dump(outputs, output_file, allow_nan=False, separators=(",", ":"))
    output_file.write("\\n")
"""

_DRIVER = """set -eu
umask 077
pidfile=$1
exitfile=$2
runtime=$3
shift 3
timeout --signal=TERM --kill-after=8s "$runtime" "$@" &
child=$!
start=$(awk '{print $22}' "/proc/$child/stat")
printf '%s %s\\n' "$child" "$start" > "$pidfile"
terminate() {
  kill -TERM "$child" 2>/dev/null || true
  wait "$child" 2>/dev/null || true
  exit 143
}
trap terminate HUP INT TERM
set +e
wait "$child"
status=$?
set -e
printf '%s\\n' "$status" > "$exitfile"
exit "$status"
"""

_CANCEL = """import os, pathlib, signal, sys
pid_path = pathlib.Path(sys.argv[1])
marker = sys.argv[2].encode()
try:
    pid_text, expected_start = pid_path.read_text().split()
    pid = int(pid_text)
    stat = pathlib.Path(f"/proc/{pid}/stat").read_text().split()
    command = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes()
    if stat[21] != expected_start or marker not in command:
        print("stale")
        raise SystemExit(0)
    os.kill(pid, signal.SIGTERM)
    print("signaled")
except (FileNotFoundError, ProcessLookupError, ValueError):
    print("absent")
"""

_TEARDOWN = """import json, pathlib, subprocess, sys
binary, state_root, room_id, namespace, host_veth, tap = sys.argv[1:]
roster_run = subprocess.run([binary, "ls", "--json"], capture_output=True, text=True)
roster_json_valid = False
roster_schema_valid = False
rooms = []
try:
    roster = json.loads(roster_run.stdout)
    roster_json_valid = True
    rooms = roster.get("rooms") if isinstance(roster, dict) else None
    roster_schema_valid = isinstance(rooms, list) and all(isinstance(room, dict) for room in rooms)
except (json.JSONDecodeError, TypeError):
    pass
if not roster_schema_valid:
    rooms = []
room_ids = {room.get("id") for room in rooms}
process_found = False
process_scan_complete = True
try:
    process_entries = list(pathlib.Path("/proc").iterdir())
except OSError:
    process_entries = []
    process_scan_complete = False
for entry in process_entries:
    if not entry.name.isdigit():
        continue
    try:
        executable = (entry / "exe").resolve().name
        command = (entry / "cmdline").read_bytes()
        if executable in {"firecracker", "jailer", "rooms"} and room_id.encode() in command:
            process_found = True
            break
    except PermissionError:
        process_scan_complete = False
    except (FileNotFoundError, ProcessLookupError):
        pass
netns_run = subprocess.run(["ip", "netns", "list"], capture_output=True, text=True)
netns = netns_run.stdout.splitlines() if netns_run.returncode == 0 else []
state = pathlib.Path(state_root)
checks = {
    "roster_command_ok": roster_run.returncode == 0,
    "roster_json_valid": roster_json_valid,
    "roster_schema_valid": roster_schema_valid,
    "room_absent_from_roster": room_id not in room_ids,
    "room_state_absent": not (state / room_id).exists(),
    "jail_absent": not (state / "jailer" / "firecracker" / room_id).exists(),
    "process_absent": not process_found,
    "process_scan_complete": process_scan_complete,
    "netns_command_ok": netns_run.returncode == 0,
    "namespace_absent": not any(line.split()[0] == namespace for line in netns if line.split()),
    "host_veth_absent": not (pathlib.Path("/sys/class/net") / host_veth).exists(),
    "tap_absent": not (pathlib.Path("/sys/class/net") / tap).exists(),
}
print(json.dumps({"checks": checks, "complete": all(checks.values())}, sort_keys=True))
"""


class _BackendError(RuntimeError):
    pass


def _failure(image, error, stderr="", seconds=0.0, evidence=None, **extra):
    result = {
        "backend": "rooms",
        "image": image,
        "outputs": None,
        "error": str(error),
        "stderr": stderr[-_STDERR_LIMIT:],
        "seconds": seconds,
        "evidence": evidence,
        "witness": None,
    }
    result.update(extra)
    return result


def _load_config(image):
    if isinstance(image, Mapping):
        config = dict(image)
    else:
        path = image or os.environ.get("BRIDGE_ROOMS_CONFIG")
        if not path:
            raise _BackendError("set BRIDGE_ROOMS_CONFIG or pass a config JSON path as image")
        with open(os.fspath(path), encoding="utf-8") as source:
            config = json.load(source)
    required = (
        "host",
        "binary",
        "snapshot",
        "snapshot_sha256",
        "image_path",
        "image_sha256",
        "toolstore",
        "toolstore_sha256",
        "remote_root",
        "state_root",
        "identity",
    )
    missing = [name for name in required if not config.get(name)]
    if missing:
        raise _BackendError(f"Rooms config missing: {', '.join(missing)}")
    config.setdefault("limactl", "limactl")
    for name in ("binary", "snapshot", "image_path", "toolstore", "remote_root", "state_root"):
        if not pathlib.PurePosixPath(config[name]).is_absolute():
            raise _BackendError(f"Rooms config {name} must be an absolute path")
    if pathlib.PurePosixPath(config["remote_root"]) == pathlib.PurePosixPath("/"):
        raise _BackendError("Rooms config remote_root cannot be /")
    return config


def _packed(value):
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return base64.b64encode(zlib.compress(raw, level=9)).decode("ascii")


def _manifest(source, requests, case_id):
    if not isinstance(source, str):
        raise _BackendError("candidate source must be a string")
    if not isinstance(requests, list):
        raise _BackendError("requests must be a list")
    try:
        request_bytes = json.dumps(
            requests, allow_nan=False, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise _BackendError(f"requests are not strict JSON: {error}") from error
    python = "/nix/var/rooms/env/bin/python3"
    decode = (
        "import base64,sys,zlib;"
        "sys.stdout.buffer.write(zlib.decompress(base64.b64decode(sys.stdin.buffer.read())))"
    )
    paths = {
        "source": "/workspace/in/candidate.py",
        "requests": "/workspace/in/requests.json",
        "runner": "/workspace/in/runner.py",
        "outputs": "/workspace/out/outputs.json",
    }
    lines = [
        "set -eu",
        "umask 077",
        "ulimit -t 5",
        "ulimit -f 128",
        "ulimit -u 32",
        "ulimit -v 262144",
        "mkdir -p /workspace/in /workspace/out",
    ]
    for name, payload in (
        ("source", _packed(source)),
        ("requests", _packed(request_bytes)),
        ("runner", _packed(_RUNNER)),
    ):
        lines.append(
            f"printf %s {shlex.quote(payload)} | {python} -c {shlex.quote(decode)}"
            f" > {paths[name]}"
        )
    lines.extend(
        [
            "chmod 0400 /workspace/in/candidate.py /workspace/in/requests.json /workspace/in/runner.py",
            f"{python} {paths['runner']} {paths['source']} {paths['requests']} {paths['outputs']}"
            " > /workspace/out/candidate.stdout 2> /workspace/out/candidate.stderr",
            "size=$(wc -c < /workspace/out/outputs.json)",
            f"test \"$size\" -le {_OUTPUT_LIMIT}",
        ]
    )
    command = "\n".join(lines) + "\n"
    if len(command.encode("utf-8")) > _COMMAND_LIMIT:
        raise _BackendError("compressed candidate batch exceeds Rooms' 16 KiB command limit")
    manifest = {
        "schema": "rooms.matrix.v1",
        "cases": [{"id": case_id, "command": command}],
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > _MANIFEST_LIMIT:
        raise _BackendError("matrix manifest exceeds Rooms' 128 KiB limit")
    return encoded


def _stop_requested(stop):
    if stop is None:
        return False
    if isinstance(stop, (str, os.PathLike)):
        return pathlib.Path(stop).exists()
    if callable(stop):
        return bool(stop())
    is_set = getattr(stop, "is_set", None)
    return bool(is_set()) if callable(is_set) else bool(stop)


def _json_line(text, what):
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise _BackendError(f"{what} did not emit a JSON record")


class _Lima:
    def __init__(self, config):
        self.config = config
        self.prefix = [config["limactl"], "shell", config["host"]]

    def run(self, argv, *, data=None, timeout=15, check=True):
        completed = subprocess.run(
            self.prefix + argv,
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if check and completed.returncode != 0:
            message = completed.stderr.decode("utf-8", "replace")[-4000:]
            raise _BackendError(f"Rooms host command failed: {message.strip()}")
        return completed

    def prepare(self, remote_dir, manifest):
        self.run(["sudo", "-H", "install", "-d", "-m", "700", remote_dir])
        self.run(
            ["sudo", "-H", "tee", f"{remote_dir}/manifest.json"],
            data=manifest,
        )
        self.run(["sudo", "-H", "chmod", "600", f"{remote_dir}/manifest.json"])

    def start(self, rooms_argv, remote_dir):
        pidfile = f"{remote_dir}/driver.pid"
        exitfile = f"{remote_dir}/driver.exit"
        argv = self.prefix + [
            "sudo",
            "-H",
            "sh",
            "-c",
            _DRIVER,
            "bridge-rooms-driver",
            pidfile,
            exitfile,
            "45s",
            *rooms_argv,
        ]
        return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def cancel(self, remote_dir):
        result = None
        for _ in range(20):
            result = self.run(
                [
                    "sudo",
                    "-H",
                    "python3",
                    "-c",
                    _CANCEL,
                    f"{remote_dir}/driver.pid",
                    remote_dir,
                ],
                check=False,
            )
            if result.stdout.strip() != b"absent":
                return result
            time.sleep(0.1)
        return result

    def read(self, path, limit=_OUTPUT_LIMIT):
        result = self.run(["sudo", "-H", "head", "-c", str(limit + 1), path])
        if len(result.stdout) > limit:
            raise _BackendError(f"Rooms artifact exceeds {limit} bytes: {path}")
        return result.stdout

    def teardown(self, clone):
        tap = f"tap-fc{clone['slot']}"
        result = self.run(
            [
                "sudo",
                "-H",
                "python3",
                "-c",
                _TEARDOWN,
                self.config["binary"],
                self.config["state_root"],
                clone["room_id"],
                clone["namespace"],
                clone["host_veth"],
                tap,
            ]
        )
        return _json_line(result.stdout.decode("utf-8", "replace"), "teardown probe")

    def _roster(self):
        result = self.run(["sudo", "-H", self.config["binary"], "ls", "--json"])
        try:
            roster = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError) as error:
            raise _BackendError("rooms ls did not emit a JSON document") from error
        if not isinstance(roster, dict) or not isinstance(roster.get("rooms"), list):
            raise _BackendError("rooms ls emitted an invalid roster")
        if not all(isinstance(room, dict) for room in roster["rooms"]):
            raise _BackendError("rooms ls emitted an invalid room record")
        return roster

    def cleanup_owned(self, case_id):
        """Reap only the room carrying this invocation's unguessable case label."""
        label = f"matrix:{case_id}"
        owned = [room for room in self._roster().get("rooms", []) if room.get("label") == label]
        if len(owned) > 1:
            return {"complete": False, "error": "multiple rooms carried the unique case label"}
        if not owned:
            return {"complete": False, "checks": {"owned_room_absent": True},
                    "error": "room identity unavailable; residue absence unproved"}
        room = owned[0]
        room_id = room.get("id")
        slot = room.get("slot", {}).get("index")
        if not isinstance(room_id, str) or not isinstance(slot, int):
            return {"complete": False, "error": "owned room identity was incomplete"}
        if room.get("state") in {"running", "kept"}:
            self.run(
                ["sudo", "-H", self.config["binary"], "kill", room_id, "--json"],
                check=False,
            )
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            current = next(
                (item for item in self._roster().get("rooms", []) if item.get("id") == room_id),
                None,
            )
            if current is None:
                break
            if current.get("state") == "orphaned-dead":
                self.run(
                    ["sudo", "-H", self.config["binary"], "gc", room_id],
                    check=False,
                )
            time.sleep(0.1)
        clone = {
            "room_id": room_id,
            "slot": slot,
            "namespace": f"rooms-c{slot}",
            "host_veth": f"veth-h{slot}",
        }
        return self.teardown(clone)

    def write_public(self, path, record):
        payload = json.dumps(record, sort_keys=True, indent=2).encode("utf-8") + b"\n"
        self.run(["sudo", "-H", "tee", path], data=payload)
        self.run(["sudo", "-H", "chmod", "600", path])


def _wait(process, lima, remote_dir, timeout, stop):
    began = time.monotonic()
    reason = None
    while process.poll() is None:
        if _stop_requested(stop):
            reason = "stopped"
            break
        if time.monotonic() - began >= timeout:
            reason = "local timeout"
            break
        time.sleep(0.05)
    if reason:
        lima.cancel(remote_dir)
    try:
        stdout, stderr = process.communicate(timeout=15 if reason else 1)
    except subprocess.TimeoutExpired:
        lima.cancel(remote_dir)
        process.terminate()
        stdout, stderr = process.communicate(timeout=5)
    return stdout, stderr, reason


def evaluate(source: str, requests: list, image=None, timeout=50, stop=None) -> dict:
    """Execute ``evaluate(request)`` once per request in one fresh Rooms VM."""
    began = time.monotonic()
    config = None
    identity = None
    stderr_text = ""
    evidence = None
    lima = None
    case_id = None
    remote_dir = None
    host_command_sha256 = None
    started = False
    teardown = None
    try:
        config = _load_config(image)
        identity = config["identity"]
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise _BackendError("timeout must be a positive number")
        run_id = uuid.uuid4().hex
        case_id = f"{_CASE_PREFIX}{run_id[:20]}"
        manifest = _manifest(source, requests, case_id)
        remote_dir = f"{config['remote_root'].rstrip('/')}/{run_id}"
        evidence = f"{config['host']}:{remote_dir}/public.json"
        lima = _Lima(config)
        lima.prepare(remote_dir, manifest)
        out_dir = f"{remote_dir}/out"
        rooms_argv = [
            config["binary"],
            "matrix",
            config["snapshot"],
            "--image",
            config["image_path"],
            "--toolstore",
            config["toolstore"],
            "--cases",
            f"{remote_dir}/manifest.json",
            "--out",
            out_dir,
            "--witness",
            "--egress",
            "none",
            "--max-wall",
            "30s",
            "--max-pool",
            "1",
            "--json",
        ]
        host_command_sha256 = hashlib.sha256(
            json.dumps(rooms_argv, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        process = lima.start(rooms_argv, remote_dir)
        started = True
        stdout, stderr, interrupted = _wait(process, lima, remote_dir, float(timeout), stop)
        stderr_text = stderr.decode("utf-8", "replace")[-_STDERR_LIMIT:]
        if interrupted:
            raise _BackendError(interrupted)
        matrix = _json_line(stdout.decode("utf-8", "replace"), "rooms matrix")
        clones = matrix.get("clones")
        if matrix.get("status") != "completed" or not isinstance(clones, list) or len(clones) != 1:
            raise _BackendError(f"Rooms matrix did not complete: {matrix.get('status')}")
        clone = clones[0]
        required = ('room_id', 'snapshot_id', 'command_sha256', 'namespace', 'host_veth')
        if (process.returncode != 0 or not isinstance(clone, dict)
                or clone.get('case_id') != case_id
                or any(not isinstance(clone.get(key), str) or not clone[key] for key in required)
                or not isinstance(clone.get('slot'), int)
                or not isinstance(matrix.get('matrix_sha256'), str)):
            raise _BackendError('incomplete Rooms execution identity')
        teardown = lima.teardown(clone)
        if teardown.get("complete") is not True:
            raise _BackendError("Rooms teardown evidence is incomplete")
        if clone.get("exit_code") != 0 or clone.get("status") != "exited":
            raise _BackendError(f"candidate batch failed with exit {clone.get('exit_code')}")
        case_dir = f"{out_dir}/{case_id}"
        outputs = json.loads(lima.read(f"{case_dir}/outputs.json").decode("utf-8"))
        if not isinstance(outputs, list) or len(outputs) != len(requests):
            raise _BackendError("candidate did not return one answer per request")
        witness = json.loads(lima.read(f"{case_dir}/witness.json").decode("utf-8"))
        if (
            witness.get("egress_policy") != "none"
            or witness.get("capture_complete") is not True
            or witness.get("permitted") != []
        ):
            raise _BackendError("Rooms zero-egress witness is incomplete")
        summary = {
            "schema": "investigation-bridge.rooms-evidence.v1",
            "run_id": run_id,
            "image": identity,
            "snapshot_sha256": config["snapshot_sha256"],
            "image_sha256": config["image_sha256"],
            "toolstore_sha256": config["toolstore_sha256"],
            "host_command_sha256": host_command_sha256,
            "matrix_sha256": matrix.get("matrix_sha256"),
            "command_sha256": clone.get("command_sha256"),
            "snapshot_id": clone.get("snapshot_id"),
            "room_id": clone.get("room_id"),
            "witness": {
                "capture_complete": True,
                "egress_policy": "none",
                "permitted": [],
                "blocked": witness.get("blocked", []),
            },
            "teardown": teardown,
        }
        lima.write_public(f"{remote_dir}/public.json", summary)
        return {
            "backend": "rooms",
            "image": identity,
            "outputs": outputs,
            "error": None,
            "stderr": stderr_text,
            "seconds": time.monotonic() - began,
            "evidence": evidence,
            "witness": summary["witness"],
            "evidence_summary": summary,
        }
    except Exception as error:
        error_text = str(error)
        if started and (not isinstance(teardown, dict) or teardown.get("complete") is not True):
            try:
                teardown = lima.cleanup_owned(case_id)
            except Exception as cleanup_error:
                teardown = {"complete": False, "error": str(cleanup_error)}
        extra = {}
        if started:
            if not isinstance(teardown, dict) or teardown.get("complete") is not True:
                error_text = f"{error_text}; cleanup incomplete"
            summary = {
                "schema": "investigation-bridge.rooms-evidence.v1",
                "run_id": run_id,
                "image": identity,
                "status": "failed",
                "snapshot_sha256": config["snapshot_sha256"],
                "host_command_sha256": host_command_sha256,
                "teardown": teardown,
            }
            try:
                lima.write_public(f"{remote_dir}/public.json", summary)
            except Exception as evidence_error:
                evidence = None
                summary["evidence_error"] = str(evidence_error)
            extra["evidence_summary"] = summary
        if not started:
            evidence = None
        return _failure(
            identity,
            error_text,
            stderr=stderr_text,
            seconds=time.monotonic() - began,
            evidence=evidence,
            **extra,
        )
