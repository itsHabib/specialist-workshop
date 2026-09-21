"""Own one native process group and its receipt independently of the driver."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from models import decode

POLL_SECONDS = 0.05
TERMINATE_GRACE_SECONDS = 3


def write(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def group_exists(pid):
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_group(process):
    # The leader may already have exited while a descendant still owns effects.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + TERMINATE_GRACE_SECONDS
    while time.monotonic() < deadline:
        process.poll()
        if not group_exists(process.pid):
            return
        time.sleep(POLL_SECONDS)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def stopping_reason(request, interrupted):
    if os.getppid() != request["owner_pid"]:
        return "owner_lost_usage_unknown"
    if interrupted or (request["stop"] and Path(request["stop"]).exists()):
        return "stopped"
    if time.monotonic() - request["started"] >= request["timeout"]:
        return "timeout_usage_unknown"
    return None


def run(out):
    request = json.loads((out / "request.json").read_text())
    provider = request["provider"]
    receipt = {"provider": provider,
               "requested_model": request.get("requested_model", "claude-opus-5" if provider == "opus" else "gpt-6-astra"),
               "usage": None, "estimated_cost_usd": None, "error": None, "response": None}
    process = None
    interrupted = False

    def request_stop(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        receipt["error"] = stopping_reason(request, interrupted)
        if receipt["error"] is None:
            with tempfile.TemporaryDirectory(prefix="bridge-model-") as cwd:
                with ((out / "prompt.txt").open("rb") as stdin,
                      (out / "stdout.log").open("wb") as stdout,
                      (out / "stderr.log").open("wb") as stderr):
                    process = subprocess.Popen(request["command"], cwd=cwd, stdin=stdin,
                                               stdout=stdout, stderr=stderr, start_new_session=True)
                    try:
                        write(out / "process.json", {"pid": process.pid, "supervisor_pid": os.getpid()})
                        while process.poll() is None:
                            receipt["error"] = stopping_reason(request, interrupted)
                            if receipt["error"]:
                                break
                            time.sleep(POLL_SECONDS)
                    finally:
                        terminate_group(process)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        receipt["error"] = "supervisor_failure:" + type(error).__name__
    receipt.update(seconds=time.monotonic() - request["started"],
                   exit_code=process.returncode if process is not None else None)
    if receipt["error"] is None:
        try:
            decode(receipt, out)
        except (ValueError, KeyError, TypeError, OSError) as error:
            receipt["error"] = "invalid_transport_response:" + type(error).__name__
    write(out / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    run(Path(sys.argv[1]))
