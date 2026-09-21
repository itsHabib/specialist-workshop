"""Disposable candidate execution; the reference and answers stay on the host."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

IMAGE = "python@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea"
WRAPPER = '''import json,resource,sys
resource.setrlimit(resource.RLIMIT_CPU,(5,5))
resource.setrlimit(resource.RLIMIT_FSIZE,(1048576,1048576))
payload=json.load(sys.stdin)
namespace={"__name__":"candidate"}
exec(compile(payload["source"],"solution.py","exec"),namespace)
answers=[]
for request in payload["requests"]:
    try:
        answers.append(namespace["evaluate"](request))
    except Exception as error:
        answers.append({"candidate_exception":type(error).__name__})
print(json.dumps(answers))
'''



def evaluate(source, requests, image=IMAGE, timeout=20, stop=None):
    """Return raw candidate outputs, never inferred successes; clean up our container."""
    if len(source.encode()) > 100_000:
        raise ValueError("candidate exceeds 100 KB")
    name = "investigation-bridge-" + uuid.uuid4().hex[:16]
    started = time.monotonic()
    command = ["docker", "run", "--rm", "--name", name, "--network", "none",
               "--read-only", "--log-driver=none", "--cap-drop=ALL", "--security-opt=no-new-privileges",
               "--pids-limit", "32", "--memory", "128m", "--memory-swap", "128m",
               "--cpus", "1", "--user", "65534:65534", "--tmpfs",
               "/tmp:rw,noexec,nosuid,size=16m,mode=1777", "-i", image,
               "timeout", "-s", "KILL", "15s", "python", "-I", "-c", WRAPPER]
    with tempfile.TemporaryFile() as stdin, tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        stdin.write(json.dumps({"source":source,"requests":requests}).encode()); stdin.seek(0)
        process = subprocess.Popen(command, stdin=stdin, stdout=stdout, stderr=stderr)
        failure = None
        try:
            while process.poll() is None:
                if stop is not None and Path(stop).exists():
                    failure = "stopped"; break
                if time.monotonic() - started > timeout:
                    failure = "timeout"; break
                if os.fstat(stdout.fileno()).st_size + os.fstat(stderr.fileno()).st_size > 2_000_000:
                    failure = "output_limit"; break
                time.sleep(0.05)
        finally:
            removed = subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            cleanup = removed.returncode == 0
            if not cleanup:
                inspected = subprocess.run(["docker", "inspect", name], capture_output=True, timeout=10)
                cleanup = inspected.returncode != 0 and b"No such" in inspected.stderr
        stdout.seek(0); stderr.seek(0)
        output = stdout.read(2_000_000).decode(errors="replace")
        errors = stderr.read(4000).decode(errors="replace")
    result = {"backend": "docker", "image": image, "container": name,
              "returncode": process.returncode, "seconds": time.monotonic() - started,
              "error": failure, "stderr": errors, "outputs": None, "cleanup_confirmed": cleanup}
    if not cleanup:
        result['error'] = 'cleanup_unconfirmed'
        return result
    if failure:
        return result
    if process.returncode:
        result["error"] = "candidate_or_runtime_exit"
        return result
    try:
        values = json.loads(output)
        if not isinstance(values, list) or len(values) != len(requests):
            raise ValueError("wrong output count")
        result["outputs"] = values
    except (ValueError, json.JSONDecodeError):
        result["error"] = "invalid_candidate_output"
    return result
