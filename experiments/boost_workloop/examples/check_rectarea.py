"""Agent-visible development checks for the example rectangle CLI."""
import json
import subprocess

for data, expected in [
    ("0 0 2 2 # square\n", "4"),
    ("0 0 1/3 3/5\n", "1/5"),
    ("0 0 2 2\n1 1 3 3\n", "7"),
]:
    result = subprocess.run(["python", "rectarea.py", "--json"], input=data,
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    wanted = json.dumps({"area": expected}, separators=(",", ":"))
    assert result.stdout.strip() == wanted, f"expected {wanted!r}, got {result.stdout!r}"
print("Exact area, inline comments and compact JSON checks passed")
