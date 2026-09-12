"""Export measured runs without redistributing local paths, model weights, or books."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workshop import store


def main():
    target = store.ROOT / "artifacts" / "experiments"
    target.mkdir(parents=True, exist_ok=True)
    summaries = []
    for job in reversed(store.jobs()):
        if job["status"] != "completed":
            continue
        excluded = {"adapter_path", "rows", "log"}
        summary = {key: value for key, value in job.items() if key not in excluded}
        summaries.append(summary)
        if job["kind"] == "train":
            log = store.STATE / "jobs" / job["id"] / "training.log"
            if log.exists():
                # Preserve numerical training telemetry without machine-specific paths.
                lines = [line for line in log.read_text().splitlines()
                         if "loss " in line or "Trainable parameters" in line]
                (target / f"{job['id']}.log").write_text("\n".join(lines) + "\n")
            continue
        rows = [{key: value for key, value in row.items() if key != "input"}
                for row in job.get("rows", [])]
        (target / f"{job['id']}.json").write_text(json.dumps(dict(run=summary, rows=rows), indent=2))
    (target / "index.json").write_text(json.dumps(summaries, indent=2))
    print(f"Exported {len(summaries)} completed runs to {target}")


if __name__ == "__main__":
    main()
