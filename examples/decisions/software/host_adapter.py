"""Translate synthetic CI events to records for the software-routing task."""

import json


def iter_records(source):
    if source is None:
        raise ValueError("provide the host event file with --source")
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                event = json.loads(line)
                yield {
                    "id": event["run"], "group": event["change"], "split": event["partition"],
                    "state": {"failed_stage": event["stage"],
                              "seen_on_default_branch": event["already_failing"],
                              "consecutive_failures": event["streak"],
                              "duration_seconds": event["seconds"]},
                    "target": event["reviewed_next_step"],
                    "candidates": event.get("offered_steps"),
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
