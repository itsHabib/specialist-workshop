"""Import canonical records produced by a host-owned Python adapter."""

from importlib import import_module
import json
import os
from pathlib import Path

from .contracts import iter_validated_records
from .core import new_output_file


def _adapter_rows(adapter, source):
    """Attribute resolution and iteration failures to the selected host adapter."""
    name = adapter if isinstance(adapter, str) else getattr(adapter, "__name__", repr(adapter))
    try:
        if isinstance(adapter, str):
            module, sep, function = adapter.partition(":")
            if not sep or not module or not function.isidentifier():
                raise ValueError("expected module:function")
            adapter = getattr(import_module(module), function)
        if not callable(adapter):
            raise TypeError("expected a callable")
        yield from adapter(source)
    except Exception as exc:
        raise ValueError(f"adapter {name!r}: {exc}") from exc


def import_records(task, adapter, source, out):
    """Call adapter(source) and publish validated JSONL at a fresh output path.

    The adapter may be a callable or an importable 'module:function'. It receives
    a Path, or None when the caller has no source file. Unknown candidate sets
    are excluded from the target-missing count. An empty candidate list counts
    as recorded and missing the target.
    """
    out = Path(out).expanduser()
    if os.path.lexists(out):
        raise FileExistsError(f"output already exists: {out}; choose a new path")
    source = Path(source).expanduser() if source is not None else None
    groups = {split: set() for split in ("train", "dev", "test")}
    rows = recorded = missing = 0
    with new_output_file(out) as handle:
        for row in iter_validated_records(task, _adapter_rows(adapter, source)):
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            rows += 1
            groups[row["split"]].add(row["group"])
            candidates = row["candidates"]
            if candidates is not None:
                recorded += 1
                missing += row["target"] not in candidates
    return {"rows": rows, "groups_per_split": {k: len(v) for k, v in groups.items()},
            "candidate_sets_recorded": recorded, "target_missing_from_candidates": missing}
