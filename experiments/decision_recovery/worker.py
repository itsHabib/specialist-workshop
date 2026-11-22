"""A local integer-batch worker with durable recovery and a caller-visible clock."""
from dataclasses import dataclass
import json
import sqlite3


FAULTS = ("none", "before", "checkpoint", "lost_reply")
ACTIONS = ("lookup", "retry", "wait", "resume", "return_result")


class WorkerTimeout(TimeoutError):
    """The reply was lost. The exception deliberately omits the fault phase."""


@dataclass(frozen=True)
class Scenario:
    request_id: str
    batches: tuple[tuple[int, ...], ...]
    fault: str
    deadline: int
    lookup_cost: int = 1
    batch_cost: int = 2
    outage_until: int = 0
    wait_cost: int = 1
    retry_fault: str = "none"

    def __post_init__(self):
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a nonempty string")
        if not self.batches or any(not batch or any(type(n) is not int for n in batch)
                                   for batch in self.batches):
            raise ValueError("provide nonempty integer batches")
        object.__setattr__(self, "batches", tuple(tuple(batch) for batch in self.batches))
        if self.fault not in FAULTS or self.retry_fault not in FAULTS:
            raise ValueError("unknown fault")
        for name in ("deadline", "lookup_cost", "batch_cost", "outage_until", "wait_cost"):
            value = getattr(self, name)
            minimum = 1 if name in ("batch_cost", "wait_cost") else 0
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY, input_json TEXT NOT NULL,
    checkpoint_batches INTEGER NOT NULL, subtotal INTEGER NOT NULL, result INTEGER
);
CREATE TABLE IF NOT EXISTS completions (
    request_id TEXT PRIMARY KEY REFERENCES requests(request_id), result INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS work_log (
    request_id TEXT NOT NULL REFERENCES requests(request_id), batch_index INTEGER NOT NULL
);
"""


def _timeout():
    raise WorkerTimeout("worker response timed out")


def _run_worker(db, case, fault, start_batch, subtotal, did_batch, *, reuse_progress=False):
    """Checkpoint monotonically; create result and completion in one SQLite transaction."""
    if fault == "before":
        _timeout()
    key = case.request_id
    payload = json.dumps(case.batches, separators=(",", ":"))
    with db:
        db.execute("INSERT OR IGNORE INTO requests VALUES (?, ?, 0, 0, NULL)", (key, payload))
        stored = db.execute(
            "SELECT input_json, checkpoint_batches, subtotal, result FROM requests WHERE request_id=?",
            (key,),
        ).fetchone()
        if stored["input_json"] != payload:
            raise ValueError("request_id already belongs to a different input")
    if reuse_progress:
        if stored["result"] is not None:
            if fault == "lost_reply":
                _timeout()
            return {"request_id": key, "result": stored["result"]}
        start_batch, subtotal = stored["checkpoint_batches"], stored["subtotal"]

    for index in range(start_batch, len(case.batches)):
        subtotal += sum(case.batches[index])
        with db:
            db.execute("INSERT INTO work_log VALUES (?, ?)", (key, index))
            db.execute(
                "UPDATE requests SET checkpoint_batches=?, subtotal=? "
                "WHERE request_id=? AND result IS NULL AND checkpoint_batches<?",
                (index + 1, subtotal, key, index + 1),
            )
        did_batch()
        if fault == "checkpoint":
            _timeout()

    with db:
        changed = db.execute(
            "UPDATE requests SET result=? WHERE request_id=? AND result IS NULL",
            (subtotal, key),
        ).rowcount
        if changed:
            db.execute("INSERT INTO completions VALUES (?, ?)", (key, subtotal))
        result = db.execute("SELECT result FROM requests WHERE request_id=?", (key,)).fetchone()[0]
    if fault == "lost_reply":
        _timeout()
    return {"request_id": key, "result": result}


def read_ledger(db, request_id):
    """Collect a consistent, request-scoped snapshot after synchronous callers have settled."""
    db.execute("BEGIN")
    try:
        requests = db.execute(
            "SELECT request_id, input_json, checkpoint_batches, subtotal, result "
            "FROM requests WHERE request_id=?", (request_id,)
        ).fetchall()
        completions = db.execute(
            "SELECT request_id, result FROM completions WHERE request_id=?", (request_id,)
        ).fetchall()
        work = db.execute(
            "SELECT batch_index FROM work_log WHERE request_id=? ORDER BY rowid", (request_id,)
        ).fetchall()
        return {"requests": [dict(row) for row in requests],
                "completions": [dict(row) for row in completions], "work": [row[0] for row in work]}
    finally:
        db.rollback()


def grade_ledger(case, ledger, caller, elapsed, stop_reason):
    """Check raw durable state and caller response against a separate sum over original input."""
    expected = sum(item for batch in case.batches for item in batch)
    wanted = {"request_id": case.request_id, "result": expected}
    requests, events, work = ledger["requests"], ledger["completions"], ledger["work"]
    on_time = elapsed <= case.deadline
    valid = len(requests) <= 1
    committed = False
    if requests:
        row = requests[0]
        index = row["checkpoint_batches"]
        valid = (valid and row["request_id"] == case.request_id
                 and json.loads(row["input_json"]) == [list(batch) for batch in case.batches]
                 and type(index) is int and 0 <= index <= len(case.batches)
                 and row["subtotal"] == sum(item for batch in case.batches[:index] for item in batch))
        committed = row["result"] is not None
        if committed:
            valid = valid and index == len(case.batches) and row["result"] == expected
    valid = valid and events == ([wanted] if committed else [])
    if not valid or (caller is not None and (caller != wanted or not committed)):
        status = "incorrect"
    elif caller is not None:
        status = "verified" if on_time else "failed"
    elif stop_reason == "review" and on_time:
        status = "review"
    else:
        status = "failed"
    return {"request_id": case.request_id, "batches": [list(batch) for batch in case.batches],
            "status": status, "caller": caller, "expected_result": expected, "deadline": case.deadline,
            "elapsed": elapsed, "on_time": on_time, "stop_reason": stop_reason, "ledger": ledger,
            "event_count": len(events), "duplicate_effects": max(0, len(events) - 1),
            "batches_processed": len(work), "repeated_batches": len(work) - len(set(work))}


class Episode:
    def __init__(self, scenario, path=":memory:", *, already_interrupted=False, reuse_progress=False):
        """already_interrupted attaches a fresh caller to an existing interrupted request."""
        self.case = scenario
        self.reuse_progress = reuse_progress
        self.db = sqlite3.connect(path, timeout=15.0)
        self.db.row_factory = sqlite3.Row
        self.elapsed = 0
        self.lookups = self.lookup_failures = self.waits = self.recovery_attempts = 0
        self.known = "unknown"
        self.checkpoint_batches = self._subtotal = 0
        self._observed_result = self.caller = None
        self.timeouts = int(already_interrupted)
        try:
            self.db.execute("PRAGMA foreign_keys = ON")
            self.db.executescript(_SCHEMA)
            if not already_interrupted:
                self._invoke(scenario.fault, 0, 0, charge_time=False)
        except BaseException:
            self.close()
            raise

    def _invoke(self, fault, start, subtotal, *, charge_time, reuse_progress=False):
        def did_batch():
            if charge_time:
                self.elapsed += self.case.batch_cost
        try:
            self.caller = _run_worker(self.db, self.case, fault, start, subtotal, did_batch,
                                      reuse_progress=reuse_progress)
        except WorkerTimeout:
            self.timeouts += 1
            self.known = "unknown"
            self.checkpoint_batches = self._subtotal = 0
            self._observed_result = None

    def is_terminal(self):
        if self.caller is not None or self.elapsed > self.case.deadline:
            return True
        # At the deadline any possibly free action gets a chance. Eligibility is
        # based entirely on observed status and public executor capabilities.
        can_finish_now = (self.known == "committed" or self.case.lookup_cost == 0
                          or self.reuse_progress or (self.known == "checkpoint"
                          and self.checkpoint_batches == len(self.case.batches)))
        return self.elapsed == self.case.deadline and not can_finish_now

    def observe(self):
        state = {"remaining": max(0, self.case.deadline - self.elapsed),
                 "batches": len(self.case.batches), "batch_cost": self.case.batch_cost,
                 "lookup_cost": self.case.lookup_cost, "wait_cost": self.case.wait_cost,
                 "lookup_failures": self.lookup_failures, "recovery_attempts": self.recovery_attempts,
                 "elapsed": self.elapsed, "checkpoint_batches": self.checkpoint_batches,
                 "known": self.known}
        if self.is_terminal():
            return state, []
        actions = ["lookup", "retry", "wait"]
        if self.known == "checkpoint":
            actions.append("resume")
        if self.known == "committed":
            actions.append("return_result")
        return state, actions

    def apply(self, action):
        if action not in self.observe()[1]:
            raise ValueError(f"unavailable recovery action: {action}")
        if action == "lookup":
            self.lookups += 1
            self.elapsed += self.case.lookup_cost
            if self.elapsed < self.case.outage_until:
                self.lookup_failures += 1
                return
            row = self.db.execute(
                "SELECT input_json, checkpoint_batches, subtotal, result FROM requests WHERE request_id=?",
                (self.case.request_id,),
            ).fetchone()
            if row is None:
                self.known = "missing"
            else:
                if json.loads(row["input_json"]) != [list(batch) for batch in self.case.batches]:
                    raise ValueError("request_id already belongs to a different input")
                if row["result"] is None:
                    self.known = "checkpoint"
                    self.checkpoint_batches, self._subtotal = row["checkpoint_batches"], row["subtotal"]
                else:
                    self.known = "committed"
                    self._observed_result = row["result"]
        elif action == "wait":
            self.elapsed += self.case.wait_cost
            self.waits += 1
        elif action == "return_result":
            self.caller = {"request_id": self.case.request_id, "result": self._observed_result}
        else:
            fault = self.case.retry_fault if self.recovery_attempts == 0 else "none"
            self.recovery_attempts += 1
            start, subtotal = (self.checkpoint_batches, self._subtotal) if action == "resume" else (0, 0)
            self._invoke(fault, start, subtotal, charge_time=True,
                         reuse_progress=self.reuse_progress and action == "retry")

    def verify(self, stop_reason):
        ledger = read_ledger(self.db, self.case.request_id)
        return {**grade_ledger(self.case, ledger, self.caller, self.elapsed, stop_reason),
                "lookups": self.lookups, "lookup_failures": self.lookup_failures, "waits": self.waits,
                "recovery_attempts": self.recovery_attempts, "timeouts": self.timeouts,
                "reuse_progress": self.reuse_progress}

    def close(self):
        self.db.close()


def hand_demo():
    """Replay the same lost reply twice and retain raw ledger evidence with each trace."""
    case = Scenario("lost-reply-demo", ((2, 3), (-1, 9)), "lost_reply", deadline=10)
    output = {}
    for mode, actions in (("status_first", ("lookup", "return_result")), ("same_id_retry", ("retry",))):
        episode = Episode(case)
        try:
            trace = [{"exception": WorkerTimeout.__name__, "observation": episode.observe()[0],
                      "ledger": read_ledger(episode.db, case.request_id)}]
            for action in actions:
                episode.apply(action)
                trace.append({"action": action, "observation": episode.observe()[0]})
            output[mode] = {"trace": trace, "outcome": episode.verify("terminal")}
        finally:
            episode.close()
    return output


def negative_controls():
    """Reject wrong sums, duplicate events, and an acknowledgement with no durable effect."""
    case = Scenario("negative-control", ((6,), (-2, 3)), "none", deadline=8)
    episode = Episode(case)
    try:
        evidence = read_ledger(episode.db, case.request_id)
        caller = episode.caller
        wrong = {**evidence["requests"][0], "result": 99, "subtotal": 99}
        bad = {
            "wrong_total": ({"requests": [wrong], "completions": [{**caller, "result": 99}],
                             "work": evidence["work"]}, {**caller, "result": 99}),
            "duplicate_event": ({**evidence, "completions": evidence["completions"] * 2}, caller),
            "reply_without_commit": ({"requests": [], "completions": [], "work": []}, caller),
        }
        return {name: grade_ledger(case, ledger, reply, 0, "terminal")
                for name, (ledger, reply) in bad.items()}
    finally:
        episode.close()
