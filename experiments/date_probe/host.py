"""A date importer with one convention per feed and a purchasable trusted witness."""

from dataclasses import dataclass
from datetime import date
from functools import lru_cache
import random


@dataclass(frozen=True)
class Feed:
    id: str
    channel: str
    notice: bool
    hint: str
    payload: bytes
    expected: tuple[str, ...]
    witness: tuple[str, str]


@dataclass(frozen=True)
class Batch:
    id: str
    family: str
    feeds: tuple[Feed, ...]


def _make_feed(rng, feed_id, channel, notice, hint_wrong):
    hint = rng.choice(("dmy", "mdy"))
    order = ({"dmy": "mdy", "mdy": "dmy"}[hint] if hint_wrong else hint)
    size = rng.randint(2, 8)
    has_day_over_twelve = rng.random() < 0.18
    wire, expected = [], []
    for i in range(size):
        month = rng.randint(1, 12)
        day = rng.randint(1, 12)
        if i == 0 and day == month:
            # Every generated feed has a record usable as an independent witness.
            day = day % 12 + 1
        if i == size - 1 and has_day_over_twelve:
            day = rng.randint(13, 28)
        value = date(rng.randint(2017, 2034), month, day)
        raw = value.strftime("%d/%m/%Y" if order == "dmy" else "%m/%d/%Y")
        wire.append(raw)
        expected.append(value.isoformat())
    payload = ("\n".join([f"feed_id={feed_id}", *wire]) + "\n").encode("ascii")
    return Feed(feed_id, channel, notice, hint, payload, tuple(expected), (wire[0], expected[0]))


def _batch(split, index, family):
    batch_id = f"{split}-{index:03}"
    rng = random.Random(f"date-probe-v1/{batch_id}")
    feeds = []
    for channel in ("alpha", "beta"):
        notice = rng.random() < 0.5
        migration_probability = 0.70 if notice else 0.25
        shared_switch = rng.random() < migration_probability
        for i in range(4):
            switch = shared_switch if family == "coupled" else rng.random() < migration_probability
            wrong = switch != (rng.random() < 0.12)
            feeds.append(_make_feed(rng, f"{batch_id}-{channel}-{i}", channel, notice, wrong))
    rng.shuffle(feeds)
    return Batch(batch_id, family, tuple(feeds))


def batches(split):
    """Generate disjoint file IDs, payloads and independent per-batch random streams."""
    sizes = {"fit": 96, "dev": 32, "test": 128}
    return [_batch(split, i, "independent" if split == "test" and i >= 64 else "coupled")
            for i in range(sizes[split])]


@lru_cache(maxsize=4096)
def _interpretations(payload):
    """Use only public bytes. A convention is feasible if every row parses under it."""
    header, *raw_dates = payload.decode("ascii").splitlines()
    if not header.startswith("feed_id=") or not raw_dates:
        raise ValueError("expected a feed header and at least one date")
    components = []
    for raw in raw_dates:
        parts = raw.split("/")
        if len(parts) != 3 or list(map(len, parts)) != [2, 2, 4] or not all(p.isdigit() for p in parts):
            raise ValueError(f"invalid wire date {raw!r}")
        components.append(tuple(map(int, parts)))

    outcomes = {}
    for order in ("dmy", "mdy"):
        try:
            outcomes[order] = tuple(
                (date(year, second, first) if order == "dmy" else date(year, first, second)).isoformat()
                for first, second, year in components
            )
        except ValueError:
            continue
    if not outcomes:
        raise ValueError("no date convention parses the complete feed")
    return raw_dates, outcomes


def observe(feed):
    """Return unassisted ISO values, count of still-ambiguous rows, and resolved flag."""
    _, outcomes = _interpretations(feed.payload)
    if len(outcomes) == 1:
        return next(iter(outcomes.values())), 0, True
    ambiguous = sum(a != b for a, b in zip(outcomes["dmy"], outcomes["mdy"]))
    return outcomes[feed.hint], ambiguous, ambiguous == 0


def free_history(batch):
    """Read hint outcomes proved by public bytes before buying any witness."""
    history = []
    for feed in batch.feeds:
        _, outcomes = _interpretations(feed.payload)
        if len(outcomes) == 1:
            history.append((feed.channel, feed.hint not in outcomes))
    return history


def probe(feed):
    """Buy one trusted raw/ISO pair to choose the convention of this entire feed."""
    if observe(feed)[2]:
        raise ValueError("this feed is already resolved")
    raw_dates, outcomes = _interpretations(feed.payload)
    raw, trusted = feed.witness
    if raw not in raw_dates:
        raise ValueError("witness date is absent from feed")
    index = raw_dates.index(raw)
    matching = [order for order, rows in outcomes.items() if rows[index] == trusted]
    if len(matching) != 1:
        raise ValueError("witness must uniquely identify one feasible convention")
    order = matching[0]
    return outcomes[order], order != feed.hint
