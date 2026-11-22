"""Generate labeled public CSV fixtures and execute a two-attempt local importer."""

import csv
import io
import random
from dataclasses import dataclass


ROUTES = {"comma": ",", "semicolon": ";"}
HEADER = ("ticket", "description", "units")
SPLITS = {"fit": (128, 11000), "collect": (256, 22000),
          "dev": (96, 33000), "test": (256, 44000)}


@dataclass(frozen=True)
class ImportJob:
    id: str
    family: str
    payload: bytes
    expected: tuple[tuple[str, ...], ...]
    route_target: str
    declared_hint: str

    @property
    def state(self):
        header = self.payload.partition(b"\n")[0]
        return {"declared_hint": self.declared_hint,
                "comma_count": header.count(b","), "semicolon_count": header.count(b";")}


def other_route(route):
    if route == "comma":
        return "semicolon"
    if route == "semicolon":
        return "comma"
    raise ValueError(f"unknown route {route!r}")


def jobs(split):
    """Allocate factors independently within each family, with fixed margins."""
    count, offset = SPLITS[split]
    families = ("plain", "quoted") if split == "test" else ("plain",)
    size = count // len(families)
    result = []
    for family_number, family in enumerate(families):
        start = offset + family_number * size
        rng = random.Random(start)
        indices = range(size)
        semicolon = set(rng.sample(indices, size // 2))
        legacy = set(rng.sample(indices, size // 5))
        stale_hint = set(rng.sample(indices, size // 4))
        malformed = set(rng.sample(indices, size // 16))
        for i in indices:
            serial = start + i
            route = "semicolon" if i in semicolon else "comma"
            hint = other_route(route) if i in stale_hint else route
            delimiter = ROUTES[route]
            expected = []
            for j in range(2 + i % 3):
                description = ("Café", "Crème", "Zoë")[(i + j) % 3]
                if family == "quoted":
                    description += f'{delimiter} room "B"\nlevel {j + 1}'
                expected.append((f"T{serial}-{j}", description, str((serial + j) % 19 + 1)))

            # The oracle retains the source records before transport corruption.
            wire_rows = list(expected)
            if i in malformed:
                wire_rows[-1] = (*wire_rows[-1][:2], "unknown")
            text = io.StringIO(newline="")
            writer = csv.writer(text, delimiter=delimiter, lineterminator="\r\n")
            writer.writerow(HEADER)
            writer.writerows(wire_rows)
            encoding = "cp1252" if i in legacy else "utf-8"
            payload = text.getvalue().encode(encoding)
            result.append(ImportJob(f"{split}-{serial}", family, payload,
                                    tuple(expected), route, hint))
    return result


def attempt(job, route, action=None):
    """Run a fresh import of the original bytes; return the status and data rows."""
    if route not in ROUTES:
        raise ValueError(f"unknown route {route!r}")
    encoding = "utf-8"
    if action == "switch_delimiter":
        route = other_route(route)
    elif action == "switch_encoding":
        encoding = "cp1252"
    elif action is not None:
        raise ValueError(f"unknown repair {action!r}")

    try:
        stream = io.StringIO(job.payload.decode(encoding), newline="")
    except UnicodeDecodeError:
        return "decode_error", ()
    reader = csv.reader(stream, delimiter=ROUTES[route], strict=True)
    try:
        if tuple(next(reader, ())) != HEADER:
            return "shape_error", ()
        rows = []
        for row in reader:
            if len(row) != len(HEADER):
                return "shape_error", ()
            ticket, description, units = row
            if (not ticket.startswith("T") or not description
                    or not units.isascii() or not units.isdecimal()):
                return "value_error", ()
            rows.append(tuple(row))
    except csv.Error:
        return "shape_error", ()
    return ("ok", tuple(rows)) if rows else ("shape_error", ())
