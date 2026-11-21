"""One predeclared repair test for the text routing specialist. Run after text_routing."""
import argparse
import hashlib
import json
from pathlib import Path

from decision_workshop.contracts import read_json, read_records
from decision_workshop.core import digest, load_model, training_rows, training_summary, write_new
from decision_workshop.maker import make
from .text_routing import FAMILIES, LABELS, TASK, replay, summarize


RECIPE = {"epochs": 12, "batch_size": 8, "learning_rate": 0.0003, "seed": 17}
# Auxiliary tickets come from another team and have already been resolved.
TRAIN_HISTORY = [
    ("The double debit on an earlier order was fixed.", "An old password reset request is closed.",
     "Someone already located my lost parcel.", "My earlier crash report has been closed."),
    ("The billing department already corrected my invoice.", "The login verification problem from last week was fixed.",
     "The old delivery mix-up was sorted out.", "The search panel no longer crashes like it used to."),
    ("I already received my earlier refund.", "I successfully regained access after a previous lockout.",
     "The courier eventually brought my previous order.", "The app update repaired the old loading bug."),
]
# Separate authorship blocks for phrases in this evaluation. None appear in either fit.
UNSEEN = [
    ("I wrote to report this particular issue. {}", "My bank lists three identical withdrawals for a single upgrade.", "Replacing the device left me stranded at the verification challenge.", "The tracking number ends at a depot in the wrong county.", "Pressing the attachment icon takes me straight back to my home screen."),
    ("Here is the one case I need assigned. {}", "I moved to the cheaper tier but the receipt still lists the premium fee.", "I can receive the login message but every token has expired on arrival.", "The driver handed our parcel to an office across town.", "The installed client stalls as soon as the table is sorted."),
    ("The open matter in today's handoff reads as follows. {}", "The store credit shown on the receipt hasn't reached my balance.", "My old phone is needed to authorize entry and I traded it in.", "An empty front step is all I found after the dispatch notice said delivered.", "The desktop editor erases what's on the display and quits during import."),
    ("I'd like this particular problem passed to the right person. {}", "The annual price was withdrawn even though I switched to monthly.", "The website refuses my recovery phrase at the final identity check.", "My replacement is travelling back to the warehouse instead of to my flat.", "Touching the map control locks up the whole app."),
]
TEST_HISTORY = (
    "History only: the finance desk settled a pricing discrepancy in my favor.",
    "History only: a technician restored my sign-in credentials successfully.",
    "History only: the carrier traced an older consignment and handed it to me.",
    "History only: a patch fixed a screen that used to go blank in another product.",
)
PROTOCOL = {"recipe": RECIPE, "threshold": 0.0, "fit_arms": ["original", "duplicate_clean", "augmented"],
            "train_source": "Original 48 training rows; control and augmentation each get a second matched copy.",
            "held_out": "Four fresh phrasing families with four labels; evaluate clean, fresh distraction and hidden clue once.",
            "primary": "Compare augmented versus duplicate-clean accuracy on the same 16 distracted records; also report clean accuracy and preference flips."}


def fresh_probes():
    probes = []
    for idx,(wrapper,*clues) in enumerate(UNSEEN):
        for j,clue in enumerate(clues):
            text = wrapper.format(clue)
            group = f"fresh-{idx+1:02}"
            distractor = (j + 1 + idx % 3) % len(LABELS)
            for condition, msg in (("clean", text), ("irrelevant", text + " " + TEST_HISTORY[distractor]),
                                   ("hidden", wrapper.format("The specific details have been removed."))):
                probes.append({"id": f"{group}-{j+1}-{condition}", "group": group, "split": "test",
                               "state": {"message": msg}, "target": LABELS[j], "candidates": LABELS})
    return probes


def save_dataset(folder, rows):
    folder.mkdir()
    write_new(folder / "task.json", TASK)
    with (folder / "records.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return read_records(folder / "records.jsonl", TASK)


def run(prior, base, out):
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "protocol.json", {**PROTOCOL, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    original_task = read_json(prior / "training-task" / "task.json")
    if original_task != TASK:
        raise ValueError("original task definition mismatch")
    rows = read_records(prior / "training-task" / "records.jsonl", TASK)
    train = training_rows(rows)
    dev = [r for r in rows if r["split"] == "dev"]
    if len(train) != 48 or len(dev) != 8:
        raise ValueError("expected the original 48 training and 8 development rows")
    probes = fresh_probes()
    stale_text = {r['state']['message'] for r in rows}
    if any(r['state']['message'] in stale_text for r in probes):
        raise ValueError("fresh probe duplicates original phrasing")
    data = out / "data"
    data.mkdir()
    save_dataset(data / "original", train + dev + probes)
    checkpoints = {"original": prior / "maker" / "model"}
    for arm in ("duplicate_clean", "augmented"):
        extra = []
        for row in train:
            family = int(row["group"].split("-")[1])-1
            target = LABELS.index(row["target"])
            distractor = (target + 1 + family % 3) % len(LABELS)
            state = row["state"]
            if arm == "augmented":
                history = TRAIN_HISTORY[(family // 3) % 3][distractor]
                state = {"message": state["message"] + " " + history}
            extra.append({**row, "id": row["id"] + "-second", "state": state})
        task_dir = data / arm
        save_dataset(task_dir, train + extra + dev + probes)
        make(task_dir, out / arm, base, **RECIPE)
        checkpoints[arm] = out / arm / "model"

    summary = {"protocol_sha256": digest(PROTOCOL), "probe_sha256": digest(probes), "models": {}}
    for arm, location in checkpoints.items():
        model = load_model(location)
        if arm == "original" and model["training"] != training_summary(TASK, train):
            raise ValueError("original checkpoint training provenance mismatch")
        output = replay(model, probes, 0.0)
        write_new(out / f"{arm}-fresh.json", output)
        by_id = {r["id"]: r["decision"] for r in output}
        results = {}
        for condition in ("clean", "irrelevant", "hidden"):
            cases = [r for r in output if r["id"].endswith("-" + condition)]
            results[condition] = summarize(cases)
            results[condition]["flips_from_clean"] = sum(
                r["decision"]["preferred"] != by_id[r["id"].rsplit("-",1)[0] + "-clean"]["preferred"] for r in cases)
        summary["models"][arm] = {"checkpoint_sha256": model["sha256"], "optimizer_steps": model["optimization"]["optimizer_steps"],
                                  "training_rows": model["training"]["rows"], "conditions": results}
    write_new(out / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, default=Path(".state/limits-text"))
    parser.add_argument("--base", type=Path, default=Path(".state/base"))
    parser.add_argument("--out", type=Path, default=Path(".state/limits-text-repair"))
    args = parser.parse_args()
    run(args.prior, args.base, args.out)
