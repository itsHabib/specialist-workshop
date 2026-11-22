"""Train one support router and replay predeclared matched text perturbations.

Run from the kit root with python -m experiments.decision_limits.text_routing.
"""
import argparse
import hashlib
import json
from pathlib import Path

from decision_workshop.contracts import read_records
from decision_workshop.core import digest, load_model, predict, write_new
from decision_workshop.evaluation import tally
from decision_workshop.maker import make

LABELS = ["billing", "account_access", "delivery", "software_fault"]
TASK = {
    "name": "support_routing", "version": "1", "question": "Which team should handle the current unresolved problem?",
    "features": {"message": "text"}, "labels": LABELS,
    "descriptions": dict(zip(LABELS, (
        "Resolve a charge, price or payment dispute.", "Restore a customer's access to their account.",
        "Locate or correct a shipment.", "Investigate a malfunction in the customer's application.",
    ))),
}
# A family has a common voice and one independently authored decisive clue per route.
# No wrapper or decisive sentence occurs in multiple splits.
FAMILIES = {
    "train": [
        ("Can you help? {}", "My card was charged twice.", "My login code has expired.", "My parcel never arrived.", "The app crashes on startup."),
        ("Today's problem: {}", "This invoice uses the wrong price.", "I cannot reset my password.", "My delivery went to the wrong building.", "Opening the search tab freezes the screen."),
        ("I need someone to check this. {}", "I paid once but there are two debits.", "The sign-in page says my account is locked.", "Tracking hasn't changed since Tuesday.", "Exporting my report closes the program."),
        ("A customer called. {}", "The refund is missing from the card statement.", "The verification email never reaches their inbox.", "The courier left the box at another address.", "The dashboard shows an error on every refresh."),
        ("Here is what went wrong. {}", "The subscription renewal cost more than quoted.", "My recovery link no longer works.", "The shipping address on the parcel is wrong.", "The button to save a draft does nothing."),
        ("I noticed the following today. {}", "One receipt includes a fee I did not authorize.", "The authenticator code keeps being rejected.", "The item is still in transit a week later.", "The mobile client restarts whenever I open settings."),
        ("Please look at my report. {}", "I need to dispute a credit card payment.", "I lost my backup sign-in codes.", "The shipment arrived at somebody else's door.", "The application closes when I import a file."),
        ("A visitor wrote: {}", "My latest bill includes two service fees.", "I am locked out after replacing my phone.", "My package appears to be stuck at a depot.", "The calendar view keeps flickering then goes blank."),
        ("This issue is still open. {}", "The promised credit never appeared on my bill.", "I cannot get past the password challenge.", "The carrier cannot locate my shipment.", "The print preview displays the wrong page every time."),
        ("My request concerns this incident. {}", "The payment amount exceeds my order total.", "My account recovery token is invalid.", "The tracking page shows delivered but there is no package.", "The software hangs whenever I edit a saved entry."),
        ("What should I do about this? {}", "I was billed after my cancellation.", "Signing in returns me to the identity check.", "The shipping label names a different street.", "The upload panel stops responding midway."),
        ("I still need help with this. {}", "The coupon was ignored on my invoice.", "I cannot use my security key to log in.", "No one has delivered the replacement box.", "The application shows a blank screen after updating."),
    ],
    "dev": [
        ("Support desk note: {}", "The customer sees an extra withdrawal for a single purchase.", "The customer misplaced the device used to confirm their identity.", "The customer's order remains at the sorting center.", "The customer's spreadsheet editor exits while sorting."),
        ("I would appreciate a hand. {}", "The invoice includes a plan I never purchased.", "I have access to email but the sign-in link keeps expiring.", "The courier marked a delivery at a house I don't recognize.", "Every time I scroll, the page goes white."),
    ],
    "test": [
        ("Still unresolved after yesterday's conversation: {}", "My statement shows the same purchase on two separate lines.", "I changed phones and can no longer prove it's me at sign-in.", "The van drove off with the box meant for my address.", "Selecting a file makes the program disappear."),
        ("Someone asked what needs attention. {}", "Our bookkeeper found an unapproved surcharge.", "The temporary link to regain entry says it has been used.", "The parcel label points to our office before we moved.", "The desktop tool closes as soon as I click preview."),
        ("I am following up because of one thing. {}", "The money for the returned item has yet to be credited.", "I cannot satisfy the second step of the identity check.", "The driver photographed a doorstep that isn't mine.", "Editing a saved note now makes the interface unresponsive."),
        ("The problem I want investigated is this. {}", "A canceled membership still appears on this month's statement.", "I forgot my secret phrase and cannot get back into my profile.", "The box has circled the same distribution center three times.", "The graphing tool displays a solid gray rectangle instead of a chart."),
        ("Please assign someone to this case. {}", "We agreed on a discount which was left off the final total.", "The one-time code is consistently refused before I can enter.", "The delivery service says the carton disappeared en route.", "The update leaves the tool frozen at the welcome screen."),
        ("This is the reason for opening the ticket. {}", "There is a bank debit for a service I never ordered.", "My membership is active but the portal refuses my credentials.", "The item went to another town after the label was misprinted.", "Whenever I export my list the window shuts without saving."),
    ],
}
RESOLVED = ("A previous problem with a duplicate charge was resolved last month.",
            "A previous problem with my password was resolved last month.",
            "A previous problem with a missing package was resolved last month.",
            "A previous problem with the app crashing was resolved last month.")
GRID = (0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
CONDITIONS = ("clean", "irrelevant", "hidden", "omitted")
PROTOCOL = {
    "train_families": 12, "dev_families": 2, "test_families": 6,
    "recipe": {"epochs": 12, "seed": 17, "batch_size": 8, "learning_rate": 0.0003},
    "grid": GRID, "required_coverage": 0.8, "required_accuracy_among_suggestions": 0.95,
    "threshold_rule": "Use the first ascending cutoff meeting both targets on clean dev; if none qualify, report zero and disclose.",
    "conditions": CONDITIONS, "reference_subset": "All four routes and all four conditions from the first test family.",
}


def prepare_data(out):
    clean, probes = [], []
    for split, families in FAMILIES.items():
        for number, (wrapper, *clues) in enumerate(families, 1):
            group = f"{split}-{number:02}"
            for i, clue in enumerate(clues):
                text = wrapper.format(clue)
                row = {"id": f"{group}-{i + 1}", "group": group, "split": split,
                       "state": {"message": text}, "target": LABELS[i], "candidates": LABELS}
                clean.append(row)
                if split == "test":
                    # Cycle through the three wrong teams as distractors across test families.
                    distractor = (i + 1 + ((number - 1) % 3)) % len(LABELS)
                    variants = {
                        "clean": row,
                        "irrelevant": {**row, "state": {"message": text + " " + RESOLVED[distractor]}},
                        "hidden": {**row, "state": {"message": wrapper.format("The details are unavailable.")}},
                        "omitted": {**row, "candidates": [option for option in LABELS if option != row["target"]]},
                    }
                    for condition, variant in variants.items():
                        probes.append({**variant, "id": row["id"] + "-" + condition})
    for name, records in (("training-task", clean), ("all-probes", [r for r in clean if r["split"] != "test"] + probes),
                          ("clef-task", [r for r in clean if r["split"] != "test"] + [r for r in probes if r["group"] == "test-01"])):
        directory = out / name
        directory.mkdir()
        write_new(directory / "task.json", TASK)
        with (directory / "records.jsonl").open("x", encoding="utf-8") as stream:
            for row in records:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        read_records(directory / "records.jsonl", TASK)
    return clean, probes


def summarize(rows):
    result = tally([(r["decision"]["action"], r["target"]) for r in rows])
    result["raw_correct"] = sum(r["decision"]["preferred"] == r["target"] for r in rows)
    result["raw_tied"] = sum(r["decision"]["preferred"] is None for r in rows)
    result["mean_target_probability"] = sum(r["decision"]["scores"][r["target"]] for r in rows) / len(rows)
    result["mean_top_probability"] = sum(max(r["decision"]["scores"].values()) for r in rows) / len(rows)
    return result


def replay(model, rows, cutoff):
    return [{"id": row["id"], "target": row["target"], "decision": predict(model, row["state"], row["candidates"], min_score=cutoff)}
            for row in rows]


def run(base, out):
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "protocol.json", {**PROTOCOL, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    clean, probes = prepare_data(out)
    make(out / "training-task", out / "maker", base, **PROTOCOL["recipe"])
    report = {"protocol_sha256": digest(PROTOCOL), "models": {}}
    for name, filename in (("bert", "model"), ("tree", "tree.json")):
        model = load_model(out / "maker" / filename)
        dev = [r for r in clean if r["split"] == "dev"]
        grid = {c: summarize(replay(model, dev, c)) for c in GRID}
        passing = [c for c,v in grid.items() if v["coverage"] >= 0.8 and (v["match_when_suggested"] or 0) >= 0.95]
        threshold = passing[0] if passing else 0.0
        policies = {}
        for c in sorted({0.0, threshold}):
            output = replay(model, probes, c)
            by_id = {r["id"]: r for r in output}
            conditions = {kind: summarize([r for r in output if r["id"].endswith("-" + kind)]) for kind in CONDITIONS}
            for kind in CONDITIONS[1:]:
                changed = 0
                for row in output:
                    if row["id"].endswith("-clean"):
                        sibling = by_id[row["id"].removesuffix("clean") + kind]
                        changed += sibling["decision"]["preferred"] != row["decision"]["preferred"]
                conditions[kind]["preference_changes_from_clean"] = changed
            write_new(out / f"{name}-{c:.2f}-test.json", output)
            policies[c] = conditions
        report["models"][name] = {"checkpoint_sha256": model["sha256"], "dev_grid": grid, "qualifies_on_dev": bool(passing),
                                  "frozen_threshold": threshold, "test": policies}
    write_new(out / "summary.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path(".state/base"))
    parser.add_argument("--out", type=Path, default=Path(".state/limits-text"))
    args = parser.parse_args()
    run(args.base, args.out)
