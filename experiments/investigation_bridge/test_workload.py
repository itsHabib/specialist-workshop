import json
import re
import unittest

from experiments.investigation_bridge import workload


def frame(parent, translation, linear, pivot=("0", "0")):
    return {
        "parent": parent,
        "translation": list(translation),
        "linear": [list(linear[0]), list(linear[1])],
        "pivot": list(pivot),
    }


def load_evaluate(source):
    namespace = {}
    exec(compile(source, "solution.py", "exec"), namespace)
    return namespace["evaluate"]


class ReferenceTests(unittest.TestCase):
    def test_identity_is_hand_calculated(self):
        request = {
            "frames": {},
            "source": "world",
            "target": "world",
            "point": ["3/2", "-4"],
        }
        self.assertEqual(workload.reference(request), {"point": ["3/2", "-4"]})

    def test_inverse_is_hand_calculated(self):
        # local [2,3] -> 4,-2 + 1,1 + diag(2,3) * ([2,3] - [1,1]) = [7,5]
        request = {
            "frames": {
                "camera": frame(
                    "world",
                    ("4", "-2"),
                    (("2", "0"), ("0", "3")),
                    ("1", "1"),
                )
            },
            "source": "world",
            "target": "camera",
            "point": ["7", "5"],
        }
        self.assertEqual(workload.reference(request), {"point": ["2", "3"]})

    def test_composition_is_hand_calculated(self):
        # [2,3] -> child parent [3,5] -> rotated and translated world [-2,4].
        request = {
            "frames": {
                "parent": frame(
                    "world", ("3", "1"), (("0", "-1"), ("1", "0"))
                ),
                "child": frame(
                    "parent", ("-1", "2"), (("2", "0"), ("0", "1"))
                ),
            },
            "source": "child",
            "target": "world",
            "point": ["2", "3"],
        }
        self.assertEqual(workload.reference(request), {"point": ["-2", "4"]})

    def test_pivot_is_hand_calculated(self):
        # [4,2] + [1,-1] + diag(2,3) * ([2,1] - [1,-1]) = [7,7].
        request = {
            "frames": {
                "tool": frame(
                    "world",
                    ("4", "2"),
                    (("2", "0"), ("0", "3")),
                    ("1", "-1"),
                )
            },
            "source": "tool",
            "target": "world",
            "point": ["2", "1"],
        }
        self.assertEqual(workload.reference(request), {"point": ["7", "7"]})

    def test_error_precedence_and_singular_direction(self):
        identity = frame("world", ("0", "0"), (("1", "0"), ("0", "1")))
        singular = frame("world", ("2", "3"), (("1", "2"), ("2", "4")))

        invalid = {
            "frames": {
                "ok": identity,
                "broken": frame("absent", ("0", "0"), (("1", "0"), ("0", "1"))),
            },
            "source": "unknown_source",
            "target": "unknown_target",
            "point": ["0", "0"],
        }
        self.assertEqual(workload.reference(invalid), {"error": "invalid_scene"})

        unknown_before_singular = {
            "frames": {"flat": singular},
            "source": "missing",
            "target": "flat",
            "point": ["0", "0"],
        }
        self.assertEqual(
            workload.reference(unknown_before_singular), {"error": "unknown_frame"}
        )

        singular_target = {
            "frames": {"flat": singular},
            "source": "world",
            "target": "flat",
            "point": ["0", "0"],
        }
        self.assertEqual(workload.reference(singular_target), {"error": "singular"})

        singular_source = {
            "frames": {"flat": singular},
            "source": "flat",
            "target": "world",
            "point": ["1", "-1"],
        }
        self.assertEqual(workload.reference(singular_source), {"point": ["1", "1"]})


class DatasetTests(unittest.TestCase):
    def test_task_shape_determinism_and_split_boundaries(self):
        first = workload.task(20260920, "composition")
        again = workload.task(20260920, "composition")
        other_seed = workload.task(20260921, "composition")
        self.assertEqual(first, again)
        self.assertNotEqual(first["development"], other_seed["development"])
        self.assertEqual(
            set(first), {"id", "contract", "starter", "development", "final"}
        )
        self.assertEqual(len(first["development"]), 12)
        self.assertEqual(len(first["final"]), 48)
        development = {json.dumps(item, sort_keys=True) for item in first["development"]}
        final = {json.dumps(item, sort_keys=True) for item in first["final"]}
        self.assertTrue(development.isdisjoint(final))
        self.assertEqual(len(development), 12)
        self.assertEqual(len(final), 48)

    def test_variants_are_single_fault_starters_and_hard_compiles(self):
        controls = workload.mutants()
        self.assertEqual(tuple(controls), workload.VARIANTS)
        for variant in workload.VARIANTS:
            self.assertEqual(workload.task("variant-seed", variant)["starter"], controls[variant])
            load_evaluate(controls[variant])
        load_evaluate(workload.task("variant-seed", "hard")["starter"])
        with self.assertRaises(ValueError):
            workload.task(1, "not-a-variant")

    def test_generated_results_are_json_and_canonical(self):
        rational = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?$")
        task = workload.task("canonical-seed", "pivot")
        for request in task["development"] + task["final"]:
            result = workload.reference(request)
            json.dumps(result)
            if "point" in result:
                self.assertEqual(len(result["point"]), 2)
                for coordinate in result["point"]:
                    self.assertRegex(coordinate, rational)


class ControlTests(unittest.TestCase):
    def test_known_good_passes_multiple_fresh_seeds(self):
        evaluate = load_evaluate(workload.correct_source())
        for seed in (0, 41, 20260920, "held-out-seed"):
            task = workload.task(seed, "composition")
            for split in ("development", "final"):
                for request in task[split]:
                    self.assertEqual(evaluate(request), workload.reference(request))

    def test_every_mutant_is_rejected_by_each_split(self):
        for seed in (7, 20260920, "mutant-seed"):
            task = workload.task(seed, "composition")
            for name, source in workload.mutants().items():
                evaluate = load_evaluate(source)
                for split in ("development", "final"):
                    failures = [
                        request
                        for request in task[split]
                        if evaluate(request) != workload.reference(request)
                    ]
                    self.assertTrue(failures, f"{name} survived {split} for seed {seed!r}")


if __name__ == "__main__":
    unittest.main()
