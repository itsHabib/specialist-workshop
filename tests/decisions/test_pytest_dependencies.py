import tempfile
import unittest
from pathlib import Path

from experiments.steering.pytest_dependencies import rank_tests


class DependencyOrderTests(unittest.TestCase):
    def test_source_edges_and_native_ties(self):
        files = {
            "src/lib/__init__.py": "from .leaf import Leaf as Leaf\n",
            "src/lib/leaf.py": "class Leaf: pass\n",
            "src/lib/helper.py": "from .leaf import Leaf\ndef run(): return Leaf()\n",
            "tests/test_host.py": (
                "import lib\nimport lib.helper as h\nfrom lib.leaf import Leaf as L\n"
                "def test_bare(): assert lib\n"
                "def test_helper(): assert h.run()\n"
                "def test_local(lib): assert lib.Leaf\n"
                "class TestNested:\n    def test_public(self): assert lib.Leaf()\n"
                "def test_alias(): assert L()\n"),
            "tests/test_other.py": "def test_edited(): pass\n",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, data in files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(data)
            base = "tests/test_host.py::"
            nodes = [base + n for n in ("test_bare", "test_helper", "test_local",
                                        "TestNested::test_public[foo::x]", "test_alias")]
            nodes.append("tests/test_other.py::test_edited")
            nodes.append(base + "test_generated")
            report = rank_tests(root, ["tests/test_other.py", "src/lib/leaf.py"], nodes)
            self.assertEqual(report["order"], [nodes[i] for i in (5, 3, 4, 1, 0, 2, 6)])
            self.assertEqual(report["tiers"], {"changed_test": 1, "direct_reference": 2,
                                               "one_import_edge": 1, "native": 3})
            self.assertEqual(report["unresolved_test_functions"], [nodes[-1]])
            with self.assertRaisesRegex(ValueError, "distinct"):
                rank_tests(root, [], nodes * 2)

    def test_nested_scopes_preserve_outer_imports_and_hide_inner_bindings(self):
        bodies = [
            "def helper(Leaf): return Leaf()\n    assert Leaf()",
            "[Leaf for Leaf in range(2)]\n    assert Leaf()",
            "def helper(Leaf): return Leaf()",
            "assert [Leaf for Leaf in range(2)]",
            "assert [x for x in Leaf()]",
            "def helper(): return Leaf()\n    assert helper()",
            "class Nested:\n        from leaf import Leaf as Item\n        value = Item()",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src/leaf.py").write_text("class Leaf: pass")
            (root / "test_host.py").write_text("from leaf import Leaf\n" + "".join(
                f"def test_{i}():\n    {body}\n" for i, body in enumerate(bodies)))
            nodes = [f"test_host.py::test_{i}" for i in range(len(bodies))]
            result = rank_tests(root, ["src/leaf.py"], nodes)
            self.assertEqual(result["order"], [nodes[i] for i in (0, 1, 4, 5, 6, 2, 3)])
            self.assertEqual(result["tiers"]["direct_reference"], 5)

    def test_analysis_reads_only_source_in_the_requested_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            src = root / "src"
            src.mkdir(parents=True)
            external = Path(tmp) / "other.py"
            external.write_text("class Example: pass\n")
            (src / "leaf.py").symlink_to(external)
            for directory in ("..", "src"):
                with self.subTest(directory=directory), self.assertRaisesRegex(ValueError, "outside checkout"):
                    rank_tests(root, ["src/leaf.py"], ["tests/test_example.py::test_a"], source_dir=directory)
            for changed, roster in [("src/leaf.py", []), ([None], []), ([], "test_example.py"),
                                    ([], None)]:
                with self.subTest(changed=changed, roster=roster), self.assertRaises(ValueError):
                    rank_tests(root, changed, roster)
