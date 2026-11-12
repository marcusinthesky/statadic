"""Check complete shard accounting and receipt rejection independently of science."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import ci_shards


def make_receipt(root: Path) -> dict:
    identity = {
        "source_files": {"a": "hash"},
        "revision": "abc",
        "architecture": "x86_64",
        "backend": "cpu",
        "x64": "1",
        "profile": "statadic",
        "flags": ci_shards.FLAGS,
        "shards": 4,
        "seed": "abc",
        "python": "3.13.1",
        "interpreter": "/python",
        "jax": "1",
        "jaxlib": "1",
        "xla_flags": "",
        "run_id": "3",
        "run_attempt": "1",
    }
    universe = [f"tests/test_{i}.py::test_case" for i in range(8)]
    assignment = ci_shards.assign(universe, {})
    plan = {
        "schema_version": 1,
        "identity": identity,
        "universe": universe,
        "assignment": assignment,
    }
    ci_shards._write(
        root / "component.json",
        {"schema_version": 1, "identity": identity, "exitstatus": 0},
    )
    ci_shards._write(root / "collection.json", {"universe": universe, "exitstatus": 0})
    ci_shards._write(root / "durations.json", {})
    (root / "component.log").write_text("passed\n")
    for shard in range(4):
        selected = [
            node for node in universe if assignment[node.split("::", 1)[0]] == shard
        ]
        reports = {
            node: {"setup": ["passed"], "call": ["passed"], "teardown": ["passed"]}
            for node in selected
        }
        record = {
            "schema_version": 1,
            "identity": identity,
            "exitstatus": 0,
            "universe": universe,
            "selected": selected,
            "reports": reports,
            "durations": {},
        }
        ci_shards._write(root / f"shard-{shard}.json", record)
        ci_shards._write(
            root / f"shard-{shard}-pytest.json",
            {
                "exitstatus": 0,
                "reports": reports,
                "workers": [{"universe": universe, "selected": selected}] * 2,
            },
        )
        (root / f"shard-{shard}.log").write_text("passed\n")
        (root / f"shard-{shard}.xml").write_text(
            "<testsuites>" + "<testcase/>" * len(selected) + "</testsuites>"
        )
    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.iterdir()
        if path.is_file()
    }
    ci_shards._write(root / "receipt.json", plan | {"files": files})
    return identity


class ShardTests(unittest.TestCase):
    def test_complete_synthetic_plan_shards_and_aggregate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = root / "package"
            package.mkdir()
            for index in range(8):
                (package / f"test_{index}.py").write_text(
                    "def test_case():\n    assert True\n", encoding="utf-8"
                )
            inputs = root / "inputs"
            plan = inputs / "plan"
            component = inputs / "component"
            component.mkdir(parents=True)
            (component / "component.log").write_text("non-test checks passed\n")

            def invoke(*args):
                with mock.patch.object(sys, "argv", ["ci_shards.py", *map(str, args)]):
                    self.assertEqual(ci_shards.main(), 0)

            with mock.patch.object(ci_shards, "PACKAGE", package):
                invoke("plan", "--output", plan)
                invoke("component", "--output", component)
                for index in range(4):
                    invoke(
                        "shard",
                        "--plan",
                        plan,
                        "--shard",
                        index,
                        "--output",
                        inputs / f"shard-{index}",
                    )
                invoke("aggregate", "--inputs", inputs, "--output", root / "receipt")
            receipt = ci_shards._load(root / "receipt/receipt.json")
            self.assertEqual(len(receipt["universe"]), 8)

    def test_xdist_streams_failure_before_summary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "test_failure.py").write_text(
                'def test_failure():\n    assert False, "early diagnostic"\n',
                encoding="utf-8",
            )
            ci_shards._write(root / "plan.json", {"assignment": {"test_failure.py": 0}})
            env = os.environ.copy()
            env.update(
                PYTHONPATH=str(ci_shards.INFRA),
                STATADIC_CI_PLUGIN_OUTPUT=str(root / "result.json"),
                STATADIC_CI_SHARD="0",
                STATADIC_CI_PLAN=str(root / "plan.json"),
            )
            env.pop("PYTEST_ADDOPTS", None)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-p",
                    "ci_shard_plugin",
                    "-p",
                    "no:monitor",
                    "-p",
                    "no:tach",
                    "-n",
                    "2",
                    "-q",
                    str(root / "test_failure.py"),
                ],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("CI failure: test_failure.py::test_failure", result.stdout)
            self.assertLess(
                result.stdout.index("CI failure:"), result.stdout.index(" FAILURES ")
            )
            self.assertIn("early diagnostic", result.stdout)

    def test_xdist_reports_exact_selected_nodes_and_setup_skips(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "test_sample.py").write_text(
                "import pytest\n"
                "def test_pass(): pass\n"
                '@pytest.mark.skip(reason="example")\n'
                "def test_skip(): pass\n",
                encoding="utf-8",
            )
            result_file = root / "result.json"
            plan_file = root / "plan.json"
            ci_shards._write(plan_file, {"assignment": {"test_sample.py": 0}})
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ci_shards.INFRA)
            env["STATADIC_CI_PLUGIN_OUTPUT"] = str(result_file)
            env["STATADIC_CI_SHARD"] = "0"
            env["STATADIC_CI_PLAN"] = str(plan_file)
            env.pop("PYTEST_ADDOPTS", None)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-p",
                    "ci_shard_plugin",
                    "-p",
                    "no:monitor",
                    "-p",
                    "no:tach",
                    "-n",
                    "2",
                    "-q",
                    str(root / "test_sample.py"),
                ],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            observed = json.loads(result_file.read_text())
            self.assertEqual(len(observed["workers"]), 2)
            self.assertEqual(len(observed["reports"]), 2)
            self.assertEqual(
                {v["setup"][0] for v in observed["reports"].values()},
                {"passed", "skipped"},
            )
            self.assertTrue(all("teardown" in v for v in observed["reports"].values()))

    def test_receipt_requires_exact_complete_successful_shards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            identity = make_receipt(root)
            kwargs = {
                "source_files": identity["source_files"],
                "revision": "abc",
                "architecture": "x86_64",
                "run_id": "3",
                "run_attempt": "1",
            }
            ci_shards.verify_receipt(root, **kwargs)
            (root / "shard-0.log").write_text("tampered")
            with self.assertRaisesRegex(ValueError, "file changed"):
                ci_shards.verify_receipt(root, **kwargs)
            with self.assertRaisesRegex(ValueError, "identity"):
                ci_shards.verify_receipt(root, **(kwargs | {"run_id": "4"}))

    def test_assignment_is_deterministic_whole_file_and_balanced(self):
        nodes = [
            f"test_{index}.py::test_{item}"
            for index in range(8)
            for item in range(index + 1)
        ]
        assignment = ci_shards.assign(nodes, {})
        self.assertEqual(assignment, ci_shards.assign(list(reversed(nodes)), {}))
        loads = [
            sum(assignment[node.split("::", 1)[0]] == index for node in nodes)
            for index in range(4)
        ]
        self.assertLessEqual(max(loads) - min(loads), 2)


if __name__ == "__main__":
    unittest.main()
