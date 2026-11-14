"""Check complete shard accounting and receipt rejection independently of science."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import ci_shards


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
            checkpoint = json.loads((root / "timings.json").read_text())
            self.assertEqual(
                checkpoint["nodes"]["test_failure.py::test_failure"]["reports"]["call"],
                ["failed"],
            )

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

    def test_completed_file_timings_are_retained_before_other_file_finishes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "test_complete.py").write_text("def test_pass(): pass\n")
            (root / "test_wait.py").write_text(
                "import time\ndef test_wait(): time.sleep(120)\n"
            )
            ci_shards._write(
                root / "plan.json",
                {
                    "universe": [
                        "test_complete.py::test_pass",
                        "test_wait.py::test_wait",
                    ],
                    "assignment": {"test_complete.py": 0, "test_wait.py": 0},
                },
            )
            env = os.environ.copy()
            env.update(
                PYTHONPATH=str(ci_shards.INFRA),
                STATADIC_CI_PLUGIN_OUTPUT=str(root / "result.json"),
                STATADIC_CI_PLAN=str(root / "plan.json"),
                STATADIC_CI_SHARD="0",
            )
            env.pop("PYTEST_ADDOPTS", None)
            env.pop("PYTEST_PLUGINS", None)
            with subprocess.Popen(
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
                    "test_complete.py",
                    "test_wait.py",
                ],
                cwd=root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            ) as child:
                try:
                    deadline = time.monotonic() + 15
                    checkpoint = root / "timings.json"
                    observed = {}
                    while time.monotonic() < deadline:
                        if checkpoint.exists():
                            observed = json.loads(checkpoint.read_text())
                            if "test_complete.py::test_pass" in observed["nodes"]:
                                break
                        if child.poll() is not None:
                            self.fail(child.stdout.read().decode())
                        time.sleep(0.05)
                    self.assertIn(
                        "test_complete.py::test_pass", observed.get("nodes", {})
                    )
                    self.assertNotIn("test_wait.py::test_wait", observed["nodes"])
                    self.assertIsNone(child.poll())
                    self.assertFalse((root / "result.json").exists())
                    self.assertTrue((root / "last-started.json").exists())
                    active = json.loads((root / "last-started.json").read_text())[
                        "active"
                    ]
                    self.assertNotIn("test_complete.py::test_pass", active)
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                    child.communicate(timeout=10)

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
