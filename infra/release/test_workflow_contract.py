"""Exercise the dependency graph that makes package extraction fail closed."""

import copy
import unittest
from pathlib import Path

import yaml
from workflow_contract import validate

ROOT = Path(__file__).resolve().parents[2]


class WorkflowContractTests(unittest.TestCase):
    def setUp(self):
        self.ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
        self.package = yaml.safe_load(
            (ROOT / ".github/workflows/package-validation.yml").read_text()
        )

    def test_complete_package_and_workspace_gates(self):
        validate(self.ci, self.package)

    def test_every_package_dependency_is_required(self):
        for name in (
            "regressions",
            "plan",
            "shard",
            "quality",
            "installed",
            "aggregate",
            "candidate",
        ):
            with self.subTest(job=name):
                changed = copy.deepcopy(self.package)
                changed["jobs"][name]["needs"] = []
                with self.assertRaises(ValueError):
                    validate(self.ci, changed)

    def test_workspace_failure_cannot_be_ignored(self):
        changed = copy.deepcopy(self.ci)
        changed["jobs"]["check"]["needs"] = ["package"]
        with self.assertRaises(ValueError):
            validate(changed, self.package)

    def test_architecture_or_shard_cannot_be_removed(self):
        changed = copy.deepcopy(self.package)
        changed["jobs"]["shard"]["strategy"]["matrix"]["shard"] = [0, 1, 2]
        with self.assertRaises(ValueError):
            validate(self.ci, changed)
        changed = copy.deepcopy(self.package)
        changed["jobs"]["installed"]["strategy"]["matrix"]["runner"] = ["ubuntu-24.04"]
        with self.assertRaises(ValueError):
            validate(self.ci, changed)


if __name__ == "__main__":
    unittest.main()
