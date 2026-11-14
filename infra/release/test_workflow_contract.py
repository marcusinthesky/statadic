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

    def test_obsolete_validation_is_cancelled_but_publication_is_preserved(self):
        workflows = ROOT / ".github/workflows"
        release = yaml.safe_load((workflows / "release.yml").read_text())
        pages = yaml.safe_load((workflows / "pages.yml").read_text())
        preflight = yaml.safe_load((workflows / "release-validation.yml").read_text())
        pr_only = "${{ github.event_name == 'pull_request' }}"
        self.assertIs(self.ci["concurrency"]["cancel-in-progress"], True)
        self.assertIs(preflight["concurrency"]["cancel-in-progress"], True)
        self.assertEqual(release["concurrency"]["cancel-in-progress"], pr_only)
        self.assertEqual(pages["concurrency"]["cancel-in-progress"], pr_only)
        self.assertIs(
            pages["jobs"]["deploy"]["concurrency"]["cancel-in-progress"], False
        )
        # The caller owns cancellation; a matching called-workflow group can
        # accidentally cancel the caller when package validation starts.
        self.assertNotIn("concurrency", self.package)
        groups = [
            workflow["concurrency"]["group"]
            for workflow in (self.ci, release, pages, preflight)
        ]
        self.assertEqual(len(groups), len(set(groups)))

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
        changed = copy.deepcopy(self.ci)
        changed["jobs"]["check-nontest-arm"]["needs"] = "package"
        with self.assertRaises(ValueError):
            validate(changed, self.package)

    def test_reference_status_cannot_bypass_installed_comparisons(self):
        changed = copy.deepcopy(self.ci)
        changed["jobs"]["reference-validation"]["needs"] = "regressions"
        with self.assertRaises(ValueError):
            validate(changed, self.package)
        changed = copy.deepcopy(self.ci)
        changed["jobs"]["reference-validation"]["steps"] = []
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
