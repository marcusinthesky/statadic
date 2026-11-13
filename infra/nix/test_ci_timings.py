"""Timing hints survive failed checks without becoming successful check evidence."""

import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import ci_shards


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.identity = {
            "source_files": {"source.py": "hash"},
            "revision": "revision",
            "architecture": "x86_64",
            "run_id": "123",
            "run_attempt": "1",
            "python": "3.13.1",
            "interpreter": "/python",
            "jax": "1",
            "jaxlib": "1",
            "xla_flags": "",
            "backend": "cpu",
            "x64": "1",
            "profile": "statadic",
            "flags": ci_shards.FLAGS,
            "shards": ci_shards.N_SHARDS,
            "seed": "seed",
        }
        universe = [
            f"tests/test_{index}.py::test_{case}"
            for index in range(8)
            for case in range(2)
        ]
        self.plan = {
            "schema_version": 1,
            "identity": self.identity,
            "universe": universe,
            "assignment": ci_shards.assign(universe, {}),
        }
        path = self.root / "plan/plan.json"
        ci_shards._write(path, self.plan)
        self.digest = hashlib.sha256(path.read_bytes()).hexdigest()

    def checkpoint(self, shard=0):
        return {
            "schema_version": 1,
            "kind": "test-timings",
            "plan_sha256": self.digest,
            "shard": shard,
            "execution": {"workers": 2, "distribution": "load"},
            "nodes": {
                node: {
                    "reports": {
                        "setup": ["passed"],
                        "call": ["passed"],
                        "teardown": ["passed"],
                    },
                    "durations": {"setup": [0.1], "call": [1.0], "teardown": [0.1]},
                }
                for node in self.plan["universe"]
                if self.plan["assignment"][node.split("::", 1)[0]] == shard
            },
        }

    def save(self, record):
        ci_shards._write(self.root / "shard-0/timings.json", record)

    def test_complete_failed_file_survives_and_partial_or_missing_files_keep_prior(
        self,
    ):
        record = self.checkpoint()
        nodes = sorted(record["nodes"])
        complete_file = nodes[0].split("::", 1)[0]
        partial_file = nodes[-1].split("::", 1)[0]
        record["nodes"][nodes[0]]["reports"]["call"] = ["failed"]
        del record["nodes"][nodes[-1]]
        self.save(record)
        previous = self.root / "previous.json"
        missing_file = next(
            name for name, shard in self.plan["assignment"].items() if shard == 1
        )
        ci_shards._write(
            previous, {complete_file: 90.0, partial_file: 80.0, missing_file: 70.0}
        )
        durations = ci_shards.collect_timings(self.root, self.identity, previous)
        self.assertAlmostEqual(durations[complete_file], 2.4)
        self.assertEqual(durations[partial_file], 80.0)
        self.assertEqual(durations[missing_file], 70.0)
        self.assertFalse((self.root / "receipt.json").exists())
        self.assertFalse((self.root / "component").exists())

    def test_invalid_checkpoint_is_not_allowed_to_replace_prior_timings(self):
        original = self.checkpoint()
        first = next(iter(original["nodes"]))
        filename = first.split("::", 1)[0]
        previous = self.root / "previous.json"
        ci_shards._write(previous, {filename: 20.0})
        defects = (
            "negative",
            "nonfinite",
            "boolean",
            "huge",
            "duplicate",
            "incomplete",
            "contradictory",
            "unknown_node",
            "digest",
            "shard",
            "execution",
        )
        for defect in defects:
            with self.subTest(defect=defect):
                record = copy.deepcopy(original)
                node = record["nodes"][first]
                if defect in ("negative", "nonfinite", "boolean", "huge"):
                    node["durations"]["call"] = [
                        {
                            "negative": -1.0,
                            "nonfinite": float("nan"),
                            "boolean": True,
                            "huge": 10**400,
                        }[defect]
                    ]
                elif defect == "duplicate":
                    node["reports"]["call"].append("passed")
                elif defect == "incomplete":
                    del node["reports"]["teardown"]
                    del node["durations"]["teardown"]
                elif defect == "contradictory":
                    node["reports"]["setup"] = ["skipped"]
                elif defect == "unknown_node":
                    record["nodes"]["tests/unknown.py::test_case"] = node
                elif defect == "digest":
                    record["plan_sha256"] = "other"
                elif defect == "shard":
                    record["shard"] = 1
                else:
                    record["execution"]["distribution"] = "loadfile"
                self.save(record)
                with redirect_stderr(io.StringIO()):
                    observed = ci_shards.collect_timings(
                        self.root, self.identity, previous
                    )
                self.assertEqual(observed, {filename: 20.0})

    def test_provenance_rejects_other_source_run_attempt_architecture_or_flags(self):
        for field, value in (
            ("source_files", {}),
            ("revision", "other"),
            ("architecture", "aarch64"),
            ("run_id", "other"),
            ("run_attempt", "2"),
            ("flags", []),
            ("backend", "gpu"),
            ("x64", "0"),
            ("seed", "other"),
        ):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "identity"),
            ):
                ci_shards.collect_timings(self.root, self.identity | {field: value})

    def test_prior_values_are_sanitized_and_truncated_checkpoint_is_ignored(self):
        files = sorted(self.plan["assignment"])
        previous = self.root / "previous.json"
        ci_shards._write(
            previous,
            {
                files[0]: 2.5,
                files[1]: -1,
                files[2]: float("inf"),
                files[3]: True,
                "removed.py": 3.0,
            },
        )
        checkpoint = self.root / "shard-0/timings.json"
        checkpoint.parent.mkdir()
        checkpoint.write_text('{"nodes":', encoding="utf-8")
        with redirect_stderr(io.StringIO()):
            observed = ci_shards.collect_timings(self.root, self.identity, previous)
        self.assertEqual(observed, {files[0]: 2.5})

    def test_corrupt_or_wrong_type_duration_cache_falls_back_to_unmeasured_plan(self):
        path = self.root / "durations.json"
        for content in (
            '{"broken":',
            "[]",
            '{"test.py": 1e999}',
            json.dumps({"test.py": 10**400}),
        ):
            with self.subTest(content=content):
                path.write_text(content, encoding="utf-8")
                with redirect_stderr(io.StringIO()):
                    durations = ci_shards._duration_cache(path)
                self.assertEqual(durations, {})
                self.assertEqual(
                    ci_shards.assign(self.plan["universe"], durations),
                    self.plan["assignment"],
                )

    def test_cli_needs_no_scientific_runtime_and_can_update_previous_path(self):
        self.save(self.checkpoint())
        output = self.root / "durations.json"
        ci_shards._write(output, {})
        with (
            mock.patch.object(
                ci_shards, "_identity", side_effect=AssertionError("JAX")
            ),
            mock.patch.object(
                ci_shards, "_timing_identity", return_value=self.identity
            ),
            mock.patch.object(
                sys,
                "argv",
                [
                    "ci_shards.py",
                    "timings",
                    "--inputs",
                    str(self.root),
                    "--output",
                    str(output),
                    "--previous",
                    str(output),
                ],
            ),
        ):
            self.assertEqual(ci_shards.main(), 0)
        self.assertEqual(len(json.loads(output.read_text())), 2)
        self.assertFalse((self.root / "receipt.json").exists())

    def test_completed_setup_skip_or_failure_is_valid_timing_evidence(self):
        for outcome in ("skipped", "failed"):
            with self.subTest(outcome=outcome):
                record = self.checkpoint()
                first = next(iter(record["nodes"]))
                record["nodes"][first] = {
                    "reports": {"setup": [outcome], "teardown": ["passed"]},
                    "durations": {"setup": [0.1], "teardown": [0.1]},
                }
                self.save(record)
                durations = ci_shards.collect_timings(self.root, self.identity)
                self.assertAlmostEqual(durations[first.split("::", 1)[0]], 1.4)

    def test_diagnostic_replay_preserves_status_and_cannot_create_check_receipt(self):
        output = self.root / "diagnostic"
        with (
            mock.patch.object(ci_shards, "_identity", return_value=self.identity),
            mock.patch.object(ci_shards, "_pytest", return_value=1) as execute,
            mock.patch.object(
                sys,
                "argv",
                [
                    "ci_shards.py",
                    "replay",
                    "--plan",
                    str(self.root / "plan"),
                    "--shard",
                    "0",
                    "--output",
                    str(output),
                    "--workers",
                    "1",
                    "--dist",
                    "loadfile",
                ],
            ),
        ):
            self.assertEqual(ci_shards.main(), 1)
        self.assertEqual(
            execute.call_args.args[0], ["-v", "-n", "1", "--dist", "loadfile"]
        )
        self.assertFalse((output / "shard-0.json").exists())
        self.assertFalse((output / "receipt.json").exists())


if __name__ == "__main__":
    unittest.main()
