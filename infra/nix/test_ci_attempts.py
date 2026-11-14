"""Reruns retain provenance without reusing incompatible execution inputs."""

import json
import tempfile
import unittest
from pathlib import Path

from ci_shards import same_execution, verify_receipt
from test_ci_receipts import make_receipt, mutate


class AttemptTests(unittest.TestCase):
    def setUp(self):
        self.origin = {
            "run_id": "123",
            "run_attempt": "1",
            "revision": "commit",
            "seed": "seed",
            "architecture": "aarch64",
            "source_files": {"file.py": "hash"},
            "jax": "version",
        }

    def test_legacy_identity_remains_strict(self):
        self.assertFalse(
            same_execution(self.origin, self.origin | {"run_attempt": "2"})
        )

    def test_prior_attempt_is_accepted_for_same_execution(self):
        self.assertTrue(
            same_execution(
                self.origin, self.origin | {"run_attempt": "2"}, partial=True
            )
        )

    def test_future_attempt_and_changed_inputs_are_rejected(self):
        for key, value in (
            ("run_attempt", "0"),
            ("run_id", "456"),
            ("revision", "other"),
            ("seed", "different"),
            ("architecture", "x86_64"),
            ("source_files", {"file.py": "changed"}),
            ("jax", "other"),
        ):
            with self.subTest(key=key):
                self.assertFalse(
                    same_execution(
                        self.origin, self.origin | {key: value}, partial=True
                    )
                )

    def test_complete_receipt_can_mix_compatible_producer_attempts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            identity = make_receipt(directory)
            receipt = json.loads((directory / "receipt.json").read_text())
            receipt["schema_version"] = 2
            receipt["identity"]["run_attempt"] = "3"
            (directory / "receipt.json").write_text(json.dumps(receipt))
            options = {
                "source_files": identity["source_files"],
                "revision": identity["revision"],
                "architecture": identity["architecture"],
                "run_id": "123",
                "run_attempt": "3",
            }
            verify_receipt(directory, **options)
            mutate(
                directory,
                "shard-0.json",
                lambda record: record["identity"].update(run_attempt="4"),
            )
            with self.assertRaises(ValueError):
                verify_receipt(directory, **options)


if __name__ == "__main__":
    unittest.main()
