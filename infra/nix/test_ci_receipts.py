"""Adversarial tests for retained CI execution receipts."""

import hashlib
import json
import platform
import tempfile
import unittest
from pathlib import Path

import ci_shards


def make_receipt(directory, identity=None):
    """Build a complete synthetic receipt with verifiable per-shard records."""
    directory.mkdir(parents=True, exist_ok=True)
    identity = identity or {
        "source_files": {"example.py": "abc"},
        "revision": "revision",
        "architecture": platform.machine(),
        "run_id": "123",
        "run_attempt": "2",
        "python": "3.13",
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
    universe = [f"tests/example_{index}.py::test_case" for index in range(4)]
    assignment = ci_shards.assign(universe, {})
    ci_shards._write(
        directory / "component.json",
        {"schema_version": 1, "identity": identity, "exitstatus": 0},
    )
    ci_shards._write(
        directory / "collection.json", {"exitstatus": 0, "universe": universe}
    )
    ci_shards._write(directory / "durations.json", {})
    (directory / "component.log").write_text("completed\n")
    for index in range(ci_shards.N_SHARDS):
        selected = [
            node for node in universe if assignment[node.split("::", 1)[0]] == index
        ]
        reports = {
            node: {"setup": ["passed"], "call": ["passed"], "teardown": ["passed"]}
            for node in selected
        }
        ci_shards._write(
            directory / f"shard-{index}.json",
            {
                "schema_version": 1,
                "identity": identity,
                "exitstatus": 0,
                "universe": universe,
                "selected": selected,
                "reports": reports,
            },
        )
        ci_shards._write(
            directory / f"shard-{index}-pytest.json",
            {
                "exitstatus": 0,
                "reports": reports,
                "workers": [{"universe": universe, "selected": selected}] * 2,
            },
        )
        (directory / f"shard-{index}.log").write_text("completed\n")
        (directory / f"shard-{index}.xml").write_text(
            "<testsuites>" + "<testcase/>" * len(selected) + "</testsuites>"
        )
    inventory = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir()
        if path.name != "receipt.json"
    }
    ci_shards._write(
        directory / "receipt.json",
        {
            "schema_version": 1,
            "identity": identity,
            "universe": universe,
            "assignment": assignment,
            "files": inventory,
        },
    )
    return identity


def verify(directory, identity):
    return ci_shards.verify_receipt(
        directory,
        source_files=identity["source_files"],
        revision=identity["revision"],
        architecture=identity["architecture"],
        run_id=identity["run_id"],
        run_attempt=identity["run_attempt"],
    )


def mutate(directory, filename, change):
    path = directory / filename
    value = json.loads(path.read_text())
    change(value)
    ci_shards._write(path, value)
    receipt = ci_shards._load(directory / "receipt.json")
    receipt["files"][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    ci_shards._write(directory / "receipt.json", receipt)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.identity = make_receipt(self.root)

    def test_complete_receipt(self):
        verify(self.root, self.identity)

    def test_recorded_setup_and_call_skips_are_complete(self):
        record = ci_shards._load(self.root / "shard-0.json")
        node = record["selected"][0]
        for phases in (
            {"setup": ["skipped"], "teardown": ["passed"]},
            {"setup": ["passed"], "call": ["skipped"], "teardown": ["passed"]},
        ):
            record["reports"][node] = phases
            mutate(self.root, "shard-0.json", lambda value: value.update(record))
            mutate(
                self.root,
                "shard-0-pytest.json",
                lambda value: value.update(reports=record["reports"]),
            )
            verify(self.root, self.identity)

    def test_bad_identity_and_schema(self):
        for field, value in (
            ("schema_version", 2),
            ("revision", "old"),
            ("architecture", "arm"),
            ("run_id", "other"),
            ("run_attempt", "3"),
            ("source_files", {}),
        ):
            with self.subTest(field=field):
                make_receipt(self.root)
                receipt = ci_shards._load(self.root / "receipt.json")
                if field == "schema_version":
                    receipt[field] = value
                else:
                    receipt["identity"][field] = value
                ci_shards._write(self.root / "receipt.json", receipt)
                with self.assertRaises((ValueError, KeyError)):
                    verify(self.root, self.identity)

    def test_invalid_reports_and_selections(self):
        for defect in (
            "missing",
            "extra",
            "duplicate",
            "failed",
            "crashed",
            "bad_outcome",
            "duplicate_phase",
            "bad_phase",
            "contradictory_skip",
        ):
            with self.subTest(defect=defect):
                make_receipt(self.root)
                path = self.root / "shard-0.json"
                record = ci_shards._load(path)
                node = record["selected"][0]
                if defect == "missing":
                    del record["reports"][node]
                elif defect == "extra":
                    record["reports"]["tests/extra.py::test_case"] = record["reports"][
                        node
                    ]
                elif defect == "duplicate":
                    record["selected"].append(node)
                elif defect == "failed":
                    record["reports"][node]["call"] = ["failed"]
                elif defect == "crashed":
                    del record["reports"][node]["teardown"]
                elif defect == "bad_outcome":
                    record["reports"][node]["call"] = ["unknown"]
                elif defect == "duplicate_phase":
                    record["reports"][node]["call"] = ["passed", "passed"]
                elif defect == "bad_phase":
                    record["reports"][node]["other"] = ["passed"]
                else:
                    record["reports"][node]["setup"] = ["skipped"]
                mutate(
                    self.root,
                    "shard-0.json",
                    lambda value, record=record: value.update(record),
                )
                if defect not in ("missing", "extra", "duplicate"):
                    mutate(
                        self.root,
                        "shard-0-pytest.json",
                        lambda value, record=record: value.update(
                            reports=record["reports"]
                        ),
                    )
                with self.assertRaises((ValueError, KeyError)):
                    verify(self.root, self.identity)

    def test_retained_junit_tampering_and_missing_file(self):
        path = self.root / "shard-0.xml"
        path.write_text("<testsuites><tampered/></testsuites>")
        with self.assertRaises(ValueError):
            verify(self.root, self.identity)
        path.unlink()
        with self.assertRaises((ValueError, OSError)):
            verify(self.root, self.identity)


if __name__ == "__main__":
    unittest.main()
