"""Reject mismatched artifacts and untrusted promotion producers."""

import io
import json
import os
import runpy
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import package_candidate as candidate


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        artifacts = self.directory / "artifacts"
        artifacts.mkdir()
        (artifacts / "statadic-1.0-py3-none-any.whl").write_bytes(b"wheel")
        (artifacts / "statadic-1.0.tar.gz").write_bytes(b"source")
        self.record = {
            "schema_version": 1,
            "kind": "statadic-package-candidate",
            "revision": "commit",
            "version": "1.0",
            "scope": "package",
            "status": "passed",
            "run_id": "123",
            "architectures": list(candidate.ARCHITECTURES),
            "artifacts": candidate.artifacts(artifacts),
            "files": {
                str(p.relative_to(self.directory)): candidate.digest(p)
                for p in artifacts.iterdir()
            },
            "run_attempt": "1",
        }
        build = patch.object(candidate, "verify_build", return_value={})
        evidence = patch.object(candidate, "verify_evidence")
        build.start()
        evidence.start()
        self.addCleanup(build.stop)
        self.addCleanup(evidence.stop)
        self.save()

    def save(self):
        candidate.write(self.directory / "candidate.json", self.record)

    def verify(self):
        return candidate.verify_candidate(
            self.directory, sha="commit", package_version="1.0", run_id="123"
        )

    def test_exact_candidate_is_accepted(self):
        self.assertEqual(self.verify(), self.record)

    def test_wrong_identity_and_scope_are_rejected(self):
        for key, value in (
            ("revision", "other"),
            ("version", "2.0"),
            ("run_id", "other"),
            ("scope", "docs"),
            ("status", "failed"),
            ("architectures", ["x86_64"]),
        ):
            with self.subTest(key=key):
                original = self.record[key]
                self.record[key] = value
                self.save()
                with self.assertRaises(ValueError):
                    self.verify()
                self.record[key] = original
        self.save()

    def test_changed_missing_or_extra_artifact_is_rejected(self):
        wheel = next((self.directory / "artifacts").glob("*.whl"))
        wheel.write_bytes(b"modified")
        with self.assertRaises(ValueError):
            self.verify()
        wheel.unlink()
        with self.assertRaises(ValueError):
            self.verify()

    def test_uninventoried_file_is_rejected(self):
        (self.directory / "extra").write_text("extra")
        with self.assertRaises(ValueError):
            self.verify()

    def test_reference_lane_cannot_skip_or_be_empty(self):
        path = self.directory / "lane.xml"
        for content in (
            "<testsuite/>",
            "<testsuite><testcase><skipped/></testcase></testsuite>",
            "<testsuite><testcase><failure/></testcase></testsuite>",
        ):
            path.write_text(content)
            with self.assertRaises(ValueError):
                candidate.verify_junit(path)
        path.write_text("<testsuite><testcase/></testsuite>")
        candidate.verify_junit(path)

    def test_only_trusted_exact_commit_producers_are_eligible(self):
        with patch.dict(os.environ, GITHUB_REPOSITORY="owner/repo"):
            record = {
                "head_sha": "commit",
                "head_branch": "main",
                "event": "push",
                "path": ".github/workflows/ci.yml",
                "repository": {"full_name": "owner/repo"},
            }
            self.assertTrue(candidate.eligible_run(record, "commit"))
            for key, value in (
                ("event", "pull_request"),
                ("head_sha", "old"),
                ("path", ".github/workflows/pages.yml"),
                ("head_branch", "feature"),
                ("repository", {"full_name": "fork/repo"}),
            ):
                self.assertFalse(
                    candidate.eligible_run(record | {key: value}, "commit")
                )

    def test_missing_candidate_fails_without_build_or_test(self):
        with (
            patch.object(candidate, "revision", return_value="commit"),
            patch.object(
                candidate,
                "api",
                side_effect=[{"status": "ahead"}, {"workflow_runs": []}],
            ),
            patch.object(candidate, "run") as run,
        ):
            with self.assertRaisesRegex(ValueError, "Dispatch CI"):
                candidate.download(self.directory / "download", None)
            run.assert_not_called()

    def test_existing_pypi_files_must_match_before_retry(self):
        wheel_name = next(
            name for name in self.record["artifacts"] if name.endswith(".whl")
        )
        remote = {
            "urls": [
                {
                    "filename": wheel_name,
                    "digests": {"sha256": self.record["artifacts"][wheel_name]},
                }
            ]
        }
        with (
            patch.object(candidate, "version", return_value="1.0"),
            patch.dict(os.environ, GITHUB_OUTPUT=str(self.directory / "outputs")),
            patch.object(
                candidate.urllib.request,
                "urlopen",
                return_value=io.BytesIO(json.dumps(remote).encode()),
            ),
        ):
            self.assertFalse(
                candidate.prepare_publication(
                    self.directory / "artifacts", self.directory / "pending"
                )
            )
            self.assertEqual(
                [p.name for p in (self.directory / "pending").iterdir()],
                ["statadic-1.0.tar.gz"],
            )
        remote["urls"][0]["digests"]["sha256"] = "different"
        with (
            patch.object(candidate, "version", return_value="1.0"),
            patch.object(
                candidate.urllib.request,
                "urlopen",
                return_value=io.BytesIO(json.dumps(remote).encode()),
            ),
            self.assertRaisesRegex(ValueError, "different distribution"),
        ):
            candidate.prepare_publication(
                self.directory / "artifacts", self.directory / "rejected"
            )


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        with patch.object(sys, "path", [str(candidate.ROOT / "infra/nix"), *sys.path]):
            builder = runpy.run_path(
                str(candidate.ROOT / "infra/nix/test_ci_receipts.py")
            )
        self.built = {
            "schema_version": 1,
            "revision": "revision",
            "version": "1.0",
            "lock_sha256": "lock",
            "artifacts": {"wheel.whl": "hash"},
        }
        identity = builder["make_receipt"](self.root / "template")
        for architecture in candidate.ARCHITECTURES:
            receipt = self.root / f"package-check-{architecture}"
            builder["make_receipt"](receipt, identity | {"architecture": architecture})
            builder["mutate"](
                receipt, "component.json", lambda record: record.update(scope="package")
            )
            installed = self.root / f"package-installed-{architecture}"
            candidate.write(
                installed / "qualification.json",
                self.built
                | {
                    "architecture": architecture,
                    "status": "passed",
                    "run_id": "123",
                    "run_attempt": "2",
                },
            )
            for lane in ("external", "precision-float32", "precision-float64"):
                (installed / f"{lane}.xml").write_text(
                    "<testsuite><testcase/></testsuite>"
                )
            (installed / "distribution").mkdir()
            (installed / "distribution/distribution.log").write_text(
                "Both wheel routes and standalone sdist development passed."
            )
        revision = patch.object(candidate, "revision", return_value="revision")
        verification = patch.object(
            candidate.runpy,
            "run_path",
            return_value={
                "source_snapshot": lambda root: identity["source_files"],
                "verify_receipt": builder["ci_shards"].verify_receipt,
            },
        )
        revision.start()
        verification.start()
        self.addCleanup(revision.stop)
        self.addCleanup(verification.stop)

    def test_complete_package_evidence_is_accepted(self):
        candidate.verify_evidence(self.root, self.built, "123", "2")

    def test_missing_architecture_report_is_rejected(self):
        (self.root / "package-installed-aarch64/qualification.json").unlink()
        with self.assertRaises(FileNotFoundError):
            candidate.verify_evidence(self.root, self.built, "123", "2")

    def test_modified_shard_payload_is_rejected(self):
        (self.root / "package-check-x86_64/shard-0.json").write_text("{}")
        with self.assertRaises(ValueError):
            candidate.verify_evidence(self.root, self.built, "123", "2")

    def test_incomplete_distribution_is_rejected(self):
        (
            self.root / "package-installed-aarch64/distribution/distribution.log"
        ).write_text("started")
        with self.assertRaises(ValueError):
            candidate.verify_evidence(self.root, self.built, "123", "2")


class PayloadTests(unittest.TestCase):
    def test_missing_extra_or_changed_wheel_modules_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "src/statadic"
            source.mkdir(parents=True)
            (source / "example.py").write_text("VALUE = 1\n")
            wheels = root / "wheels"
            wheels.mkdir()
            wheel = wheels / "statadic-1.0-py3-none-any.whl"
            with patch.object(candidate, "PACKAGE", root):
                for entries in (
                    {"statadic/example.py": "VALUE = 1\n"},
                    {},
                    {"statadic/example.py": "VALUE = 2\n"},
                    {"statadic/example.py": "VALUE = 1\n", "statadic/extra.py": ""},
                ):
                    with zipfile.ZipFile(wheel, "w") as archive:
                        for name, content in entries.items():
                            archive.writestr(name, content)
                    if entries == {"statadic/example.py": "VALUE = 1\n"}:
                        candidate.verify_payload(wheels)
                    else:
                        with self.assertRaises(ValueError):
                            candidate.verify_payload(wheels)


if __name__ == "__main__":
    unittest.main()
