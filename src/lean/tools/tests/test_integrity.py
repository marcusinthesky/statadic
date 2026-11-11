"""Failure-inclusive tests for the compiled Lean trust gate."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import integrity


class IntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="statadic-audit-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "tools").mkdir()
        shutil.copy(
            integrity.ROOTLEAN / "tools/Audit.lean", self.root / "tools/Audit.lean"
        )
        shutil.copy(
            integrity.ROOTLEAN / "tools/integrity.py", self.root / "tools/integrity.py"
        )
        (self.root / "lakefile.lean").write_text(
            "import Lake\nopen Lake DSL\npackage fixture\nlean_lib Fixture\n"
        )
        (self.root / "lean-toolchain").write_text(
            (integrity.ROOTLEAN / "lean-toolchain").read_text()
        )
        (self.root / "lake-manifest.json").write_text(
            '{"version":"1.2.0","packages":[],"name":"fixture","lakeDir":".lake"}'
        )
        (self.root / "check_setup.py").write_text("# fixture\n")
        (self.root / "Fixture.lean").write_text(
            "module\npublic section\ndef answer : Nat := 42\n"
        )

    def test_clean_fixture(self) -> None:
        report = integrity.certify(self.root)
        self.assertEqual(report["modules"], ["Fixture"])
        self.assertIn("Fixture.lean", report["source_hashes"])
        self.assertTrue(
            any(d["declaration"] == "answer" for d in report["declarations"])
        )

    def test_custom_axiom(self) -> None:
        (self.root / "Fixture.lean").write_text(
            "module\npublic section\naxiom mystery : False\n"
        )
        with self.assertRaisesRegex(ValueError, "mystery"):
            integrity.certify(self.root)

    def test_private_module_axiom(self) -> None:
        (self.root / "Fixture.lean").write_text(
            "module\nprivate axiom hiddenPrivate : False\n"
        )
        with self.assertRaisesRegex(ValueError, "hiddenPrivate"):
            integrity.certify(self.root)

    def test_hidden_sorry(self) -> None:
        (self.root / "Fixture.lean").write_text(
            "module\npublic section\ndef hidden : Nat := by sorry\n"
        )
        with self.assertRaises(ValueError):
            integrity.certify(self.root)

    def test_true_alias(self) -> None:
        (self.root / "Fixture.lean").write_text(
            "module\npublic section\ndef Alias : Prop := True\ntheorem disguised : Alias := by trivial\n"
        )
        with self.assertRaisesRegex(ValueError, "true-"):
            integrity.certify(self.root)

    def test_unimported_module(self) -> None:
        (self.root / "Fixture").mkdir()
        (self.root / "Fixture/Hidden.lean").write_text(
            "module\npublic section\naxiom hidden : False\n"
        )
        with self.assertRaisesRegex(ValueError, "hidden"):
            integrity.certify(self.root)

    def test_debug_preflight_ignores_comments_but_rejects_option(self) -> None:
        source = self.root / "Fixture.lean"
        source.write_text("-- set_option debug.skipKernelTC true\ndef answer := 1\n")
        integrity.check_debug_options([source])
        source.write_text("set_option «debug».skipKernelTC true\ndef answer := 1\n")
        with self.assertRaisesRegex(ValueError, "forbidden set_option"):
            integrity.certify(self.root)

    def test_source_changed_during_audit(self) -> None:
        def mutate(lean: Path, modules: dict[str, Path]) -> list[dict]:
            modules["Fixture"].write_text("def changed := 2\n")
            return []

        with (
            patch.object(integrity, "collect", side_effect=mutate),
            self.assertRaisesRegex(ValueError, "source files changed"),
        ):
            integrity.certify(self.root)


if __name__ == "__main__":
    unittest.main()
