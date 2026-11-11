"""Fail-fast quality orchestration without compiling proof libraries."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/quality.sh"


class QualityTests(unittest.TestCase):
    def run_gate(self, failure: str = "") -> tuple[int, list[str]]:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            (root / "packages/StatadicFormal").mkdir(parents=True)
            shutil.copyfile(SCRIPT, root / "scripts/quality.sh")
            binaries = root / "bin"
            binaries.mkdir()
            lake = binaries / "lake"
            lake.write_text(
                '#!/usr/bin/env bash\nset -eu\nprintf "%s\\n" "$*" >> "$CALLS"\n'
                'if [[ "$*" == *"--wfail build"* ]]; then\n'
                '  [[ "$FAILURE" != build ]] || exit 17\n'
                '  touch "$FRESH"\nfi\n'
                'if [[ "$1" == lint ]]; then\n'
                '  [[ -f "$FRESH" ]] || exit 41\n'
                '  [[ "$FAILURE" != lint ]] || exit 23\nfi\n'
            )
            indexer = root / ".lake/packages/mathlib/.lake/build/bin/mk_all"
            indexer.parent.mkdir(parents=True)
            indexer.write_text(
                '#!/usr/bin/env bash\n[[ "$FAILURE" != indexes ]] || exit 29\n'
            )
            for executable in (lake, indexer):
                executable.chmod(0o755)
            calls = root / "calls"
            result = subprocess.run(
                ["bash", str(root / "scripts/quality.sh"), "lint"],
                env={
                    **os.environ,
                    "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
                    "CALLS": str(calls),
                    "FRESH": str(root / "fresh"),
                    "FAILURE": failure,
                },
                capture_output=True,
                text=True,
                check=False,
            )
            return result.returncode, calls.read_text().splitlines()

    def test_build_precedes_lint(self) -> None:
        status, calls = self.run_gate()
        self.assertEqual(status, 0, calls)
        self.assertIn("lint", calls)
        self.assertIn("env bash scripts/quality.sh text-style", calls)

    def test_index_or_build_failure_prevents_lint(self) -> None:
        for failure, expected in (("indexes", 29), ("build", 17)):
            with self.subTest(failure=failure):
                status, calls = self.run_gate(failure)
                self.assertEqual(status, expected, calls)
                self.assertNotIn("lint", calls)
                self.assertNotIn("env bash scripts/quality.sh text-style", calls)

    def test_lint_failure_stops_gate(self) -> None:
        status, calls = self.run_gate("lint")
        self.assertEqual(status, 23, calls)
        self.assertNotIn("env bash scripts/quality.sh text-style", calls)


if __name__ == "__main__":
    unittest.main()
