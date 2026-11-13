"""Lean setup fetches the discovered Mathlib imports, never an accidental full cache."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

JUSTFILE = Path(__file__).resolve().parents[2] / "src/lean/justfile"


class LeanCacheTests(unittest.TestCase):
    def run_cache(self, rg_script):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, script in {
                "uv": "exit 0",
                "rg": rg_script,
                "lake": 'printf "%s\\n" "$@" > "$LAKE_ARGS"',
            }.items():
                executable = root / name
                executable.write_text(f"#!{shutil.which('bash')}\n{script}\n")
                executable.chmod(0o755)
            env = dict(os.environ, PATH=f"{root}{os.pathsep}{os.environ['PATH']}")
            env.pop("BASH_FUNC_uv%%", None)
            env["LAKE_ARGS"] = str(root / "lake-args")
            result = subprocess.run(
                [
                    "just",
                    "--justfile",
                    str(JUSTFILE),
                    "--working-directory",
                    directory,
                    "cache",
                    "Mathlib.Extra",
                ],
                cwd=directory,
                env=env,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            args = root / "lake-args"
            return result, args.read_text().splitlines() if args.exists() else None

    def test_imports_and_explicit_modules_reach_lake(self):
        result, args = self.run_cache(
            "printf '%s\\n' 'import Mathlib.Z' 'public import Mathlib.A' 'import Mathlib.Z'"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            args, ["exe", "cache", "get", "Mathlib.A", "Mathlib.Z", "Mathlib.Extra"]
        )

    def test_no_imports_or_search_failure_never_fetches_full_cache(self):
        for script in ("exit 0", "exit 2"):
            with self.subTest(script=script):
                result, args = self.run_cache(script)
                self.assertNotEqual(result.returncode, 0)
                self.assertIsNone(args)


if __name__ == "__main__":
    unittest.main()
