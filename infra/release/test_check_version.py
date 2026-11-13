"""Release gates must reject mismatched tags and stale version records."""

import json
import tempfile
import unittest
from pathlib import Path

from check_version import check


class VersionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write_versions(self, version="0.5.0a1", lock=None, registry=None):
        files = {
            "src/python/packages/statadic/pyproject.toml": (
                f'[project]\nname = "statadic"\nversion = "{version}"\n'
            ),
            "src/python/uv.lock": (
                f'[[package]]\nname = "statadic"\nversion = "{lock or version}"\n'
            ),
            "src/lean/registry/public-api.json": json.dumps(
                {"package_version": registry or version}
            ),
        }
        for relative, data in files.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(data)

    def test_stable_and_prerelease_tags(self):
        for version, expected in (
            ("1.0.0", False),
            ("0.5.0a1", True),
            ("0.5.0b2", True),
            ("0.5.0rc3", True),
        ):
            with self.subTest(version=version):
                self.write_versions(version)
                self.assertEqual(check(self.root, f"v{version}"), expected)

    def test_wrong_tag_fails(self):
        self.write_versions()
        for tag in ("v0.5.0", "0.5.0a1", "v0.5.0-alpha.1"):
            with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, "Tag"):
                check(self.root, tag)

    def test_stale_lock_fails(self):
        self.write_versions(lock="0.4.0")
        with self.assertRaisesRegex(ValueError, "uv.lock"):
            check(self.root)

    def test_stale_registry_fails(self):
        self.write_versions(registry="0.4.0")
        with self.assertRaisesRegex(ValueError, "registry"):
            check(self.root)

    def test_invalid_or_local_versions_fail(self):
        for version in ("1.0", "1.0.0+local", "1.0.0.dev1", "01.0.0"):
            with self.subTest(version=version):
                self.write_versions(version)
                with self.assertRaisesRegex(ValueError, "Unsupported"):
                    check(self.root)


if __name__ == "__main__":
    unittest.main()
