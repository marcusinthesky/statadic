"""Source binding must reject a changed actual implementation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import correspondence


class CorrespondenceTests(unittest.TestCase):
    def test_bound_functions_exist_and_changes_invalidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "src/python/packages/statadic/src/statadic/example.py"
            path.parent.mkdir(parents=True)
            path.write_text("def actual(x):\n    return x + 1\n")
            with patch.object(
                correspondence, "SUBJECTS", {"actual": ("example.py", "actual")}
            ):
                before = correspondence.source_bindings(root)
                path.write_text("def actual(x):\n    return x + 2\n")
                after = correspondence.source_bindings(root)
            self.assertNotEqual(before, after)
