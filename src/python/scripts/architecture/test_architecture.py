"""Exercise real forbidden imports rather than configuration snapshots."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import script


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.python = self.root / "src/python"
        self.write(
            "pyproject.toml",
            """
[project]
name = "workspace"
version = "0.1.0"
[tool.statadic_architecture.workspace_dependencies]
statadic = []
statadic_evidence = []
statadic_studies = ["statadic", "statadic_evidence"]
statadic_performance = ["statadic", "statadic_evidence"]
statadic_docs_labs = ["statadic", "statadic_studies", "statadic_evidence"]
""",
        )
        self.write(
            "tach.toml",
            """
source_roots = [
    "packages/statadic/src", "packages/evidence/src", "apps/studies/src",
    "apps/performance/src", "apps/docs/src",
]
root_module = "forbid"
ignore_type_checking_imports = false
include_string_imports = true
forbid_circular_dependencies = true
[[modules]]
path = "statadic"
depends_on = []
[[modules]]
path = "statadic.core"
depends_on = []
[[modules]]
path = "statadic.api"
depends_on = ["statadic.core"]
[[modules]]
paths = ["statadic_evidence", "statadic_studies", "statadic_performance", "statadic_docs_labs"]
depends_on = []
[[interfaces]]
from = ["statadic.core"]
expose = ["Public"]
[[interfaces]]
from = ["statadic.core"]
expose = ["_producer"]
visibility = ["statadic_studies.bounds"]
""",
        )
        self.packages = {
            "statadic": "packages/statadic",
            "statadic_evidence": "packages/evidence",
            "statadic_studies": "apps/studies",
            "statadic_performance": "apps/performance",
            "statadic_docs_labs": "apps/docs",
        }
        for name, path in self.packages.items():
            self.write(f"{path}/src/{name}/__init__.py", "")
            self.write(
                f"{path}/pyproject.toml",
                f'[project]\nname = "{name}"\nversion = "0.1.0"\n',
            )
        self.write(
            "packages/statadic/src/statadic/core.py", "Public = 1\n_producer = 2\n_secret = 3\n"
        )
        self.write("packages/statadic/src/statadic/api.py", "from statadic.core import Public\n")

    def write(self, relative, content):
        path = self.python / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def codes(self):
        return {finding.code for finding in script.check(self.root)}

    def native(self, command="check"):
        return subprocess.run(
            [sys.executable, "-m", "tach", command],
            cwd=self.python,
            text=True,
            capture_output=True,
        )

    def test_native_rejects_reverse_type_only_edge_and_private_import(self):
        self.assertEqual(self.native().returncode, 0)
        self.write(
            "packages/statadic/src/statadic/core.py",
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n"
            " from statadic.api import Model\n",
        )
        result = self.native()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.write("packages/statadic/src/statadic/core.py", "_secret = 1\n")
        self.write("packages/statadic/src/statadic/api.py", "from statadic.core import _secret\n")
        result = self.native()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_native_rejects_declared_module_cycle(self):
        path = self.python / "tach.toml"
        path.write_text(
            path.read_text().replace(
                'path = "statadic.core"\ndepends_on = []',
                'path = "statadic.core"\ndepends_on = ["statadic.api"]',
            )
        )
        self.write("packages/statadic/src/statadic/core.py", "from statadic.api import Model\n")
        self.assertNotEqual(self.native().returncode, 0)

    def test_workspace_fence_catches_native_gap_even_when_declared(self):
        self.write("packages/statadic/src/statadic/consumer.py", "import statadic_studies\n")
        self.write(
            "packages/statadic/pyproject.toml",
            '[project]\nname = "statadic"\nversion = "0.1.0"\n'
            'dependencies = ["statadic_studies"]\n',
        )
        self.assertEqual(self.native().returncode, 0)
        self.assertIn("WORKSPACE", self.codes())

    def test_allowed_workspace_directions_and_private_test_seams(self):
        self.write("apps/docs/src/statadic_docs_labs/use.py", "import statadic_studies\n")
        self.write(
            "apps/performance/src/statadic_performance/use.py",
            "from statadic.core import Public\n",
        )
        self.write(
            "packages/statadic/tests/test_private.py", "from statadic.core import _secret\n"
        )
        self.assertEqual(self.codes(), set())
        self.write(
            "apps/performance/src/statadic_performance/use.py", "import statadic_studies\n"
        )
        self.assertEqual(self.codes(), {"WORKSPACE"})
        self.write("packages/statadic/tests/test_private.py", "import statadic_docs_labs\n")
        self.assertEqual(sum(f.code == "WORKSPACE" for f in script.check(self.root)), 2)

    def test_public_interfaces_aliases_and_scoped_exception(self):
        self.write(
            "apps/studies/src/statadic_studies/bounds.py",
            "from statadic.core import _producer\n",
        )
        self.write(
            "apps/docs/src/statadic_docs_labs/use.py",
            "import statadic.core as c\nx = c.Public\n",
        )
        self.assertEqual(self.codes(), set())
        self.write(
            "apps/docs/src/statadic_docs_labs/use.py",
            "import statadic.core as c\nx = c._producer\n",
        )
        self.assertEqual(self.codes(), {"INTERFACE"})
        self.write(
            "apps/docs/src/statadic_docs_labs/use.py",
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n"
            " from statadic.core import _secret\n",
        )
        self.assertEqual(self.codes(), {"INTERFACE"})

    def test_relative_workspace_import_and_module_alias_access(self):
        self.write("apps/docs/src/statadic_docs_labs/use.py", "from . import native\n")
        self.write(
            "apps/docs/src/statadic_docs_labs/native.py",
            "import statadic\nx = statadic.core._secret\n",
        )
        self.assertEqual(self.codes(), {"INTERFACE"})
        self.write(
            "apps/performance/src/statadic_performance/use.py",
            "from importlib import import_module as load\nload('statadic_studies')\n",
        )
        self.assertIn("WORKSPACE", self.codes())

    def test_hidden_imports_and_import_system_aliases(self):
        cases = [
            (
                "from importlib import import_module as load\nload('statadic.api')\n",
                "DYNAMIC-IMPORT",
            ),
            ("import sys as s\ns.modules['statadic'] = object()\n", "SYS-MUTATION"),
            ("from sys import path as p\np.append('/tmp')\n", "SYS-MUTATION"),
            ("try:\n import scipy\nexcept ImportError:\n pass\n", "TRY-IMPORT"),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                self.write("packages/statadic/src/statadic/consumer.py", source)
                self.assertIn(expected, self.codes())
        self.write(
            "packages/statadic/src/statadic/consumer.py",
            """
from importlib.metadata import version
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from shape_extensions import Shape
def optional():
    from jax.scipy.special import betainc
    return betainc
""",
        )
        self.assertEqual(self.codes(), set())

    def test_parse_errors_missing_policy_and_no_tracked_writes(self):
        before = {p: p.read_bytes() for p in self.python.rglob("*") if p.is_file()}
        self.assertEqual(self.codes(), set())
        self.assertEqual(
            before,
            {p: p.read_bytes() for p in self.python.rglob("*") if p.is_file()},
        )
        self.write("apps/docs/src/statadic_docs_labs/broken.py", "def broken(\n")
        self.assertIn("PARSE", self.codes())
        self.write("apps/docs/src/unregistered/__init__.py", "")
        with self.assertRaises(ValueError):
            script.check(self.root)

    def test_purity_rejects_direct_and_hidden_capabilities(self):
        cases = [
            "import posix\nposix.system('python')\n",
            "from _ctypes import dlopen\n",
            "from importlib.metadata import EntryPoint\n",
            "from importlib import metadata\nmetadata.entry_points()\n",
            "import importlib.metadata as metadata\n"
            "getattr(metadata, 'entry_points')()\n",
            "import os\nspawn = os.posix_spawnp\n",
            "import numpy as numerical\n",
            "from scipy.linalg import solve\n",
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n import numpy\n",
            "import importlib as loader\nloader.import_module('num' + 'py')\n",
            "loader = __import__\nloader('numpy')\n",
            "exec('import numpy')\n",
            "import os\nlaunch = os.system\nlaunch('python -c import numpy')\n",
            "import ctypes\n",
            "import pickle\npickle.loads(b'payload')\n",
            "import sys\nget = sys.modules.get\nget('numpy')\n",
            "import jax\njax._src.numpy.lax_numpy.np\n",
            "from jax.numpy import load as read\n",
            "import builtins\ngetattr(builtins, 'eval')('import numpy')\n",
            "getattr(__builtins__, 'eval')('import numpy')\n",
        ]
        for source in cases:
            with self.subTest(source=source):
                self.write("packages/statadic/src/statadic/consumer.py", source)
                self.assertTrue(self.codes() & {"JAX-PURITY", "DYNAMIC-IMPORT"})
        self.write(
            "packages/statadic/src/statadic/consumer.py",
            "import jax.numpy as jnp\n"
            "from jax.scipy.special import betainc\n"
            "from importlib.metadata import version\n"
            "import os\nos.fdopen(1, 'w')\n",
        )
        self.assertEqual(self.codes(), set())
        self.write("packages/statadic/tests/test_oracle.py", "import numpy\nimport scipy\n")
        self.assertEqual(self.codes(), set())

    def test_custom_derivatives_and_detached_gradients_require_contracts(self):
        cases = [
            "import jax\n@jax.custom_jvp\ndef f(x): return x\n",
            "import jax\ncustom = jax.custom_vjp\nf = custom(lambda x: x)\n",
            "from jax.lax import stop_gradient as detach\nx = detach(1.)\n",
            "import jax\ndetach: object = jax.lax.stop_gradient\nx = detach(1.)\n",
        ]
        for source in cases:
            with self.subTest(source=source):
                self.write("packages/statadic/src/statadic/consumer.py", source)
                self.assertIn("AUTODIFF", self.codes())

    def test_native_external_ban_overrides_declaration(self):
        path = self.python / "tach.toml"
        path.write_text(
            path.read_text().replace(
                'path = "statadic"',
                'path = "statadic"\ncannot_depend_on_external = ["numpy", "scipy"]',
            )
        )
        self.write(
            "packages/statadic/pyproject.toml",
            '[project]\nname = "statadic"\nversion = "0.1.0"\ndependencies = ["numpy"]\n',
        )
        self.write("packages/statadic/src/statadic/consumer.py", "import numpy\n")
        result = self.native("check-external")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_native_external_rejects_undeclared_dependency(self):
        self.write("packages/statadic/src/statadic/consumer.py", "import numpy\n")
        result = self.native("check-external")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_unused_external_dependencies_are_advisory(self):
        self.write("packages/statadic/src/statadic/consumer.py", "import numpy\n")
        with (self.python / "tach.toml").open("a") as handle:
            handle.write('[rules]\nunused_external_dependencies = "warn"\n')
        self.write(
            "packages/statadic/pyproject.toml",
            '[project]\nname = "statadic"\nversion = "0.1.0"\n'
            'dependencies = ["numpy", "rich"]\n',
        )
        result = self.native("check-external")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("WARN", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
