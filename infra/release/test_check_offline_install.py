"""Exercise artifact-locked consumer installs with no registry or warm cache."""

import base64
import csv
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from check_offline_install import install_locked_environment, sync_locked_dependencies

UV = shutil.which("uv")


def wheel(directory, name, requires=(), module="VALUE = 42\n"):
    """Make a tiny standards-compliant wheel without downloading build tools."""
    dist = f"{name}-1.0.dist-info"
    entries = {
        f"{name}/__init__.py": module,
        f"{dist}/METADATA": (
            f"Metadata-Version: 2.3\nName: {name}\nVersion: 1.0\n"
            + "".join(f"Requires-Dist: {requirement}\n" for requirement in requires)
        ),
        f"{dist}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: statadic-test\n"
            "Root-Is-Purelib: true\nTag: py3-none-any\n"
        ),
    }
    record = io.StringIO()
    writer = csv.writer(record, lineterminator="\n")
    for path, value in entries.items():
        data = value.encode()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        writer.writerow([path, f"sha256={digest.decode()}", len(data)])
    writer.writerow([f"{dist}/RECORD", "", ""])
    entries[f"{dist}/RECORD"] = record.getvalue()
    path = directory / f"{name}-1.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for filename, data in entries.items():
            archive.writestr(filename, data)
    return path


@unittest.skipUnless(UV, "The offline installer integration tests require uv")
class OfflineInstallTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.environment = os.environ.copy()
        for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            self.environment.pop(name, None)
        self.environment.update(
            UV_CACHE_DIR=str(self.directory / "empty-cache"),
            UV_OFFLINE="true",
            UV_PYTHON_DOWNLOADS="never",
        )
        self.target = self.directory / "venv"
        subprocess.run(
            [UV, "venv", "--python", sys.executable, str(self.target)],
            cwd=self.directory,
            env=self.environment,
            check=True,
        )
        self.python = self.target / "bin/python"
        self.dependency = wheel(self.directory, "locked_dependency")
        digest = hashlib.sha256(self.dependency.read_bytes()).hexdigest()
        self.lock = self.directory / "pylock.toml"
        self.lock.write_text(
            'lock-version = "1.0"\ncreated-by = "statadic-test"\n'
            'requires-python = ">=3.11"\n'
            '[[packages]]\nname = "locked-dependency"\nversion = "1.0"\n'
            f'wheels = [{{ path = "{self.dependency.name}", '
            f'hashes = {{ sha256 = "{digest}" }} }}]\n'
        )

    def install(self, consumer):
        install_locked_environment(
            UV,
            self.python,
            self.lock,
            [consumer],
            self.directory,
            self.environment,
        )

    def test_empty_cache_offline_install_and_isolated_import(self):
        consumer = wheel(self.directory, "consumer", ["locked-dependency==1.0"])
        self.install(consumer)
        subprocess.run(
            [
                str(self.python),
                "-I",
                "-c",
                (
                    "import consumer, locked_dependency, sys; from pathlib import Path; "
                    "assert locked_dependency.VALUE == 42; "
                    "assert all(Path(module.__file__).resolve().is_relative_to("
                    "Path(sys.prefix).resolve()) for module in (consumer, locked_dependency))"
                ),
            ],
            cwd=self.directory,
            env=self.environment,
            check=True,
        )

    def test_changed_locked_artifact_is_rejected(self):
        consumer = wheel(self.directory, "consumer", ["locked-dependency==1.0"])
        with zipfile.ZipFile(self.dependency, "a") as archive:
            archive.writestr("unexpected.txt", "changed after locking")
        with self.assertRaises(subprocess.CalledProcessError):
            self.install(consumer)

    def test_no_deps_does_not_hide_missing_wheel_requirement(self):
        consumer = wheel(self.directory, "consumer", ["missing-dependency==1.0"])
        with self.assertRaises(subprocess.CalledProcessError):
            self.install(consumer)

    def source_dependency(self):
        """Provide a source-only artifact and an independently locked build backend."""
        backend = wheel(
            self.directory,
            "locked_builder",
            module=(
                "from pathlib import Path\nimport shutil\n"
                "def build_wheel(wheel_directory, config_settings=None, "
                "metadata_directory=None):\n"
                f"    wheel = Path({self.dependency.name!r})\n"
                "    shutil.copyfile(wheel, Path(wheel_directory) / wheel.name)\n"
                "    return wheel.name\n"
            ),
        )
        source = self.directory / "source"
        source.mkdir()
        (source / "pyproject.toml").write_text(
            '[build-system]\nrequires = ["locked-builder==1.0"]\n'
            'build-backend = "locked_builder"\n'
        )
        shutil.copy2(self.dependency, source / self.dependency.name)
        archive = self.directory / "locked_dependency-1.0.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(source, arcname="locked_dependency-1.0")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        self.lock.write_text(
            'lock-version = "1.0"\ncreated-by = "statadic-test"\n'
            'requires-python = ">=3.11"\n'
            '[[packages]]\nname = "locked-dependency"\nversion = "1.0"\n'
            f'sdist = {{ path = "{archive.name}", '
            f'hashes = {{ sha256 = "{digest}" }} }}\n'
        )
        constraints = self.directory / "build-constraints.txt"
        digest = hashlib.sha256(backend.read_bytes()).hexdigest()
        constraints.write_text(
            f"locked-builder @ {backend.as_uri()} --hash=sha256:{digest}\n"
        )
        # Even the preparation phase must use only these local test artifacts.
        self.environment["UV_NO_INDEX"] = "true"
        return constraints, backend

    def test_source_dependency_prepared_then_installed_offline(self):
        constraints, _ = self.source_dependency()
        sync_locked_dependencies(
            UV,
            self.python,
            self.lock,
            self.directory,
            self.environment,
            prepare=True,
            build_constraints=constraints,
        )
        # Discard the preparation environment: the next consumer is genuinely fresh.
        shutil.rmtree(self.target)
        subprocess.run(
            [UV, "venv", "--python", sys.executable, str(self.target)],
            cwd=self.directory,
            env=self.environment,
            check=True,
        )
        consumer = wheel(self.directory, "consumer", ["locked-dependency==1.0"])
        install_locked_environment(
            UV,
            self.python,
            self.lock,
            [consumer],
            self.directory,
            self.environment,
            build_constraints=constraints,
        )
        subprocess.run(
            [
                str(self.python),
                "-I",
                "-c",
                "import locked_dependency; assert locked_dependency.VALUE == 42",
            ],
            cwd=self.directory,
            env=self.environment,
            check=True,
        )

    def test_source_build_rejects_changed_locked_backend(self):
        constraints, backend = self.source_dependency()
        with zipfile.ZipFile(backend, "a") as archive:
            archive.writestr("unexpected.txt", "changed after locking")
        with self.assertRaises(subprocess.CalledProcessError):
            sync_locked_dependencies(
                UV,
                self.python,
                self.lock,
                self.directory,
                self.environment,
                prepare=True,
                build_constraints=constraints,
            )


if __name__ == "__main__":
    unittest.main()
