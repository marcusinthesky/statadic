"""Behavioral tests for the public scientific API inventory."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "statadic_inventory", Path(__file__).parents[1] / "inventory.py"
)
inventory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory)


class InventoryTest(unittest.TestCase):
    def test_type_checking_adapter_exports_its_runtime_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            source = package / "src/statadic"
            source.mkdir(parents=True)
            (package / "pyproject.toml").write_text('[project]\nversion = "0.4.1"\n')
            (source / "__init__.py").write_text("__all__ = []\n")
            (source / "science.py").write_text(
                "from typing import TYPE_CHECKING\n"
                "if TYPE_CHECKING:\n"
                "    from external_stubs import Array\n"
                "else:\n"
                "    class Array: pass\n"
                "__all__ = ['Array']\n"
            )
            exported = inventory.inventory(source, package)["modules"][1]["exports"]
            self.assertEqual(
                [(item["name"], item["kind"], item["target"]) for item in exported],
                [("Array", "class", "statadic.science.Array")],
            )

    def test_new_class_and_public_alias_of_private_function(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            source = package / "src/statadic"
            source.mkdir(parents=True)
            (package / "pyproject.toml").write_text('[project]\nversion = "0.4.1"\n')
            (source / "__init__.py").write_text("__all__ = []\n")
            (source / "_private.py").write_text("def hidden(): pass\n")
            public = source / "science.py"
            public.write_text(
                "from ._private import hidden as exposed\n__all__ = ['exposed']\n"
            )
            before = inventory.inventory(source, package)
            exported = before["modules"][1]["exports"][0]
            self.assertEqual(
                (exported["name"], exported["kind"], exported["target"]),
                ("exposed", "function", "statadic._private.hidden"),
            )
            public.write_text(
                "from ._private import hidden as exposed\nclass Estimate: pass\n__all__ = ['exposed', 'Estimate']\n"
            )
            after = inventory.inventory(source, package)
            self.assertNotEqual(before, after)
            self.assertEqual(
                {item["name"] for item in after["modules"][1]["exports"]},
                {"Estimate", "exposed"},
            )

    def test_public_modules_need_literal_all(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            source = package / "src/statadic"
            source.mkdir(parents=True)
            (source / "__init__.py").write_text("__all__: list[str] = []\n")
            missing = source / "missing.py"
            missing.write_text("from external import thing\ndef absent(): pass\n")
            nonliteral = source / "nonliteral.py"
            nonliteral.write_text("class Unlisted: pass\n__all__ = exports\n")
            augmented = source / "augmented.py"
            augmented.write_text("__all__ = []\n__all__ += ['thing']\n")
            with self.assertRaises(ValueError) as raised:
                inventory.inventory(source, package)
            self.assertEqual(
                str(raised.exception).splitlines(),
                [
                    f"{augmented}: public module needs a literal __all__",
                    f"{missing}: public module needs a literal __all__",
                    f"{missing}: absent is not in __all__; list it or add a leading underscore",
                    f"{nonliteral}: public module needs a literal __all__",
                    f"{nonliteral}: Unlisted is not in __all__; list it or add a leading underscore",
                ],
            )

    def test_unlisted_public_definitions_are_reported_together(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            source = package / "src/statadic"
            source.mkdir(parents=True)
            (source / "__init__.py").write_text("__all__ = []\n")
            public = source / "science.py"
            public.write_text(
                "from typing import TYPE_CHECKING\n"
                "from external import imported\n"
                "ordinary = 1\n"
                "type Result = int\n"
                "async def compute(): pass\n"
                "def _private(): pass\n"
                "if TYPE_CHECKING:\n"
                "    class StaticOnly: pass\n"
                "else:\n"
                "    class Runtime: pass\n"
                "__all__ = []\n"
            )
            with self.assertRaises(ValueError) as raised:
                inventory.inventory(source, package)
            self.assertEqual(
                str(raised.exception).splitlines(),
                [
                    f"{public}: {name} is not in __all__; list it or add a leading underscore"
                    for name in ("Result", "Runtime", "compute")
                ],
            )

    def test_runtime_static_jaxtyping_must_be_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            source = package / "src/statadic"
            core = source / "core"
            core.mkdir(parents=True)
            (source / "__init__.py").write_text("__all__ = []\n")
            (core / "__init__.py").write_text("__all__ = []\n")
            typing = core / "typing.py"
            typing.write_text(
                "from typing import TYPE_CHECKING\n"
                "if TYPE_CHECKING:\n"
                "    from external import static_jaxtyping\n"
                "else:\n"
                "    def static_jaxtyping(): pass\n"
                "__all__ = []\n"
            )
            with self.assertRaises(ValueError) as raised:
                inventory.inventory(source, package)
            self.assertEqual(
                str(raised.exception),
                f"{typing}: static_jaxtyping is not in __all__; list it or add a leading underscore",
            )

    def test_private_class_alias_reports_own_methods_and_properties(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            source = package / "src/statadic"
            source.mkdir(parents=True)
            (package / "pyproject.toml").write_text('[project]\nversion = "0.4.1"\n')
            (source / "__init__.py").write_text("__all__ = []\n")
            private = source / "_private.py"
            private.write_text(
                "class Estimator:\n    def fit(self): pass\n    @property\n    def result(self): pass\n"
            )
            (source / "science.py").write_text(
                "from ._private import Estimator as Model\n__all__ = ['Model']\n"
            )
            before = inventory.inventory(source, package)
            exported = before["modules"][1]["exports"][0]
            self.assertEqual(exported["target"], "statadic._private.Estimator")
            self.assertEqual(
                {(m["name"], m["kind"]) for m in exported["members"]},
                {("fit", "method"), ("result", "property")},
            )
            private.write_text(private.read_text() + "    def predict(self): pass\n")
            after = inventory.inventory(source, package)
            self.assertNotEqual(before, after)
            self.assertIn(
                "predict",
                {m["name"] for m in after["modules"][1]["exports"][0]["members"]},
            )


if __name__ == "__main__":
    unittest.main()
