"""Behavioral tests for the formal-claim certification boundary."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bridge


def claim() -> dict:
    return {
        "id": "STATADIC-TEST-001",
        "lean": {"module": "Example", "declaration": "Example.fact"},
        "source": {"key": "paper", "locator": "Theorem 1", "proof_locator": "p. 2"},
        "scope": {
            "conclusion": "A",
            "assumptions": ["A"],
            "diagnostics": ["Check a_b & 5%."],
        },
        "python": {"symbols": []},
        "methods": ["linear"],
    }


def compiled() -> dict:
    return {
        "source_hashes": {"proof.lean": "original"},
        "declarations": [
            {
                "declaration": "Example.fact",
                "module": "Example",
                "is_theorem": True,
                "statement_type": "A → A",
                "axioms": [],
            }
        ],
    }


def node() -> dict:
    return {
        "latexLabel": "STATADIC-TEST-001",
        "name": "Example.fact",
        "notReady": False,
        "hasLean": True,
        "latex": {
            "statement": "A implies A",
            "proof": "Identity",
            "artifact": "formal/test.tex",
            "sha256": "a" * 64,
        },
    }


class BridgeTests(unittest.TestCase):
    def test_original_claim_requires_review_instead_of_publication(self) -> None:
        import jsonschema

        data = bridge.read(bridge.REGISTRY / "claims.json")
        original = copy.deepcopy(
            next(claim for claim in data["claims"] if "source" in claim)
        )
        original.pop("source")
        original["kind"] = "algorithm-refinement"
        original["review"] = (
            "Exact-arithmetic model reviewed; Python link remains open."
        )
        schema = bridge.read(bridge.REGISTRY / "schema.json")
        jsonschema.validate({"schema_version": 1, "claims": [original]}, schema)
        original.pop("review")
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate({"schema_version": 1, "claims": [original]}, schema)
        original["kind"] = "published-specialization"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate({"schema_version": 1, "claims": [original]}, schema)

    def test_definition_change_invalidates_unchanged_statement(self) -> None:
        before = bridge.validate_claims([claim()], compiled(), [node()])[0]
        changed = compiled()
        changed["source_hashes"]["definition.lean"] = "changed-definition-body"
        after = bridge.validate_claims([claim()], changed, [node()])[0]
        self.assertEqual(before["statement_type"], after["statement_type"])
        self.assertNotEqual(before["statement_pin"], after["statement_pin"])

    def test_registry_math_and_special_text_are_safe_for_tex(self) -> None:
        block = bridge.manuscript_list(["β̂ and Ĝ: x₀ ≥ 10⁻⁸, a_b & 5%. Wⱼ ≈ α/√N."])
        for expected in (
            r"\hat\beta",
            r"\hat G",
            "{}_j",
            r"\approx",
            r"\alpha",
            r"\sqrt{N}",
            "{}_0",
            r"\ge",
            "{}^{-8}",
            r"a\_b \& 5\%",
        ):
            self.assertIn(expected, block)
        self.assertNotIn("̂", block)

    def test_manuscript_uses_native_statement_without_theorem_environment(self) -> None:
        joined = bridge.validate_claims([claim()], compiled(), [node()])
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            bridge.manuscript_fragments(joined, destination)
            self.assertEqual(
                (destination / "STATADIC-TEST-001-statement.tex").read_text(),
                "A implies A\n",
            )
            macros = (destination / "statadic-mathematics.tex").read_text()
            self.assertIn(r"\newcommand{\statadicstatement}", macros)
            self.assertIn("Unknown statement #1", macros)
            self.assertNotIn(r"\begin{theorem}", macros)
            self.assertIn(r"\newcommand{\statadicdiagnostics}", macros)
            self.assertIn(r"\item Check a\_b \& 5\%.", macros)

    def test_external_dependency_change_invalidates_unchanged_statement(self) -> None:
        initial = compiled()
        initial["source_hashes"]["lake-manifest.json"] = "old"
        changed = copy.deepcopy(initial)
        changed["source_hashes"]["lake-manifest.json"] = "new"
        before = bridge.validate_claims([claim()], initial, [node()])[0]
        after = bridge.validate_claims([claim()], changed, [node()])[0]
        self.assertEqual(before["statement_type"], after["statement_type"])
        self.assertNotEqual(before["statement_pin"], after["statement_pin"])

    def test_claim_join_passes_and_pins_statement(self) -> None:
        joined = bridge.validate_claims([claim()], compiled(), [node()])
        self.assertEqual(joined[0]["proof_status"], "kernel-checked-and-axiom-audited")
        self.assertEqual(len(joined[0]["statement_pin"]), 64)
        self.assertEqual(joined[0]["latex"], node()["latex"])

    def test_unmapped_native_label_and_wrong_declaration_fail(self) -> None:
        extra = node() | {"latexLabel": "STATADIC-TEST-002"}
        with self.assertRaisesRegex(ValueError, "labels differ"):
            bridge.validate_claims([claim()], compiled(), [node(), extra])
        with self.assertRaisesRegex(ValueError, "native declaration mismatch"):
            bridge.validate_claims(
                [claim()], compiled(), [node() | {"name": "Example.other"}]
            )
        bad = copy.deepcopy(compiled())
        bad["declarations"][0]["module"] = "Other"
        with self.assertRaisesRegex(ValueError, "missing from fresh compiled census"):
            bridge.validate_claims([claim()], bad, [node()])

    def test_unready_and_duplicate_native_nodes_fail(self) -> None:
        for changed in ({"notReady": True}, {"hasLean": False}):
            with (
                self.subTest(changed=changed),
                self.assertRaisesRegex(ValueError, "not ready"),
            ):
                bridge.validate_claims([claim()], compiled(), [node() | changed])
        with self.assertRaisesRegex(ValueError, "duplicate native"):
            bridge.validate_claims([claim()], compiled(), [node(), node()])

    def test_open_obligation_remains_open_with_related_checked_claim(self) -> None:
        data = {
            "claims": [claim() | {"proof_status": "kernel-checked-and-axiom-audited"}],
            "methods": [
                {
                    "id": "linear",
                    "obligations": [
                        {
                            "id": "linear-001",
                            "title": "full theorem",
                            "source_status": "open",
                            "claims": ["STATADIC-TEST-001"],
                        }
                    ],
                }
            ],
        }
        self.assertEqual(bridge.coverage(data), ["linear: full theorem"])

    def test_unrelated_checked_claim_does_not_satisfy_another_method(self) -> None:
        data = {
            "claims": [claim() | {"proof_status": "kernel-checked-and-axiom-audited"}],
            "methods": [
                {
                    "id": "energy",
                    "obligations": [
                        {
                            "id": "energy-001",
                            "title": "energy theorem",
                            "source_status": "source-matched",
                            "claims": ["STATADIC-TEST-001"],
                        }
                    ],
                }
            ],
        }
        self.assertEqual(bridge.coverage(data), ["energy: energy theorem"])

    def test_reviewed_exact_obligation_mapping_closes(self) -> None:
        mapped_claim = claim() | {
            "proof_status": "kernel-checked-and-axiom-audited",
            "obligations": ["linear-001"],
        }
        obligation = {
            "id": "linear-001",
            "title": "proved theorem",
            "source_status": "source-matched",
            "scope_review": "agent-reviewed",
            "claims": ["STATADIC-TEST-001"],
        }
        self.assertEqual(
            bridge.coverage(
                {
                    "claims": [mapped_claim],
                    "methods": [{"id": "linear", "obligations": [obligation]}],
                }
            ),
            [],
        )

    def test_statement_pin_changes_when_type_changes(self) -> None:
        first = bridge.validate_claims([claim()], compiled(), [node()])[0][
            "statement_pin"
        ]
        changed = copy.deepcopy(compiled())
        changed["declarations"][0]["statement_type"] = "A → B → A"
        second = bridge.validate_claims([claim()], changed, [node()])[0][
            "statement_pin"
        ]
        self.assertNotEqual(first, second)

    def test_annotation_changes_change_statement_pin(self) -> None:
        first = bridge.validate_claims([claim()], compiled(), [node()])[0][
            "statement_pin"
        ]
        for field in ("statement", "proof"):
            modified = node()
            modified["latex"][field] += " changed"
            with self.subTest(field=field):
                second = bridge.validate_claims([claim()], compiled(), [modified])[0][
                    "statement_pin"
                ]
                self.assertNotEqual(first, second)

    def test_empty_and_duplicate_obligation_registry_rejected(self) -> None:
        for methods in ([], [{"id": "linear", "obligations": []}]):
            with self.subTest(methods=methods), self.assertRaises(ValueError):
                bridge.coverage({"claims": [], "methods": methods})
        obligation = {"id": "shared-001", "title": "example", "source_status": "open"}
        methods = [
            {"id": "linear", "obligations": [obligation]},
            {"id": "energy", "obligations": [dict(obligation)]},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            bridge.coverage({"claims": [], "methods": methods})

    def test_compiled_source_change_during_native_extraction_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(bridge, "REGISTRY", root),
                patch.object(bridge, "OUTPUT", root / "out.json"),
                patch.object(
                    bridge,
                    "validate_registry",
                    return_value=(
                        [claim()],
                        {
                            "methods": [
                                {
                                    "id": "linear",
                                    "obligations": [
                                        {
                                            "id": "linear-001",
                                            "title": "pending",
                                            "source_status": "open",
                                        }
                                    ],
                                }
                            ]
                        },
                    ),
                ),
                patch.object(bridge, "tracked_inputs", return_value={}),
                patch.object(bridge.subprocess, "run"),
                patch.object(bridge, "audit_lean", return_value=compiled()),
                patch.object(bridge, "native_nodes", return_value=[node()]),
                patch("integrity.discover", return_value={}),
                patch("integrity.snapshot", return_value={"proof.lean": "changed"}),
            ):
                with self.assertRaisesRegex(ValueError, "changed|stale"):
                    bridge.certify(pin=True)
                self.assertFalse((root / "out.json").exists())

    def test_statement_pin_change_blocks_certify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "statement-pins.json").write_text(
                json.dumps({"pins": {"STATADIC-TEST-001": "old"}})
            )
            with (
                patch.object(bridge, "REGISTRY", root),
                patch.object(bridge, "OUTPUT", root / "out.json"),
                patch.object(
                    bridge,
                    "validate_registry",
                    return_value=([claim()], {"methods": []}),
                ),
                patch.object(bridge, "tracked_inputs", return_value={}),
                patch.object(bridge.subprocess, "run"),
                patch.object(bridge, "audit_lean", return_value=compiled()),
                patch.object(bridge, "native_nodes", return_value=[node()]),
                patch("integrity.discover", return_value={}),
                patch("integrity.snapshot", return_value={"proof.lean": "original"}),
            ):
                with self.assertRaisesRegex(
                    ValueError, "statement/source mapping changed"
                ):
                    bridge.certify()
                self.assertFalse((root / "out.json").exists())

    def test_stale_certification_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "src/lean/build/certification.json"
            output.parent.mkdir(parents=True)
            output.write_text(json.dumps({"inputs": {"source": "old"}}))
            with (
                patch.object(bridge, "tracked_inputs", return_value={"source": "new"}),
                self.assertRaisesRegex(ValueError, "certification is stale"),
            ):
                bridge.fresh_output(root)

    def _assert_nested_input_changes_stale_certification(self, relative: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in (
                "src/latex/refs.bib",
                "src/lean/tools/bridge.py",
                "src/python/apps/docs/src/statadic_docs_labs/formal_extension.py",
            ):
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("# original input\n")
            bridge.write(root / "src/lean/registry/public-api.json", {})
            output_path = root / "src/lean/build/certification.json"
            output = {
                "inputs": bridge.tracked_inputs(root),
                "compiled": {"source_hashes": {}},
                "claims": [],
            }
            # Keep the existing top-level trust inputs in the recursive inventory.
            for path in (
                "src/lean/tools/bridge.py",
                "src/python/apps/docs/src/statadic_docs_labs/formal_extension.py",
            ):
                self.assertIn(path, output["inputs"])
            bridge.write(output_path, output)
            nested = root / relative
            nested.parent.mkdir(parents=True, exist_ok=True)
            with (
                patch("integrity.discover", return_value={}),
                patch("integrity.snapshot", return_value={}),
                patch.object(bridge, "inventory", return_value={}),
            ):
                self.assertEqual(bridge.fresh_output(root), output)
                for change in ("created", "changed"):
                    with self.subTest(path=relative, change=change):
                        nested.write_text(f"# {change} nested implementation\n")
                        with self.assertRaisesRegex(
                            ValueError, "certification is stale"
                        ):
                            bridge.fresh_output(root)
                        self.assertIn(relative, bridge.tracked_inputs(root))
                        # Refresh only this synthetic certificate's byte inventory;
                        # the next mutation must stale it again without mocked hashes.
                        output["inputs"] = bridge.tracked_inputs(root)
                        bridge.write(output_path, output)
                        self.assertEqual(bridge.fresh_output(root), output)

    def test_nested_lean_tool_creation_and_change_stale_certification(self) -> None:
        self._assert_nested_input_changes_stale_certification(
            "src/lean/tools/statadic_bridge/render/manuscript.py"
        )

    def test_nested_docs_renderer_creation_and_change_stale_certification(self) -> None:
        self._assert_nested_input_changes_stale_certification(
            "src/python/apps/docs/src/statadic_docs_labs/evidence/claims/render.py"
        )

    def test_stale_compiled_source_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "src/lean/build/certification.json"
            output.parent.mkdir(parents=True)
            output.write_text(
                json.dumps(
                    {"inputs": {}, "compiled": {"source_hashes": {"proof.lean": "old"}}}
                )
            )
            with (
                patch.object(bridge, "tracked_inputs", return_value={}),
                patch("integrity.discover", return_value={}),
                patch("integrity.snapshot", return_value={"proof.lean": "new"}),
                self.assertRaisesRegex(ValueError, "proof evidence is stale"),
            ):
                bridge.fresh_output(root)

    def test_missing_python_symbol_rejected(self) -> None:
        with self.assertRaises((ValueError, AttributeError, ModuleNotFoundError)):
            bridge.resolve_symbol("no_such_statadic_test_module.some_symbol")

    def test_duplicate_claims_and_missing_citation_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src/lean/registry").mkdir(parents=True)
            fixture = claim()
            documents = {
                "claims.json": {"claims": [fixture, copy.deepcopy(fixture)]},
                "schema.json": {},
            }
            fake_schema = types.ModuleType("jsonschema")
            fake_schema.validate = lambda *_: None
            fake_pybtex = types.ModuleType("pybtex")
            fake_database = types.ModuleType("pybtex.database")
            fake_database.parse_file = lambda _: types.SimpleNamespace(entries={})
            fake_pybtex.database = fake_database

            def fake_read(path: Path) -> dict:
                return documents[path.name]

            with (
                patch.dict(
                    sys.modules,
                    {
                        "jsonschema": fake_schema,
                        "pybtex": fake_pybtex,
                        "pybtex.database": fake_database,
                    },
                ),
                patch.object(bridge, "read", side_effect=fake_read),
            ):
                with self.assertRaisesRegex(ValueError, "duplicate claim"):
                    bridge.validate_registry(root)
                documents["claims.json"]["claims"] = [fixture]
                with self.assertRaisesRegex(ValueError, "missing bibliography key"):
                    bridge.validate_registry(root)


if __name__ == "__main__":
    unittest.main()
