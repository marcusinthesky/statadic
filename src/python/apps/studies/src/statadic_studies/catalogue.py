"""Bidirectional discoverability checks against the existing public API inventory."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from statadic_studies.contracts import STUDIES

_REFERENCE_FIELDS = {
    "python_symbols",
    "construction_paths",
    "nodeids",
    "oracle_kind",
    "distribution",
    "call",
    "quantities",
    "conventions",
    "limits",
    "lanes",
}
_REFERENCE_LANES = {"external-float64", "precision-float32", "precision-float64"}


def _check_reference_mappings(repository, record, public, *, numerical=False):
    """Check bounded declarations without inferring execution or inherited coverage."""
    mappings = record.get("reference_mappings", [])
    if not isinstance(mappings, list):
        raise ValueError(f"Reference mappings must be a list: {record['id']}")
    identities = set()
    for mapping in mappings:
        if not isinstance(mapping, dict) or not _REFERENCE_FIELDS <= mapping.keys():
            raise ValueError(f"Incomplete reference mapping: {record['id']}")
        if set(mapping) - (_REFERENCE_FIELDS | {"configuration"}):
            raise ValueError(f"Unknown reference mapping fields: {record['id']}")
        for field in ("python_symbols", "construction_paths", "nodeids", "quantities"):
            values = mapping[field]
            if (
                not isinstance(values, list)
                or not values
                or any(
                    not isinstance(value, str) or not value.strip() for value in values
                )
                or len(values) != len(set(values))
            ):
                raise ValueError(f"Invalid reference {field}: {record['id']}")
        for field in ("distribution", "call", "conventions", "limits"):
            if not isinstance(mapping[field], str) or not mapping[field].strip():
                raise ValueError(f"Missing reference {field}: {record['id']}")
        lanes = mapping["lanes"]
        if (
            not isinstance(lanes, list)
            or any(not isinstance(lane, str) for lane in lanes)
            or len(lanes) != len(set(lanes))
            or not set(lanes) <= _REFERENCE_LANES
            or mapping["oracle_kind"]
            not in {"external", "independent-formula", "trusted-fixture"}
            or (mapping["oracle_kind"] == "external" and not lanes)
        ):
            raise ValueError(f"Invalid reference oracle/lanes: {record['id']}")
        if numerical and (
            not isinstance(mapping.get("configuration"), str)
            or not mapping["configuration"].strip()
        ):
            raise ValueError(
                f"Missing numerical reference configuration: {record['id']}"
            )
        if not set(mapping["python_symbols"]) <= set(record["python_symbols"]):
            raise ValueError(f"Reference changes its owner symbols: {record['id']}")
        if not set(mapping["construction_paths"]) <= public:
            raise ValueError(f"Unknown reference construction path: {record['id']}")
        identity = json.dumps(mapping, sort_keys=True)
        if identity in identities:
            raise ValueError(f"Duplicate reference mapping: {record['id']}")
        identities.add(identity)
        for nodeid in mapping["nodeids"]:
            segments = nodeid.split("::")
            relative = Path(segments[0])
            if (
                len(segments) < 2
                or not relative.parts
                or relative.is_absolute()
                or ".." in relative.parts
                or relative.parts[0] != "tests"
                or relative.suffix != ".py"
                or any(character in nodeid for character in "*?")
            ):
                raise ValueError(f"Invalid exact reference nodeid: {nodeid}")
            path = repository / "src/python/packages/statadic" / relative
            if not path.is_file():
                raise ValueError(f"Missing reference test file: {nodeid}")
            body = ast.parse(path.read_text()).body
            for segment in segments[1:]:
                name = segment.split("[", 1)[0]
                matches = [
                    node
                    for node in body
                    if isinstance(
                        node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                    )
                    and node.name == name
                ]
                if len(matches) != 1:
                    raise ValueError(f"Missing reference test definition: {nodeid}")
                body = matches[0].body
    return mappings


def _reference_definitions(package):
    """Inventory literal reference markers without importing optional test modules."""

    def marked(nodes):
        return any(
            isinstance(node, ast.Attribute)
            and node.attr == "reference"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "mark"
            for expression in nodes
            for node in ast.walk(expression)
        )

    result = set()
    for path in sorted((package / "tests").rglob("test*.py")):
        module = ast.parse(path.read_text())
        inherited = marked(
            [
                node.value
                for node in module.body
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "pytestmark"
                    for target in node.targets
                )
            ]
        )
        for node in module.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_") and (
                    inherited or marked(node.decorator_list)
                ):
                    result.add(f"{path.relative_to(package).as_posix()}::{node.name}")
            elif isinstance(node, ast.ClassDef):
                class_marked = inherited or marked(node.decorator_list)
                for child in node.body:
                    if (
                        isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and child.name.startswith("test_")
                        and (class_marked or marked(child.decorator_list))
                    ):
                        result.add(
                            f"{path.relative_to(package).as_posix()}::{node.name}::{child.name}"
                        )
    return result


def public_inventory(repository: Path) -> dict:
    """Read the formal inventory, preserving aliases and immediate members."""
    source = json.loads((repository / "src/lean/registry/public-api.json").read_text())
    result = {}
    for module in source["modules"]:
        for item in module["exports"]:
            name = f"{module['name']}.{item['name']}"
            result[name] = {
                "target": item.get("target", name),
                "kind": item["kind"],
                "family": module["method_id"],
            }
            for member in item.get("members", []):
                member_name = f"{name}.{member['name']}"
                result[member_name] = {
                    "target": f"{item.get('target', name)}.{member['name']}",
                    "kind": member["kind"],
                    "family": module["method_id"],
                }
    return result


def check_catalogue(repository: Path) -> dict:
    """Reject stale/missing symbols and report gaps in scientific coverage."""
    catalogue = json.loads(
        (repository / "src/python/apps/studies/catalogue.json").read_text()
    )
    inventory = public_inventory(repository)
    if set(catalogue["public_symbols"]) != set(inventory):
        raise ValueError(
            "Research catalogue differs from public inventory; review changed symbols"
        )
    methods = {item["id"]: item for item in catalogue["methods"]}
    if len(methods) != len(catalogue["methods"]):
        raise ValueError("Duplicate method identities")
    families = {
        item["id"]
        for item in json.loads(
            (repository / "src/lean/registry/methods.json").read_text()
        )["methods"]
    }
    for symbol, identity in catalogue["public_symbols"].items():
        if (
            identity not in methods
            or inventory[symbol]["target"] not in methods[identity]["python_symbols"]
        ):
            raise ValueError(f"Unresolved reverse link {symbol}")
    targets = {item["target"] for item in inventory.values()}
    labs = json.loads((repository / "src/python/apps/docs/labs.json").read_text())[
        "labs"
    ]
    lab_ids = [lab["slug"] for lab in labs]
    if len(lab_ids) != len(set(lab_ids)):
        raise ValueError("Duplicate lab identities")
    for lab in labs:
        if (
            not (
                repository / "src/python/apps/docs/notebooks" / lab["notebook"]
            ).is_file()
            or not (
                repository / "src/python/packages/statadic/docs/labs" / f"{lab['slug']}.md"
            ).is_file()
        ):
            raise ValueError(f"Missing lab notebook or page: {lab['slug']}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate identity: {key}")
            result[key] = value
        return result

    identities = json.loads(
        (
            repository
            / "src/python/apps/performance/src/statadic_performance/identities.json"
        ).read_text(),
        object_pairs_hook=unique_keys,
    )
    benchmarks = set()
    for workload, profiles in identities.items():
        for variants in profiles.values():
            if len(variants) != len(set(variants)):
                raise ValueError(f"Duplicate benchmark identity: {workload}")
            benchmarks.update(f"{workload}/{variant}" for variant in variants)
    benchmark_page = (
        repository / "src/python/packages/statadic/docs/reference/performance-workloads.md"
    )
    if not benchmark_page.is_file():
        raise ValueError("Missing benchmark page")
    page = benchmark_page.read_text()
    for identity in benchmarks:
        if f"(performance-{identity.replace('/', '-')})=" not in page:
            raise ValueError(f"Missing benchmark anchor: {identity}")
    for method in methods.values():
        for field, valid in (("tutorials", set(lab_ids)), ("benchmarks", benchmarks)):
            links = method.get(field, [])
            if len(links) != len(set(links)) or not set(links) <= valid:
                raise ValueError(f"Invalid or duplicate {field}: {method['id']}")
        if method["family"] not in families or not method["rationale"]:
            raise ValueError(f"Missing family/disposition rationale: {method['id']}")
        if not set(method["python_symbols"]) <= targets:
            raise ValueError(f"Unknown Python target: {method['id']}")
        if not set(method["studies"]) <= set(STUDIES):
            raise ValueError(f"Unknown study: {method['id']}")
        _check_reference_mappings(repository, method, set(inventory), numerical=True)
    for field, registered in (("tutorials", set(lab_ids)), ("benchmarks", benchmarks)):
        linked = {
            identity
            for method in methods.values()
            for identity in method.get(field, [])
        }
        missing = registered - linked
        if missing:
            raise ValueError(f"Unlinked {field}: {', '.join(sorted(missing))}")
    legacy_required = [
        item
        for item in methods.values()
        if item["disposition"] == "evaluation-required"
    ]
    ownership = [
        item["id"]
        for item in methods.values()
        if item["disposition"] == "ownership-review"
    ]
    procedures = catalogue.get("scientific_procedures", [])
    procedure_ids = [item["id"] for item in procedures]
    if len(procedure_ids) != len(set(procedure_ids)):
        raise ValueError("Duplicate scientific procedure identity")
    variants = []
    required_studies = set()
    for procedure in procedures:
        if (
            procedure["family"] not in families
            or not procedure.get("rationale")
            or not procedure.get("variants")
            or not procedure.get("python_symbols")
        ):
            raise ValueError(f"Incomplete scientific procedure: {procedure['id']}")
        if not set(procedure["python_symbols"]) <= targets:
            raise ValueError(f"Unknown scientific Python target: {procedure['id']}")
        for variant in procedure["variants"]:
            if not variant.get("selectors") or not variant.get("python_symbols"):
                raise ValueError(
                    f"Missing concrete variant selectors/symbols: {variant['id']}"
                )
            if not set(variant["python_symbols"]) <= set(procedure["python_symbols"]):
                raise ValueError(
                    f"Variant changes its scientific owner: {variant['id']}"
                )
            _check_reference_mappings(repository, variant, set(inventory))
            mappings = variant.get("study_mappings", [])
            expected_status = "covered" if mappings else "uncovered"
            if variant.get("adapter_status") != expected_status:
                raise ValueError(
                    f"Variant coverage contradicts study mappings: {variant['id']}"
                )
            for mapping in mappings:
                study = STUDIES.get(mapping["study"])
                if study is None:
                    raise ValueError(f"Unknown variant study: {variant['id']}")
                if (
                    not mapping.get("methods")
                    or not mapping.get("scenarios")
                    or not mapping.get("scope")
                    or not set(mapping["methods"]) <= set(study.methods)
                    or not set(mapping["scenarios"]) <= set(study.scenarios)
                ):
                    raise ValueError(
                        f"Invalid method/scenario study mapping: {variant['id']}"
                    )
                required_studies.add(study.id)
            if variant.get("source_review") == "verified-method-source":
                evidence = variant.get("source_evidence", {})
                keys = variant.get("citation_keys", [])
                if (
                    not keys
                    or not evidence.get("locator")
                    or not evidence.get("review")
                    or not all(
                        (
                            repository / ".context/reference" / key / "README.md"
                        ).is_file()
                        for key in keys
                    )
                ):
                    raise ValueError(
                        f"Verified method source lacks evidence: {variant['id']}"
                    )
            variants.append(variant)
    variant_ids = [variant["id"] for variant in variants]
    if len(variant_ids) != len(set(variant_ids)):
        raise ValueError("Duplicate scientific variant identity")
    if not procedures and not ownership:
        ownership.append("scientific-procedure-inventory")
    missing = [variant["id"] for variant in variants if not variant["study_mappings"]]
    pending = [
        variant["id"]
        for variant in variants
        if variant.get("source_review") != "verified-method-source"
    ]
    # Legacy fixtures remain visibly incomplete until explicitly migrated.
    missing.extend(item["id"] for item in legacy_required if not item["studies"])
    pending.extend(
        item["id"]
        for item in legacy_required
        if item["citation_review"] != "verified-method-source"
    )
    reference_definitions = _reference_definitions(
        repository / "src/python/packages/statadic"
    )
    reference_mappings = [
        mapping
        for item in [*methods.values(), *variants]
        for mapping in item.get("reference_mappings", [])
    ]
    mapped_definitions = {
        node.split("[", 1)[0]
        for mapping in reference_mappings
        for node in mapping["nodeids"]
    }
    return {
        "public_symbols": len(inventory),
        "method_records": len(methods),
        "scientific_procedures": len(procedures),
        "scientific_variants": len(variants),
        "evaluation_required": len(variants) + len(legacy_required),
        "ownership_review_pending": ownership,
        "coverage_status": "incomplete"
        if ownership or missing or pending
        else "registration-complete-evaluation-unassessed",
        "registered_study_methods": sum(
            bool(variant["study_mappings"]) for variant in variants
        ),
        "missing_study": missing,
        "source_review_pending": pending,
        "required_studies": sorted(required_studies),
        "execution_status": "not-inferred-from-registration",
        "reference_index": {
            "scientific_variants": len(variants),
            "mapped_scientific_variants": sum(
                bool(item.get("reference_mappings")) for item in variants
            ),
            "unmapped_scientific_variants": [
                item["id"] for item in variants if not item.get("reference_mappings")
            ],
            "method_records": len(methods),
            "mapped_method_records": sum(
                bool(item.get("reference_mappings")) for item in methods.values()
            ),
            "unmapped_method_records": [
                item["id"]
                for item in methods.values()
                if not item.get("reference_mappings")
            ],
            "execution_status": "not-inferred-from-mappings",
            "reference_definitions": len(reference_definitions),
            "mapped_reference_definitions": len(
                reference_definitions & mapped_definitions
            ),
            "unmapped_reference_definitions": sorted(
                reference_definitions - mapped_definitions
            ),
        },
    }
