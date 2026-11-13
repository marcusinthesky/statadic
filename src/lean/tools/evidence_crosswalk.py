"""List exact Python-to-Lean mappings and reproducible numerical comparisons."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from bridge import ROOT, fresh_output, read


def validate_comparisons(root: Path, comparisons: dict) -> None:
    """Fail if a documented test is removed, renamed or lacks its comparison details."""
    for identifier, case in comparisons.items():
        for key in (
            "symbols",
            "pytest",
            "reference",
            "quantity",
            "tolerance",
            "boundary",
        ):
            if not case.get(key):
                raise ValueError(f"{identifier}: missing {key}")
        filename, name = case["pytest"].split("::")
        tree = ast.parse((root / filename).read_bytes())
        if not any(
            isinstance(n, ast.FunctionDef) and n.name == name for n in tree.body
        ):
            raise ValueError(f"{identifier}: unknown pytest function {case['pytest']}")


def crosswalk(public: dict, claims: list[dict], comparisons: dict) -> list[dict]:
    """Collect exports and class members sharing exactly the same resolved target."""
    rows = {}
    names = {}
    for module in public["modules"]:
        for export in module["exports"]:
            name = f"{module['name']}.{export['name']}"
            entries = [(name, export)] + [
                (f"{name}.{member['name']}", member)
                for member in export.get("members", [])
            ]
            for alias, item in entries:
                target = item["target"]
                names[alias] = target
                row = rows.setdefault(
                    target, {"public_names": [], "claims": set(), "comparisons": set()}
                )
                row["public_names"].append(
                    {"name": alias, "kind": item["kind"], "family": item["method_id"]}
                )

    def bind(symbol: str, field: str, identifier: str) -> None:
        target = names.get(symbol, symbol)
        if target not in rows:
            raise ValueError(
                f"{identifier}: symbol absent from public inventory: {symbol}"
            )
        rows[target][field].add(identifier)

    for claim in claims:
        if claim["proof_status"] == "kernel-checked-and-axiom-audited":
            for symbol in claim["python"]["symbols"]:
                bind(symbol, "claims", claim["id"])
    for identifier, case in comparisons.items():
        for symbol in case["symbols"]:
            bind(symbol, "comparisons", identifier)

    return [
        {
            "target": target,
            "public_names": sorted(
                row["public_names"], key=lambda alias: alias["name"]
            ),
            "claims": sorted(row["claims"]),
            "comparisons": sorted(row["comparisons"]),
        }
        for target, row in sorted(rows.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", help="look up one qualified public name or target")
    args = parser.parse_args()
    proof = fresh_output()
    public = read(ROOT / "src/lean/registry/public-api.json")
    comparisons = read(ROOT / "src/lean/registry/implementation-comparisons.json")[
        "comparisons"
    ]
    validate_comparisons(ROOT, comparisons)
    rows = crosswalk(public, proof["claims"], comparisons)
    if args.symbol:
        rows = [
            row
            for row in rows
            if row["target"] == args.symbol
            or any(alias["name"] == args.symbol for alias in row["public_names"])
        ]
        if not rows:
            parser.error(f"unknown public symbol {args.symbol}")
    used = {identifier for row in rows for identifier in row["comparisons"]}
    result = {
        "schema_version": 1,
        "package_version": public["package_version"],
        "public_targets": rows,
        "comparison_definitions": {key: comparisons[key] for key in sorted(used)},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
