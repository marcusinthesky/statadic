"""Source-bound research inventory, not a proof of Python/JAX semantics.

The exact Lean masking theorem is one completed mathematical subproblem. These
bindings record the actual source functions that a future refinement must cover.
Only explicit `pin` accepts a changed AST; `check` never repairs the record.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PINS = ROOT / "src/lean/registry/implementation-pins.json"
SUBJECTS = {
    "masked_sum": ("sample/packed.py", "masked_sum"),
    "pair_chunks": ("sample/packed.py", "pair_chunks"),
    "mmd_squared_matrix": ("discrepancy/mmd/matrix.py", "mmd_squared_matrix"),
    "empirical_mmd": ("discrepancy/mmd/scalar.py", "_mmd_squared_core"),
    "shared_reference": ("discrepancy/mmd/scalar.py", "_shared_reference"),
    "linear_mmd_cross": ("discrepancy/mmd/weighted.py", "_linear_mmd_cross"),
}
GATES = [
    "source-operation semantics and array broadcasting",
    "pytree batching and inactive atom masking",
    "packed/scalar MMD exact equivalence",
    "IEEE rounding, overflow and numerical error",
    "XLA lowering, compiler and runtime trust",
]


def source_bindings(root: Path = ROOT) -> dict:
    """Bind function bodies/signatures without treating comments as semantics."""
    records = {}
    for identifier, (relative, name) in SUBJECTS.items():
        path = root / "src/python/packages/statadic/src/statadic" / relative
        tree = ast.parse(path.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        records[identifier] = {
            "source": path.relative_to(root).as_posix(),
            "function": name,
            "ast_sha256": hashlib.sha256(
                ast.dump(function, include_attributes=False).encode()
            ).hexdigest(),
        }
    return records


def check(root: Path = ROOT) -> dict:
    """Reject changed implementations; retain unresolved proof gates explicitly."""
    data = json.loads((root / PINS.relative_to(ROOT)).read_text())
    if data["bindings"] != source_bindings(root):
        raise ValueError("Implementation source changed; review correspondence pins")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "pin"))
    args = parser.parse_args()
    if args.command == "pin":
        PINS.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "source-bound-research-prototype",
                    "exact_mask_theorem": "StatadicFormal.Discrepancy.MMD.maskedReduction_eq_leading",
                    "bindings": source_bindings(),
                    "unresolved_gates": GATES,
                    "python_formally_verified": False,
                },
                indent=2,
            )
            + "\n"
        )
    print(json.dumps(check(), indent=2))


if __name__ == "__main__":
    main()
