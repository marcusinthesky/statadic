"""Reproduce the frozen alpha review's read-only inventory and evidence census."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

from statadic_studies.contracts import STUDIES
from statadic_studies.study_types import metrics_for

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
CANDIDATE = "17fdc5410de31356df72d6fc6459853ca39260f9"
BASE = "52268aa5078d94c9bf8b092c7655fbf3ae54031a"


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def write_csv(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def owner(path: str) -> str:
    lower = path.lower()
    if any(
        part in lower for part in ("spatial", "survival", "panel", "graph_", "graph-")
    ):
        return "dependent-models"
    if any(
        part in lower
        for part in ("gmm", "gel", "moments", "inference/iv", "estimation/iv")
    ):
        return "moments"
    if any(
        part in lower
        for part in (
            "linear",
            "regression",
            "glm",
            "likelihood",
            "diagnostic",
            "longrun",
        )
    ):
        return "linear-conditional"
    if any(
        part in lower
        for part in (
            "geometry",
            "discrepancy",
            "sample/",
            "association",
            "resampling",
            "mmd",
            "energy",
            "transport",
            "barycentre",
            "median",
            "sharpe",
            "equivalence",
            "multiplicity",
            "permutation",
        )
    ):
        return "measures"
    if any(
        part in path
        for part in (
            "src/lean/",
            "src/statadic/core/",
            "tests/core/",
            "tests/contracts/",
            "src/statadic/operators/",
            "src/statadic/optimize/",
            "src/statadic/api/base",
            "src/statadic/api/roles",
            "src/statadic/api/inputs",
            "src/statadic/api/prediction",
            "src/statadic/api/data",
            "src/statadic/api/design",
            "src/statadic/api/inference",
        )
    ):
        return "shared-contracts"
    if path.startswith(
        ("src/python/", "src/latex/", ".context/", "insitu/")
    ) or path in ("README.md", "ARCHITECTURE.md", "CONTRIBUTING.md", "CHANGELOG.md"):
        return "delivery"
    return "infrastructure"


def main() -> None:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if head != CANDIDATE:
        raise SystemExit(
            "Review baseline changed; explicitly adjudicate a new candidate"
        )
    public = read("src/lean/registry/public-api.json")
    catalogue = read("src/python/apps/studies/catalogue.json")
    claims = read("src/lean/registry/claims.json")["claims"]
    comparisons = read("src/lean/registry/implementation-comparisons.json")[
        "comparisons"
    ]
    entries = []
    for module in public["modules"]:
        for export in module["exports"]:
            name = module["name"] + "." + export["name"]
            entries.append((name, export))
            entries.extend(
                (name + "." + member["name"], member)
                for member in export.get("members", [])
            )
    targets = {name: item["target"] for name, item in entries}
    by_target: dict[str, list[str]] = {}
    for name, target in targets.items():
        by_target.setdefault(target, []).append(name)
    claim_links: dict[str, set[str]] = {}
    comparison_links: dict[str, set[str]] = {}
    for claim in claims:
        for symbol in claim["python"]["symbols"]:
            claim_links.setdefault(targets.get(symbol, symbol), set()).add(claim["id"])
    for identifier, case in comparisons.items():
        for symbol in case["symbols"]:
            comparison_links.setdefault(targets.get(symbol, symbol), set()).add(
                identifier
            )
    write_csv(
        "api-evidence.csv",
        [
            {
                "target": target,
                "public_names": ";".join(sorted(names)),
                "declared_claims": ";".join(sorted(claim_links.get(target, []))),
                "curated_comparisons": ";".join(
                    sorted(comparison_links.get(target, []))
                ),
                "scope": "exact links only; no inherited or transitive qualification",
            }
            for target, names in sorted(by_target.items())
        ],
    )
    scope = json.loads((OUT / "scope.json").read_text())
    dispositions = scope["procedures"]
    assert set(dispositions) == {p["id"] for p in catalogue["scientific_procedures"]}
    assert set(scope["direct_variant_findings"]) <= {
        variant["id"]
        for procedure in catalogue["scientific_procedures"]
        for variant in procedure["variants"]
    }
    variant_rows = []
    for procedure in catalogue["scientific_procedures"]:
        for variant in procedure["variants"]:
            selected_metrics = set()
            for mapping in variant.get("study_mappings", []):
                definition = STUDIES[mapping["study"]].record()
                for method in mapping["methods"]:
                    selected_metrics.update(metrics_for(definition, method))
            variant_rows.append(
                {
                    "procedure": procedure["id"],
                    "variant": variant["id"],
                    "family": procedure["family"],
                    "selectors": json.dumps(variant["selectors"], sort_keys=True),
                    "python_targets": ";".join(variant["python_symbols"]),
                    "citations": ";".join(variant.get("citation_keys", [])),
                    "source_status": variant.get("source_review", "unknown"),
                    "adapter_status": variant.get("adapter_status", "unknown"),
                    "evaluation_status": variant.get("evaluation_status", "unknown"),
                    "alpha_recommendation": dispositions[procedure["id"]][
                        "recommendation"
                    ],
                    "procedure_workstream_items": ";".join(
                        dispositions[procedure["id"]]["workflow_items"]
                    ),
                    "direct_variant_findings": ";".join(
                        scope["direct_variant_findings"].get(variant["id"], [])
                    ),
                    "declared_mc_metrics": ";".join(sorted(selected_metrics)),
                    "study_mappings": json.dumps(
                        variant.get("study_mappings", []), sort_keys=True
                    ),
                    "coverage_limit": "catalogue membership is not exhaustive semantic coverage or execution",
                }
            )
    write_csv("scientific-evidence.csv", variant_rows)
    write_csv(
        "claim-evidence.csv",
        [
            {
                "claim": claim["id"],
                "kind": claim["kind"],
                "module": claim["lean"]["module"],
                "declaration": claim["lean"]["declaration"],
                "python_symbols": ";".join(claim["python"]["symbols"]),
                "body_review": "principal body read; no fresh compilation",
                "source_review": claim.get("source", {}).get(
                    "review", claim.get("review", "pending")
                ),
                "correspondence": "bounded chains only; see formal-review.md",
                "human_review": "pending",
            }
            for claim in claims
        ],
    )
    raw = (
        subprocess.check_output(
            ["git", "diff", "--name-status", "-z", BASE + "..." + CANDIDATE], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    files = []
    inspections = json.loads((OUT / "reviews.json").read_text())["reviews"]
    inspected = {}
    for inspection in inspections:
        paths = list(inspection["paths"])
        if inspection.get("claims") == "all-current-principal-claim-bodies":
            paths.extend(
                "src/lean/packages/StatadicFormal/"
                + claim["lean"]["module"].replace(".", "/")
                + ".lean"
                for claim in claims
            )
        for path in paths:
            inspected.setdefault(path, []).append(inspection["id"])
    index = 0
    while index < len(raw) - 1:
        status, path = raw[index : index + 2]
        index += 2
        old_path = ""
        if status.startswith(("R", "C")):
            old_path, path = path, raw[index]
            index += 1
        binary = Path(path).suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".pdf",
            ".zip",
            ".woff",
            ".woff2",
            ".ttf",
            ".tar",
            ".ico",
        }
        retained = "/results/" in path or "/history/" in path
        generated = (
            path.endswith("refs.bib")
            or path.startswith("src/lean/registry/")
            or "/docs/formal/claims/" in path
        )
        files.append(
            {
                "path": path,
                "base_path": old_path,
                "change": status,
                "primary_slice": owner(path),
                "kind": "binary-artifact"
                if binary
                else "retained-evidence"
                if retained
                else "generated-or-source-bound"
                if generated
                else "authored",
                "sha256": sha(ROOT / path) or "deleted",
                "review_status": "inspected-see-log"
                if path in inspected
                else "pending-primary-review",
                "inspection_records": ";".join(inspected.get(path, [])),
            }
        )
    write_csv("changed-paths.csv", files)
    logs = []
    for path in sorted((ROOT / ".tmp/chat_logs").glob("*.md")):
        logs.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha(path),
                "bytes": path.stat().st_size,
                "lines": len(path.read_text().splitlines()),
            }
        )
    mismatches = [
        path
        for path, expected in public["source_manifest"].items()
        if sha(ROOT / "src/python/packages/statadic/src/statadic" / path) != expected
    ]
    summary = {
        "candidate": CANDIDATE,
        "base": BASE,
        "public_modules": len(public["modules"]),
        "public_names": len(entries),
        "resolved_targets": len(by_target),
        "public_source_manifest_mismatches": mismatches,
        "catalogue_owners": len(catalogue["methods"]),
        "scientific_procedures": len(catalogue["scientific_procedures"]),
        "catalogued_variants": len(variant_rows),
        "source_statuses": dict(Counter(row["source_status"] for row in variant_rows)),
        "citation_linked_variants": sum(bool(row["citations"]) for row in variant_rows),
        "study_mapped_variants": sum(
            row["study_mappings"] != "[]" for row in variant_rows
        ),
        "claims": len(claims),
        "exact_claim_linked_targets": len(claim_links),
        "curated_comparisons": len(comparisons),
        "exact_comparison_targets": len(comparison_links),
        "changed_paths": len(files),
        "slice_assignment_counts": dict(Counter(row["primary_slice"] for row in files)),
        "chat_logs": logs,
        "limits": [
            "Counts describe metadata links, not proof coverage or scientific qualification.",
            "Every path is assigned; pending status is not a completed review.",
            "Known semantic inventory omissions are recorded as findings, not silently added to source registries.",
            "No tests, certification or campaigns run by this audit.",
        ],
    }
    summary["registered_study_designs"] = len(STUDIES)
    summary["path_inspection_counts"] = dict(
        Counter(row["review_status"] for row in files)
    )
    summary["variant_declared_mc_metric_counts"] = {
        metric: sum(
            metric in row["declared_mc_metrics"].split(";") for row in variant_rows
        )
        for metric in ("bias", "mse", "size", "power", "coverage")
    }
    summary["registered_design_metric_counts"] = {
        metric: sum(
            any(
                metric in metrics_for(study.record(), method)
                for method in study.methods
            )
            for study in STUDIES.values()
        )
        for metric in ("bias", "mse", "size", "power", "coverage")
    }
    (OUT / "study-designs.json").write_text(
        json.dumps(
            {identifier: study.record() for identifier, study in STUDIES.items()},
            indent=2,
        )
        + "\n"
    )
    (OUT / "baseline.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "chat_logs"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
