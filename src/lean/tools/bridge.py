"""Join reviewed literature mappings to fresh compiled Lean evidence.

Authored metadata never supplies proof status. All generated outputs are ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from integrity import certify as audit_lean
from inventory import inventory

ROOT = Path(__file__).resolve().parents[3]
LEAN = ROOT / "src/lean"
REGISTRY = LEAN / "registry"
OUTPUT = LEAN / "build/certification.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def tracked_inputs(root: Path = ROOT) -> dict[str, str]:
    """Bind mappings, references and recursive proof-tool/docs implementation."""
    paths = [
        *sorted((root / "src/lean/registry").glob("*.json")),
        *sorted((root / ".context/reference").glob("*/README.md")),
        *sorted((root / ".context/reference").glob("*/source.json")),
        *sorted((root / "src/lean/tools").rglob("*.py")),
        root / "src/latex/refs.bib",
        root / "src/python/apps/docs/src/statadic_docs_labs/formal_extension.py",
        *sorted((root / "src/python/apps/docs/src/statadic_docs_labs").rglob("*.py")),
    ]
    comparisons = root / "src/lean/registry/implementation-comparisons.json"
    if comparisons.is_file():
        paths.extend(
            root / case["pytest"].split("::", 1)[0]
            for case in read(comparisons)["comparisons"].values()
        )
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(set(paths))}


def resolve_symbol(name: str) -> None:
    """Resolve a qualified object without hiding a failed import of its owner."""
    parts = name.split(".")
    for count in range(len(parts), 0, -1):
        module = ".".join(parts[:count])
        try:
            obj = importlib.import_module(module)
        except ModuleNotFoundError as error:
            if error.name != module and not module.startswith(f"{error.name}."):
                raise
            continue
        for part in parts[count:]:
            obj = getattr(obj, part)
        return
    raise ValueError(f"unresolvable Python symbol: {name}")


def validate_registry(root: Path = ROOT) -> tuple[list[dict], dict]:
    """Validate structure and provenance fields; scholarly review is separate."""
    import jsonschema
    from pybtex.database import parse_file

    registry = root / "src/lean/registry"
    data = read(registry / "claims.json")
    jsonschema.validate(data, read(registry / "schema.json"))
    claims = data["claims"]
    ids = [c["id"] for c in claims]
    declarations = [c["lean"]["declaration"] for c in claims]
    if len(set(ids)) != len(ids) or len(set(declarations)) != len(declarations):
        raise ValueError("duplicate claim ID or principal declaration")
    bib = parse_file(str(root / "src/latex/refs.bib"))
    for claim in claims:
        source = claim.get("source")
        if source is not None:
            key = source["key"]
            if key not in bib.entries:
                raise ValueError(f"{claim['id']}: missing bibliography key {key}")
            record = root / ".context/reference" / key
            if not (record / "source.json").is_file():
                raise ValueError(f"{claim['id']}: missing source dossier")
            if not source["locator"].strip() or not source["proof_locator"].strip():
                raise ValueError(
                    f"{claim['id']}: missing precise source/proof location"
                )
        page = (
            root / "src/python/packages/statadic/docs/formal/claims" / f"{claim['id']}.md"
        )
        if (
            not page.is_file()
            or f"<!-- statadic:claim {claim['id']} -->" not in page.read_text()
        ):
            raise ValueError(
                f"{claim['id']}: missing canonical documentation projection"
            )
        for symbol in claim["python"]["symbols"]:
            resolve_symbol(symbol)
    snapshot = read(registry / "public-api.json")
    if snapshot != inventory(
        root / "src/python/packages/statadic/src/statadic", root / "src/python/packages/statadic"
    ):
        raise ValueError(
            "public API snapshot is stale; explicitly run lean::inventory-refresh"
        )
    methods = read(registry / "methods.json")
    validate_methods(methods["methods"])
    method_ids = [m["id"] for m in methods["methods"]]
    if len(set(method_ids)) != len(method_ids):
        raise ValueError("duplicate method family ID")
    for module in snapshot["modules"]:
        if module["method_id"] not in method_ids:
            raise ValueError(f"uncovered public module: {module['name']}")
        for export in module["exports"]:
            if export["method_id"] not in method_ids or export["kind"] == "unresolved":
                raise ValueError(
                    f"unreviewed public export: {module['name']}.{export['name']}"
                )
    for claim in claims:
        if not set(claim["methods"]) <= set(method_ids):
            raise ValueError(f"{claim['id']}: unknown method association")
    families_by_claim = {claim["id"]: claim["methods"] for claim in claims}
    for method in methods["methods"]:
        for obligation in method["obligations"]:
            for identifier in obligation.get("related_claims", []):
                if method["id"] not in families_by_claim.get(identifier, []):
                    raise ValueError(
                        f"{obligation['id']}: unrelated or unknown claim {identifier}"
                    )
    return claims, methods


def native_nodes(lean: Path = LEAN) -> list[dict]:
    subprocess.run(
        [
            "lake",
            "--no-cache",
            "--wfail",
            "build",
            "StatadicFormal:blueprintJson",
            "StatadicFormal:blueprint",
        ],
        cwd=lean,
        env=os.environ | {"LAKE_NO_CACHE": "true"},
        check=True,
    )
    path = lean / ".lake/build/blueprint/library/StatadicFormal.json"
    data = json.loads(path.read_text())
    # Library JSON contains module references; module JSON owns actual nodes.
    if not isinstance(data, dict) or not data.get("modules"):
        raise ValueError("missing or malformed native blueprint library export")
    nodes = []
    for module in data["modules"]:
        module_path = lean / ".lake/build/blueprint" / module
        for item in json.loads(module_path.read_text()):
            if item.get("type") == "node":
                node = item["data"]
                module_name = node["location"]["module"]
                artifact = (
                    lean
                    / ".lake/build/blueprint/module"
                    / Path(*module_name.split("."))
                ).with_suffix(".artifacts") / f"{node['latexLabel']}.tex"
                tex = artifact.read_text()
                statement = node["statement"]["text"]
                proof = node["proof"]["text"]
                if (
                    not statement.strip()
                    or not proof.strip()
                    or statement not in tex
                    or proof not in tex
                ):
                    raise ValueError(
                        f"native LaTeX artifact disagrees with its compiled annotation: {artifact}"
                    )
                # Slice the generated TeX itself, with compiled metadata locating each part.
                node["latex"] = {
                    "statement": tex[
                        tex.index(statement) : tex.index(statement) + len(statement)
                    ],
                    "proof": tex[tex.index(proof) : tex.index(proof) + len(proof)],
                    "artifact": artifact.relative_to(lean).as_posix(),
                    "sha256": digest(artifact),
                }
                nodes.append(node)
    if not nodes:
        raise ValueError("empty native blueprint node export")
    return nodes


def validate_claims(
    claims: list[dict], compiled: dict, nodes: list[dict]
) -> list[dict]:
    records = {d["declaration"]: d for d in compiled["declarations"]}
    labels = {}
    for node in nodes:
        label = node.get("latexLabel")
        if label in labels:
            raise ValueError(f"duplicate native blueprint label: {label}")
        labels[label] = node
    if set(labels) != {c["id"] for c in claims}:
        raise ValueError("registry IDs and native blueprint labels differ")
    result = []
    for claim in claims:
        node = labels[claim["id"]]
        declaration = claim["lean"]["declaration"]
        if node.get("name") != declaration:
            raise ValueError(f"{claim['id']}: native declaration mismatch")
        if node.get("notReady") is not False or node.get("hasLean") is not True:
            raise ValueError(f"{claim['id']}: native annotation is not ready")
        record = records.get(declaration)
        if (
            not record
            or not record["is_theorem"]
            or record["module"] != claim["lean"]["module"]
        ):
            raise ValueError(
                f"{claim['id']}: principal theorem missing from fresh compiled census"
            )
        statement = re.sub(r"\s+", " ", record["statement_type"]).strip()
        binding = {
            "declaration": declaration,
            "statement_type": statement,
            "source": claim.get("source"),
            "kind": claim.get("kind", "published-specialization"),
            "review": claim.get("review"),
            "scope": claim["scope"],
            "latex_statement": node["latex"]["statement"],
            "latex_proof": node["latex"]["proof"],
            # Conservative dependency closure: changing a referenced definition
            # need not change the pretty-printed theorem type. Bind the complete
            # audited first-party source graph and pinned external environment.
            # This deliberately also requests review after unrelated Lean edits.
            "definition_environment": compiled["source_hashes"],
        }
        pin = hashlib.sha256(
            json.dumps(binding, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        result.append(
            claim
            | {
                "proof_status": "kernel-checked-and-axiom-audited",
                "statement_type": statement,
                "axioms": record["axioms"],
                "statement_pin": pin,
                "latex": node["latex"],
                "review_dependency_scope": "all-audited-first-party-sources-and-pins",
            }
        )
    return result


def manuscript_list(items: list[str]) -> str:
    """Quote registry prose as literal LaTeX text in a Unicode manuscript."""
    escapes = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "&": r"\&",
        "#": r"\#",
        "_": r"\_",
        "%": r"\%",
        "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}",
    }
    # Registry prose also uses mathematical Unicode outside of TeX environments.
    math = {
        "α": r"\alpha",
        "β": r"\beta",
        "δ": r"\delta",
        "ε": r"\varepsilon",
        "η": r"\eta",
        "λ": r"\lambda",
        "ρ": r"\rho",
        "σ": r"\sigma",
        "φ": r"\phi",
        "β̂": r"\hat\beta",
        "Ĝ": r"\hat G",
        "√n": r"\sqrt{n}",
        "√N": r"\sqrt{N}",
        "⁻¹": "{}^{-1}",
        "⁻⁸": "{}^{-8}",
        "¹": "{}^1",
        "²": "{}^2",
        "³": "{}^3",
        "ᵀ": "{}^T",
        "ᵢ": "{}_i",
        "ⱼ": "{}_j",
        "₀": "{}_0",
        "×": r"\times",
        "−": "-",
        "≈": r"\approx",
        "≥": r"\ge",
        "≤": r"\le",
        "⟨": r"\langle",
        "⟩": r"\rangle",
    }
    escapes.update({c: r"\ensuremath{" + tex + "}" for c, tex in math.items()})
    # Match superscripts and accented letters as a unit before their constituents.
    tokens = re.compile(
        "|".join(map(re.escape, sorted(escapes, key=len, reverse=True)))
    )
    return "\n".join(
        [
            r"\begin{itemize}",
            *(r"\item " + tokens.sub(lambda m: escapes[m[0]], item) for item in items),
            r"\end{itemize}",
        ]
    )


def manuscript_fragments(claims: list[dict], destination: Path) -> dict[str, str]:
    """Export native mathematical exposition without blueprint environments.

    Call only on the fresh certified join. A manuscript owns theorem numbering
    and labels; these fragments own neither. The proof is reviewed exposition,
    not an automatic translation of a Lean proof term.
    """
    destination.mkdir(parents=True, exist_ok=True)
    for old in destination.glob("STATADIC-*-*.tex"):
        old.unlink()
    macros = ["% Native statements and proofs with reviewed registry prose."]
    parts = ("statement", "proof", "assumptions", "diagnostics")
    for claim in claims:
        identifier = claim["id"]
        if not re.fullmatch(r"STATADIC-[A-Z]+-[0-9]{3}", identifier):
            raise ValueError(f"Invalid manuscript claim identifier: {identifier}")
        fragments = {
            "statement": claim["latex"]["statement"],
            "proof": claim["latex"]["proof"],
            "assumptions": manuscript_list(claim["scope"]["assumptions"]),
            "diagnostics": manuscript_list(claim["scope"]["diagnostics"]),
        }
        for part in parts:
            value = fragments[part]
            (destination / f"{identifier}-{part}.tex").write_text(value + "\n")
            macros.append(
                rf"\expandafter\def\csname statadic@{part}@{identifier}\endcsname{{{value}}}"
            )
    # Self-contained macro registry: loading works from any manuscript cwd.
    for part in parts:
        macros.append(
            rf"\newcommand{{\statadic{part}}}[1]{{\ifcsname statadic@{part}@#1\endcsname"
            rf"\csname statadic@{part}@#1\endcsname\else"
            rf"\PackageError{{statadic}}{{Unknown {part} #1}}{{Regenerate certified fragments}}\fi}}"
        )
    (destination / "statadic-mathematics.tex").write_text("\n".join(macros) + "\n")
    return {path.name: digest(path) for path in sorted(destination.glob("*.tex"))}


def certify(*, pin: bool = False) -> dict:
    """Build, inspect and bind claims; no status survives stale source inputs."""
    OUTPUT.unlink(missing_ok=True)
    claims, methods = validate_registry()
    before = tracked_inputs()
    subprocess.run([sys.executable, str(LEAN / "check_setup.py")], cwd=LEAN, check=True)
    compiled = audit_lean(LEAN)
    joined = validate_claims(claims, compiled, native_nodes())
    from integrity import discover, snapshot

    if compiled["source_hashes"] != snapshot(LEAN, discover(LEAN)):
        raise ValueError("Lean source changed during blueprint extraction")
    if before != tracked_inputs():
        raise ValueError("registry or references changed during certification")
    pins = {c["id"]: c["statement_pin"] for c in joined}
    pin_file = REGISTRY / "statement-pins.json"
    if pin:
        write(pin_file, {"schema_version": 1, "pins": pins})
    elif read(pin_file)["pins"] != pins:
        raise ValueError(
            "statement/source mapping changed; review then run lean::statement-pin"
        )
    output = {
        "schema_version": 1,
        "trust": "Pinned Lean kernel and dependencies; propext, Classical.choice, Quot.sound only",
        "claims": joined,
        "compiled": compiled,
        "inputs": tracked_inputs(),
        "methods": methods["methods"],
    }
    pending = unresolved_obligations(output)
    output["open_obligation_ids"] = [o["id"] for _, o in pending]
    output["programme_complete"] = not pending
    output["manuscript"] = manuscript_fragments(joined, LEAN / "build/manuscript")
    write(OUTPUT, output)
    return output


def fresh_output(root: Path = ROOT) -> dict:
    output = read(root / "src/lean/build/certification.json")
    if output["inputs"] != tracked_inputs(root):
        raise ValueError("formal certification is stale; run just lean::certify")
    if "manuscript" in output:
        actual = {
            p.name: digest(p)
            for p in sorted((root / "src/lean/build/manuscript").glob("*.tex"))
        }
        if output["manuscript"] != actual:
            raise ValueError("Generated manuscript evidence changed")
    lean = root / "src/lean"
    from integrity import discover, snapshot

    if output["compiled"]["source_hashes"] != snapshot(lean, discover(lean)):
        raise ValueError("compiled proof evidence is stale; run just lean::certify")
    api = read(lean / "registry/public-api.json")
    if api != inventory(
        root / "src/python/packages/statadic/src/statadic", root / "src/python/packages/statadic"
    ):
        raise ValueError("Python source changed after formal certification")
    for claim in output["claims"]:
        artifact = lean / claim["latex"]["artifact"]
        if not artifact.is_file() or digest(artifact) != claim["latex"]["sha256"]:
            raise ValueError("generated LeanArchitect LaTeX evidence is stale")
    return output


def validate_methods(methods: list[dict]) -> None:
    if not methods:
        raise ValueError("empty method registry")
    ids = []
    for method in methods:
        if not method.get("obligations"):
            raise ValueError("method has no obligations")
        for obligation in method["obligations"]:
            identifier = obligation.get("id")
            if not identifier:
                raise ValueError("missing obligation ID")
            if identifier in ids:
                raise ValueError("duplicate obligation ID")
            ids.append(identifier)


def unresolved_obligations(data: dict) -> list[tuple[str, dict]]:
    """Unresolved method obligations never become passes through partial coverage."""
    validate_methods(data["methods"])
    checked = {
        c["id"]: c
        for c in data["claims"]
        if c["proof_status"] == "kernel-checked-and-axiom-audited"
    }
    pending = []
    for method in data["methods"]:
        for obligation in method["obligations"]:
            required = set(obligation.get("claims", []))
            mapped = all(
                identifier in checked
                and method["id"] in checked[identifier].get("methods", [])
                and obligation.get("id") in checked[identifier].get("obligations", [])
                for identifier in required
            )
            if (
                obligation.get("source_status") != "source-matched"
                or obligation.get("scope_review") != "agent-reviewed"
                or not required
                or not mapped
            ):
                pending.append((method["id"], obligation))
    return pending


def coverage(data: dict) -> list[str]:
    return [
        f"{method}: {item['title']}" for method, item in unresolved_obligations(data)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["certify", "pin", "check", "coverage"])
    args = parser.parse_args()
    try:
        data = (
            certify(pin=args.command == "pin")
            if args.command in {"certify", "pin"}
            else fresh_output()
        )
        print(
            f"{len(data['claims'])} literature mappings; {len(data['compiled']['declarations'])} compiled declarations; {len(data['methods'])} method families"
        )
        if args.command == "coverage":
            pending = coverage(data)
            if pending:
                print(
                    "Programme incomplete; unresolved obligations:\n"
                    + "\n".join(pending)
                )
                raise SystemExit(1)
    except (
        ValueError,
        OSError,
        subprocess.CalledProcessError,
        KeyError,
        AttributeError,
    ) as error:
        if args.command in {"certify", "pin"}:
            OUTPUT.unlink(missing_ok=True)
        raise SystemExit(f"formal bridge: {error}") from error


if __name__ == "__main__":
    main()
