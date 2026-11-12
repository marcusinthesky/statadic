"""Render literature mappings only from a current compiled certification artifact."""

from __future__ import annotations

import hashlib
import json
import re
import runpy
import shutil
import subprocess
import sys
import textwrap
from contextlib import redirect_stdout
from html import escape
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from docutils.nodes import make_id

from statadic_docs_labs.study_plots import operating_plot, operating_points

ROOT = Path(__file__).resolve().parents[6]


def bullet(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def native_math(value: str) -> str:
    """Render native LeanArchitect display delimiters as MyST display math."""
    return re.sub(
        r"\\\[(.*?)\\\]",
        lambda m: f"\n\n$$\n{m.group(1).strip()}\n$$\n\n",
        value,
        flags=re.DOTALL,
    )


def math_excerpt(claim: dict, *, link: bool = False) -> str:
    """Show the certified native statement and its annotated proof."""
    latex = claim["latex"]
    statement = native_math(latex["statement"])
    proof = native_math(latex["proof"])
    result = f"**Statement.** {statement}\n\n**Proof.** {proof}"
    if link:
        result += f"\n\n{{ref}}`Full claim and assumptions <formal-{claim['id']}>`"
    return result


def claim_page(claim: dict) -> str:
    source = claim.get("source")
    scope = claim["scope"]
    lean = claim["lean"]
    links = (
        "\n".join(f"- {{py:obj}}`{symbol}`" for symbol in claim["python"]["symbols"])
        or "No Python implementation is claimed for this result."
    )
    lean_path = "/".join(lean["module"].split(".")) + ".lean"
    lean_href = escape(f"../downloads/lean/{lean_path}", quote=True)
    latex_href = escape(f"../downloads/latex/{claim['id']}.tex", quote=True)
    statement_type = textwrap.fill(
        claim["statement_type"],
        width=100,
        break_long_words=False,
        break_on_hyphens=False,
    )
    provenance = (
        f"{{cite:t}}`{source['key']}` — {source['locator']}.\n\n"
        f"[Read the source]({source['url']}). "
        f"Proof location: {source['proof_locator']}.\n\n"
        f"Source correspondence: {source['review']}.\n\n"
        f"Claim review: {claim.get('review', 'not recorded')}."
        if source is not None
        else f"{claim['kind']}. Review: {claim['review']}."
    )
    return f"""(formal-{claim["id"]})=
# {claim["title"]}

**{claim["id"]}** · {claim["proof_status"]}

## Source and correspondence

{provenance}

{scope["relationship"]}

Kernel checking establishes the Lean
statement below; source fidelity is a separate review.

## Assumptions

{bullet(scope["assumptions"])}

## Applying this result

{bullet(scope["diagnostics"])}

## Conclusion

{scope["conclusion"]}

## Equations from the Lean companion

{math_excerpt(claim)}

## Mathematical argument in Lean

{bullet(scope["proof_steps"])}

Declaration: `{lean["declaration"]}`.

```{{raw}} html
<a href="{lean_href}">Read the Lean source and proof</a> ·
<a href="{latex_href}">Download Lean companion TeX</a> ·
<a href="../blueprint/index.html">Browse the proof blueprint</a> ·
<a href="../downloads/blueprint.pdf">Download the proof blueprint PDF</a>
```

```lean
{statement_type}
```

Transitive axioms: {", ".join(f"`{a}`" for a in claim["axioms"]) or "none"}.

## Relationship to STATADIC

{links}

{claim["python"]["relationship"]}. Numerical validation and statistical
qualification are not inferred from this theorem.

## Limits and remaining work

{bullet(scope["limitations"])}

[All formal results](../index.md) · [Bibliography](../../reference/bibliography.md)
"""


def index_page(data: dict) -> str:
    linked = {method for claim in data["claims"] for method in claim["methods"]}
    total = len(data["methods"])
    lines = [
        "## Checked mathematical results",
        "",
        f"{len(linked)} of {total} method families link to at least one checked claim "
        f"({len(linked) / total:.1%}). Each claim below lists its assumptions and "
        "the diagnostics relevant to applying it.",
        "",
        "| Result | Formalization scope |",
        "| --- | --- |",
    ]
    for claim in data["claims"]:
        lines.append(
            f"| [{claim['id']}: {claim['title']}](claims/{claim['id']}.md) | "
            f"{claim['scope']['relationship']} |"
        )
    open_ids = set(data["open_obligation_ids"])
    lines += [
        "",
        "## Coverage by obligation",
        "",
        f"{len(open_ids)} obligations remain open across the method inventory.",
        "",
        "| Method family | Obligation | Evidence |",
        "| --- | --- | --- |",
    ]
    for method in data["methods"]:
        for obligation in method["obligations"]:
            is_open = obligation["id"] in open_ids
            ids = obligation.get("related_claims" if is_open else "claims", [])
            links = ", ".join(f"[{claim}](claims/{claim}.md)" for claim in ids)
            evidence = links
            if is_open:
                evidence = f"Open · related results {links}" if links else "Open"
            scope = obligation.get("scope")
            title = obligation["title"] + (f". {scope}" if scope else "")
            lines.append(f"| {method['title']} | {title} | {evidence} |")
    lines += [
        "",
        "The public source inventory assigns exported symbols to these families. "
        "A family row does not mean its guarantees have been fully identified or "
        "proved. Sources and exact statements must be reviewed "
        "before each new proof batch.",
        "",
        "## Inspect and reproduce",
        "",
        "```bash",
        "just lean::certify",
        "just lean::programme-check",
        "```",
        "",
        "`programme-check` deliberately fails while required source or proof "
        "obligations remain open. The regular integrity check accepts honestly "
        "documented open work.",
        "",
        "```{raw} html",
        '<a href="downloads/claims.json">Claim mappings</a> · ',
        '<a href="downloads/public-api.json">Public API snapshot</a> · ',
        '<a href="downloads/public-evidence.json">Proofs by Python symbol</a> · ',
        '<a href="downloads/methods.json">Method obligations</a> · ',
        '<a href="downloads/certification.json">Compiled evidence</a> · ',
        '<a href="downloads/legacy-candidates.json">'
        "Original candidate inventory</a> · ",
        '<a href="downloads/refs.bib">BibTeX</a> · ',
        '<a href="blueprint/index.html">Proof blueprint</a> · ',
        '<a href="downloads/blueprint.pdf">Blueprint PDF</a>',
        "```",
        "",
        "The blueprint follows mathematical dependencies. Downloads record the version "
        "and source hashes used for this build.",
    ]
    return "\n".join(lines)


def load(app) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "src/lean/tools/bridge.py"), "check"],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    app._statadic_formal = json.loads(
        (ROOT / "src/lean/build/certification.json").read_text()
    )
    app._statadic_claims_by_symbol = claims_by_symbol(app._statadic_formal["claims"])
    app._statadic_research = []
    app._statadic_procedures = []
    app._statadic_evaluations = []
    app._statadic_reference_catalogue = None
    if app.config.statadic_study_exports:
        from statadic_studies.analysis import verify_export
        from statadic_studies.campaign import manifest

        for export_path in app.config.statadic_study_exports:
            retained = Path(export_path)
            report = verify_export(retained)
            frozen = manifest(retained / "campaign")
            app._statadic_evaluations.append(
                {"path": retained, "report": report, "frozen": frozen}
            )
            app.env.note_dependency(str(retained / "evidence-manifest.json"))
    path = app.config.statadic_research_catalogue
    if path is not None:
        catalogue = Path(path)
        if not catalogue.is_absolute():
            catalogue = Path(app.confdir) / catalogue
        data = json.loads(catalogue.read_text())
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported STATADIC research catalogue version")
        public = json.loads((ROOT / "src/lean/registry/public-api.json").read_text())
        symbols = {
            f"{module['name']}.{item['name']}"
            for module in public["modules"]
            for item in module["exports"]
        }
        symbols.update(
            item["target"]
            for module in public["modules"]
            for item in module["exports"]
            if item.get("target")
        )
        for module in public["modules"]:
            for item in module["exports"]:
                for member in item.get("members", []):
                    symbols.add(f"{module['name']}.{item['name']}.{member['name']}")
                    if member.get("target"):
                        symbols.add(member["target"])
        for method in data["methods"]:
            if not set(method["python_symbols"]) <= symbols:
                raise ValueError(f"Unknown catalogue Python symbol: {method['id']}")
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", method["id"]):
                raise ValueError(f"Invalid research method identity: {method['id']}")
        app._statadic_research = data["methods"]
        app._statadic_procedures = data.get("scientific_procedures", [])
        app._statadic_reference_catalogue = catalogue
        app.env.note_dependency(str(catalogue))
        for owner in reference_owners(app):
            for mapping in owner.get("reference_mappings", []):
                for node in mapping["nodeids"]:
                    app.env.note_dependency(
                        str(
                            ROOT
                            / "src/python/packages/statadic"
                            / node.split("::", 1)[0]
                        )
                    )


def claims_by_symbol(claims: list[dict]) -> dict[str, list[dict]]:
    """Index explicit qualified names, including registered aliases and members."""
    index: dict[str, list[dict]] = {}
    for claim in claims:
        for symbol in claim["python"]["symbols"]:
            index.setdefault(symbol, []).append(claim)
    return index


def reference_owners(app) -> list[dict]:
    """Keep scientific variants separate from residual numerical comparisons."""
    return [
        variant | {"family": procedure["family"], "procedure": procedure["id"]}
        for procedure in getattr(app, "_statadic_procedures", [])
        for variant in procedure["variants"]
    ] + [
        method
        for method in getattr(app, "_statadic_research", [])
        if method.get("reference_mappings")
    ]


def reference_index(app) -> str:
    """Render declarations and their exact scope without inferring execution."""
    procedures = getattr(app, "_statadic_procedures", [])
    variants = [v for p in procedures for v in p["variants"]]
    numerical = [
        m for m in getattr(app, "_statadic_research", []) if m.get("reference_mappings")
    ]
    mapped = sum(bool(v.get("reference_mappings")) for v in variants)
    lines = [
        f"**{mapped} / {len(variants)} scientific configurations have declared "
        f"comparisons; {len(numerical)} implementation owners have separate "
        "mappings.**",
        "",
        "These are reviewed test mappings, not completed results. The reference "
        "runner retains exact executed nodes, dependency versions, precision and "
        "outcomes. Unmapped means no comparison is declared here; numerical "
        "mappings never confer coverage on a scientific variant.",
        "",
        '<a href="../downloads/reference-tests/catalogue.json">'
        "Download the authored catalogue</a> "
        "· [Run and interpret comparisons](../validation.md)",
        "",
    ]
    for owner in reference_owners(app):
        target = f"reference-{owner['id']}"
        mappings = owner.get("reference_mappings", [])
        if not mappings:
            continue
        lines.extend(
            [
                f"({target})=",
                f'<details id="{make_id(target)}">',
                f"<summary>{escape(owner['title'])}</summary>",
                "",
                f"`{owner['id']}` · "
                "**Declared; execution requires retained evidence.**",
                "",
            ]
        )
        if "procedure" in owner:
            lines.extend(
                [
                    "Configuration: "
                    + ", ".join(
                        f"`{key}={value}`"
                        for key, value in owner.get("selectors", {}).items()
                    )
                    + ".",
                    "",
                    f"{{ref}}`Source review and registered studies "
                    f"<research-procedure-{owner['procedure']}>`.",
                    "",
                ]
            )
        for mapping in mappings:
            oracle = mapping["oracle_kind"]
            lines.extend(
                [
                    f"**{oracle}** · `{mapping.get('distribution', '')}` · "
                    f"`{mapping['call']}`",
                    "",
                    "Construction: "
                    + ", ".join(f"`{p}`" for p in mapping["construction_paths"])
                    + ".",
                    "",
                    f"Checks: {', '.join(mapping['quantities'])}.",
                    "",
                    f"Conventions: {mapping['conventions']}",
                    "",
                    f"Limits: {mapping['limits']}",
                    "",
                ]
            )
            if mapping.get("configuration"):
                lines.extend([f"Configuration: {mapping['configuration']}", ""])
            for node in mapping["nodeids"]:
                module = node.split("::", 1)[0]
                # These local downloads contain the actual build's tests, including
                # uncommitted changes, rather than a link to an older Git revision.
                lines.append(
                    '- <a href="../downloads/reference-tests/'
                    f'{escape(module, quote=True)}">'
                    f"<code>{escape(node)}</code></a>"
                )
            lines.append("")
        claims = [
            claim
            for claim in getattr(app, "_statadic_formal", {}).get("claims", [])
            if owner["family"] in claim["methods"]
        ]
        if claims:
            lines.extend(
                [
                    "Related mathematics; inspect each result's assumptions and scope: "
                    + ", ".join(
                        f"{{ref}}`{c['id']} <formal-{c['id']}>`" for c in claims
                    )
                    + ".",
                    "",
                ]
            )
        lines.extend(["</details>", ""])
    lines.extend(["## Unmapped scientific configurations", ""])
    for procedure in procedures:
        missing = [v for v in procedure["variants"] if not v.get("reference_mappings")]
        if missing:
            lines.append(
                f"- **{procedure['title']}**: "
                + ", ".join(f"`{v['id']}`" for v in missing)
                + "."
            )
    if mapped == len(variants):
        lines.append(
            "Every registered scientific configuration has a declared mapping."
        )
    return "\n".join(lines)


def api_evidence(app, what, name, obj, options, lines: list[str]) -> None:
    """Add evidence only during autodoc rendering; never mutate source or __doc__."""
    claims = app._statadic_claims_by_symbol.get(name, [])
    methods = [
        m for m in getattr(app, "_statadic_research", []) if name in m["python_symbols"]
    ]
    references = [
        owner
        for owner in reference_owners(app)
        if any(
            name in mapping["construction_paths"]
            for mapping in owner.get("reference_mappings", [])
        )
    ]
    if references:
        lines.extend(["", "Reference comparisons", "~~~~~~~~~~~~~~~~~~~~~", ""])
        for owner in references:
            lines.append(
                f"* :ref:`{owner['title']} <reference-{owner['id']}>` — "
                "declared comparison; inspect configuration and retained execution."
            )
    if methods:
        lines.extend(["", "Research evaluations", "~~~~~~~~~~~~~~~~~~~~", ""])
        for method in methods:
            lines.append(
                f"* :ref:`{method['title']} <research-method-{method['id']}>` — "
                f"{method['disposition']}."
            )
    if not claims:
        return
    lines.extend(
        ["", "Related mathematical results", "~~~~~~~~~~~~~~~~~~~~~~~~~~~~", ""]
    )
    for claim in claims:
        lines.extend(
            [
                f"* :ref:`{claim['title']} <formal-{claim['id']}>` — "
                f"{claim['proof_status']}.",
                f"  Scope: {claim['scope']['relationship']}",
            ]
        )
    lines.extend(
        [
            "",
            "These results concern the stated mathematical models. They do not alone "
            "verify this Python implementation or its floating-point arithmetic.",
            "",
        ]
    )


def _parameter_table(reports) -> str:
    """Format retained parameter evidence without recomputing inference."""

    def number(value):
        return "unavailable" if value is None else f"{value:.4f}"

    table = [
        "| Procedure | Parameter | Estimate | Standard error | "
        "95% interval | Availability |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    warnings = []
    for name, report in reports:
        for row in report.parameters:
            interval = (
                f"[{number(row.ci_lower)}, {number(row.ci_upper)}]"
                if row.ci_lower is not None
                else "not requested"
                if row.interval_status is None
                else "unavailable"
            )
            reason = row.unavailable_reason or row.interval_unavailable_reason
            table.append(
                f"| {name} | {row.name} | {number(row.estimate)} | "
                f"{number(row.std_error)} | {interval} | {reason or row.status} |"
            )
        warnings.extend(
            f"**Warning ({name}):** {warning}" for warning in report.warnings
        )
    return "\n".join(table) + ("\n\n" + "\n\n".join(warnings) if warnings else "")


def _quick_regression_report():
    """Read the quick start's own saved report, not another example's fit."""
    from statadic.api.artifacts import load_analysis

    example = ROOT / "src/python/packages/statadic/examples/first_regression.py"
    with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
        output = Path(directory)
        runpy.run_path(str(example))["analyze"](output)
        return load_analysis(output / "analysis").report


def _ols_output(analysis: dict, section: str) -> str:
    """Expose only outputs retained by the canonical chapter example."""
    if section == "fit":
        summary = "\n".join(
            f"| {name} | {value:.6g} |" for name, value in analysis["summary"].items()
        )
        return (
            f"```text\n{analysis['fit_report'].to_text()}\n```\n\n"
            f"| Quantity | Value |\n| --- | ---: |\n{summary}"
        )
    if section == "diagnose":
        return (
            '```{raw} html\n<img src="figures/residuals.svg" '
            'class="statadic-study-plot" alt="Residuals against fitted values '
            'for this OLS sample">\n```\n\n'
            f"```text\n{analysis['diagnostics_report'].to_text()}\n```"
        )
    if section == "infer":
        table = _parameter_table(analysis["reports"].items())
        tests = "\n\n".join(
            f"**{name}**\n\n```text\n{outcome.report().to_text()}\n```"
            for name, outcome in analysis["tests"].items()
        )
        return f"{table}\n\n{tests}"
    return (
        '```{raw} html\n<img src="figures/coefficients.svg" '
        'class="statadic-study-plot" alt="HC3 normal-reference coefficient '
        'intervals for this OLS sample">\n```\n\n'
        f"```text\n{analysis['reports']['hc3'].to_text()}\n```\n\n"
        "New-row point predictions (without prediction intervals):\n\n```text\n"
        + str(analysis["predictions"])
        + "\n```"
    )


def source(app, docname: str, text: list[str]) -> None:
    if "<!-- statadic:reference-index -->" in text[0]:
        text[0] = text[0].replace(
            "<!-- statadic:reference-index -->", reference_index(app)
        )
    if "<!-- statadic:research-catalogue -->" in text[0]:
        sections = []
        if getattr(app, "_statadic_evaluations", []):
            release = getattr(app.config, "statadic_release_evidence", False)
            sections.extend(
                [
                    "## Selected release evaluations\n"
                    if release
                    else "## Historical evaluations\n",
                    (
                        "These exports were explicitly selected for this release "
                        "build. Consult their wheel identities, exact scenarios and "
                        "scientific dispositions; selection alone does not qualify "
                        "the complete release scope. "
                        if release
                        else "These historical results describe the frozen design "
                        "and implementation in each download. Source-checkout "
                        "evidence does not qualify an installed release. "
                    )
                    + "Procedure failures and Monte Carlo uncertainty remain "
                    "in the summaries.\n",
                    "| Study | Profile | Attempts / budget | "
                    "Installation | Downloads |",
                    "| --- | --- | ---: | --- | --- |",
                ]
            )
            plots = []
            for evaluation in app._statadic_evaluations:
                frozen = evaluation["frozen"]
                payload = frozen["payload"]
                report = evaluation["report"]
                attempted = sum(
                    cell["attempted"] for cell in report["scenarios"].values()
                )
                requested = sum(
                    cell["requested"] for cell in report["scenarios"].values()
                )
                prefix = f"../downloads/studies/{frozen['id']}"
                sections.append(
                    f"| `{payload['study']['id']}` | {payload['profile']} | "
                    f"{attempted} / {requested} | "
                    f"{payload['installation_scope']} | "
                    f'<a href="{prefix}/results.md">Results</a>, '
                    f'<a href="{prefix}/summary.json">JSON</a>, '
                    f'<a href="{prefix}/results.tex">LaTeX</a>, '
                    f'<a href="{prefix}/campaign.json">Provenance</a> |'
                )
                if operating_points(report):
                    plots.append(
                        "\n<details><summary>Operating characteristics: "
                        f"{escape(payload['study']['id'])}</summary>\n\n"
                        f'<img class="statadic-study-plot" '
                        f'src="{prefix}/operating-characteristics.svg" '
                        'alt="Retained rejection, coverage and failure rates with '
                        "uncertainty bounds and frozen decision thresholds. "
                        "Exact values and dispositions are available in the "
                        'linked summary.">\n\n'
                        "Endpoints are separate one-sided 95% bounds, not simultaneous "
                        "confidence intervals. Dashed lines show the frozen study "
                        "criteria. Failure-inclusive rates and adverse dispositions "
                        "remain visible. See the JSON summary for exact values.\n\n"
                        "</details>\n"
                    )
                else:
                    plots.append(
                        "\nThis export has no retained operating-characteristic "
                        "bounds to plot. Consult its metric-specific results; "
                        "absence is not zero error or statistical qualification.\n"
                    )
            sections.extend(plots)
            sections.append(
                "\nDownloads contain summaries and provenance. Independent "
                "reanalysis requires the complete per-trial campaign retained "
                "by the study runner.\n"
            )
        elif getattr(getattr(app, "config", None), "statadic_release_evidence", False):
            sections.extend(
                [
                    "## Selected release evaluations\n",
                    "No study exports were selected. This page supplies no "
                    "current-release Monte Carlo qualification evidence.\n",
                ]
            )
        for procedure in getattr(app, "_statadic_procedures", []):
            rows = [
                f"(research-procedure-{procedure['id']})=",
                f"## {procedure['title']}",
                "",
                procedure["rationale"],
                "",
                "| Variant | Registered study | Source correspondence |",
                "| --- | --- | --- |",
            ]
            for variant in procedure["variants"]:
                studies = ", ".join(
                    f"`{mapping['study']}`" for mapping in variant["study_mappings"]
                )
                rows.append(
                    f"| {variant['title']} | {studies or 'Not registered'} | "
                    f"{variant['source_review']} |"
                )
            rows.extend(
                [
                    "",
                    "These links identify study designs. Execution and scientific "
                    "dispositions require retained campaign evidence.",
                ]
            )
            sections.append("\n".join(rows))
        sections.append("## Implementation ownership\n")
        for method in app._statadic_research:
            # MyST registers labels on raw HTML nodes, whose HTML writer does
            # not emit their attached IDs. Keep the reference label and give
            # the disclosure its matching normalized browser target explicitly.
            target = make_id(f"research-method-{method['id']}")
            sections.append(
                f"(research-method-{method['id']})=\n"
                f'<details id="{escape(target, quote=True)}">\n'
                f"<summary>{escape(method['title'])}</summary>\n\n"
                f"Disposition: **{method['disposition']}**. "
                f"Citation review: **{method['citation_review']}**.\n\n"
                f"{method['rationale']}\n\n"
                f"Studies: {', '.join(method['studies']) or 'none registered'}. "
                "Registration does not establish completed or passing evidence.\n\n"
                + "Tutorials: "
                + (
                    ", ".join(
                        f"[{slug}](../labs/{slug}.md)"
                        for slug in method.get("tutorials", [])
                    )
                    or "none registered"
                )
                + ".\n\n"
                + "Benchmarks: "
                + (
                    ", ".join(
                        f"{{ref}}`{identity} "
                        f"<performance-{identity.replace('/', '-')}>`"
                        for identity in method.get("benchmarks", [])
                    )
                    or "none registered"
                )
                + ".\n"
                "\n</details>\n"
            )
        text[0] = text[0].replace(
            "<!-- statadic:research-catalogue -->",
            "\n".join(sections)
            or "No research catalogue was configured for this build.",
        )
    methods = getattr(app, "_statadic_research", [])
    if docname.startswith("labs/"):
        slug = docname.split("/", 1)[1]
        related = [m for m in methods if slug in m.get("tutorials", [])]
        if related:
            text[0] += (
                "\n\n## Related catalogue methods\n\n"
                + "\n".join(
                    f"- {{ref}}`{m['title']} <research-method-{m['id']}>`"
                    for m in related
                )
                + "\n"
            )
    marker = "<!-- statadic:performance-catalogue -->"
    if marker in text[0]:
        identities = sorted(
            {identity for m in methods for identity in m.get("benchmarks", [])}
        )
        sections = ["## Related catalogue methods\n"]
        for identity in identities:
            links = ", ".join(
                f"{{ref}}`{m['title']} <research-method-{m['id']}>`"
                for m in methods
                if identity in m.get("benchmarks", [])
            )
            sections.append(
                f"- {{ref}}`{identity} "
                f"<performance-{identity.replace('/', '-')}>`: {links}"
            )
        text[0] = text[0].replace(marker, "\n".join(sections))
    regression_marker = "<!-- statadic:regression-result -->"
    covariance_marker = "<!-- statadic:covariance-comparison -->"
    if regression_marker in text[0]:
        example = ROOT / "src/python/packages/statadic/examples/first_regression.py"
        text[0] = text[0].replace(
            regression_marker,
            "Computed from this quick start's saved HC3/normal report:\n\n"
            + _parameter_table((("HC3", _quick_regression_report()),)),
        )
        app.env.note_dependency(str(example))
    if covariance_marker in text[0]:
        example = ROOT / "src/python/packages/statadic/examples/change_covariance.py"
        reports = runpy.run_path(str(example))["compare"]()
        text[0] = text[0].replace(
            covariance_marker,
            "Computed from the covariance example (normal-reference intervals):\n\n"
            + _parameter_table(zip(("HC0", "HC3"), reports, strict=True)),
        )
        app.env.note_dependency(str(example))
    pattern = r"<!-- statadic:ols-result (fit|diagnose|infer|report) -->"
    if re.search(pattern, text[0]):
        example = ROOT / "src/python/packages/statadic/examples/ols_analysis.py"
        if not hasattr(app, "_statadic_ols_analysis"):
            app._statadic_ols_analysis = runpy.run_path(str(example))["analyze"]()
        text[0] = re.sub(
            pattern,
            lambda match: _ols_output(app._statadic_ols_analysis, match.group(1)),
            text[0],
        )
        app.env.note_dependency(str(example))
    if docname == "formal/index":
        text[0] = text[0].replace(
            "<!-- statadic:formal-registry -->", index_page(app._statadic_formal)
        )
    elif docname.startswith("formal/claims/"):
        identifier = docname.rsplit("/", 1)[1]
        claim = next(c for c in app._statadic_formal["claims"] if c["id"] == identifier)
        text[0] = text[0].replace(
            f"<!-- statadic:claim {identifier} -->", claim_page(claim)
        )
    claims = {claim["id"]: claim for claim in app._statadic_formal["claims"]}
    pattern = r"<!-- statadic:math ([A-Z0-9-]+) -->"
    if re.search(pattern, text[0]):
        text[0] = re.sub(
            pattern,
            lambda match: math_excerpt(claims[match.group(1)], link=True),
            text[0],
        )
        app.env.note_dependency(str(ROOT / "src/lean/build/certification.json"))
    if docname.startswith("formal/"):
        app.env.note_dependency(str(ROOT / "src/lean/build/certification.json"))


def outputs(app, exception) -> None:
    if exception is not None or app.builder.format != "html":
        return
    if hasattr(app, "_statadic_ols_analysis"):
        from statadic.api.plots import residual_plot

        analysis = app._statadic_ols_analysis
        figures = Path(app.outdir) / "tutorials/ols/figures"
        figures.mkdir(parents=True, exist_ok=True)
        for name, plot in (
            ("residuals", residual_plot(analysis["fit"])),
            ("coefficients", analysis["reports"]["hc3"].coefficient_plot()),
        ):
            plot.to_matplotlib().savefig(figures / f"{name}.svg", bbox_inches="tight")
    catalogue = getattr(app, "_statadic_reference_catalogue", None)
    if catalogue is not None:
        destination = Path(app.outdir) / "downloads/reference-tests"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(catalogue, destination / "catalogue.json")
        package = ROOT / "src/python/packages/statadic"
        modules = {
            node.split("::", 1)[0]
            for owner in reference_owners(app)
            for mapping in owner.get("reference_mappings", [])
            for node in mapping["nodeids"]
        }
        for module in sorted(modules):
            origin = (package / module).resolve()
            if not origin.is_relative_to(package.resolve()) or not origin.is_file():
                raise ValueError(f"Invalid reference-test source: {module}")
            target = destination / module
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)
    for evaluation in getattr(app, "_statadic_evaluations", []):
        destination = (
            Path(app.outdir) / "downloads/studies" / evaluation["frozen"]["id"]
        )
        destination.mkdir(parents=True, exist_ok=True)
        for name in (
            "results.md",
            "results.tex",
            "summary.json",
            "evidence-manifest.json",
        ):
            published = destination / name
            shutil.copy2(evaluation["path"] / name, published)
            published.chmod(0o644)
        (destination / "campaign.json").write_text(
            json.dumps(evaluation["frozen"], indent=2) + "\n"
        )
        operating_plot(
            evaluation["report"],
            evaluation["frozen"]["payload"]["study"],
            destination / "operating-characteristics.svg",
        )
    output = Path(app.outdir) / "formal"
    downloads = output / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    for name in (
        "claims.json",
        "public-api.json",
        "methods.json",
        "legacy-candidates.json",
        "citation-aliases.json",
    ):
        shutil.copy2(ROOT / "src/lean/registry" / name, downloads / name)
    shutil.copy2(
        ROOT / "src/lean/build/certification.json", downloads / "certification.json"
    )
    with (downloads / "public-evidence.json").open("w", encoding="utf-8") as matrix:
        subprocess.run(
            [sys.executable, str(ROOT / "src/lean/tools/evidence_crosswalk.py")],
            check=True,
            stdout=matrix,
        )
    shutil.copy2(ROOT / "src/latex/refs.bib", downloads / "refs.bib")
    shutil.copytree(
        ROOT / "src/lean/packages/StatadicFormal",
        downloads / "lean",
        dirs_exist_ok=True,
    )
    shutil.copytree(
        ROOT / "src/lean/blueprint/web", output / "blueprint", dirs_exist_ok=True
    )
    shutil.copy2(
        ROOT / "src/lean/blueprint/print/print.pdf", downloads / "blueprint.pdf"
    )
    latex_downloads = downloads / "latex"
    latex_downloads.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        ROOT / "src/lean/build/manuscript", downloads / "manuscript", dirs_exist_ok=True
    )
    lean_root = (ROOT / "src/lean").resolve()
    for claim in app._statadic_formal["claims"]:
        artifact = (lean_root / claim["latex"]["artifact"]).resolve()
        if not artifact.is_relative_to(lean_root):
            raise RuntimeError(f"Invalid Lean artifact path: {artifact}")
        destination = latex_downloads / f"{claim['id']}.tex"
        shutil.copy2(artifact, destination)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest != claim["latex"]["sha256"]:
            raise RuntimeError(f"Lean artifact hash mismatch for {claim['id']}")
    expected = [
        downloads / "claims.json",
        downloads / "public-api.json",
        downloads / "public-evidence.json",
        downloads / "methods.json",
        downloads / "certification.json",
        downloads / "legacy-candidates.json",
        downloads / "refs.bib",
        downloads / "blueprint.pdf",
        output / "blueprint/index.html",
    ]
    expected.extend(
        downloads
        / "lean"
        / Path(*claim["lean"]["module"].split(".")).with_suffix(".lean")
        for claim in app._statadic_formal["claims"]
    )
    expected.extend(
        latex_downloads / f"{claim['id']}.tex"
        for claim in app._statadic_formal["claims"]
    )
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"Missing formal documentation artifacts: {', '.join(missing)}"
        )


def unlinked_claim_symbol(app, env, node, contnode):  # noqa: ARG001
    """Render a claim's Python symbol as code when the API has no page for it.

    Claim pages name every implementing symbol, but the curated API documents
    only selected objects. A symbol without a target keeps its literal name;
    every other unresolved reference still fails the strict build.
    """
    if (
        node.get("refdomain") == "py"
        and node.get("reftype") == "obj"
        and node.get("refdoc", "").startswith("formal/claims/")
        and node.get("reftarget", "").startswith("statadic.")
    ):
        return contnode
    return None


def setup(app) -> dict:
    app.add_config_value("statadic_release_evidence", False, "env")
    app.add_config_value("statadic_research_catalogue", None, "env")
    app.add_config_value("statadic_study_exports", [], "env")
    app.connect("builder-inited", load)
    app.connect("source-read", source)
    app.connect("autodoc-process-docstring", api_evidence)
    app.connect("missing-reference", unlinked_claim_symbol)
    app.connect("build-finished", outputs)
    return {"version": "1", "parallel_read_safe": False, "parallel_write_safe": False}
