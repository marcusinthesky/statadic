# Evidence: what has actually been checked?

A validation claim belongs to a **method, configuration, output and reference version**. The evidence categories below answer different questions.

The [reference index](references.md) connects familiar constructors and explicit
compositions to exact tests, checked outputs, conventions and visible gaps.

## Evidence categories

| Category | What it contributes |
|---|---|
| Contract/numerical check | Behavior on specified domains, edge cases, shapes and numerical witnesses |
| Internal consistency | Agreement across eager/JIT/vmap or related internal constructions |
| Independent reference | Matched analytic, historical or independently implemented comparison |
| External reference | Live comparison with an optional peer implementation |
| Conformance | An explicitly adopted external compatibility promise |
| Statistical qualification | Prespecified complete-procedure studies with failure-inclusive operating-characteristic bounds |

Internal consistency is not independent confirmation. A marker is not a coverage denominator. A peer-library match does not certify statistical coverage or every configuration in the family.

[Hypothesis reports](../reference/hypotheses.md) retain the requested null,
reference, construction and assumptions. These describe the procedure; they do
not certify those assumptions. Auxiliary-regression composition parity is
internal consistency, while the matched statsmodels and linearmodels controls
provide independent numerical comparisons on their declared configurations.

## A method evidence record

A useful evidence record identifies the canonical method, estimand/null, input regime, covariance/reference conventions, precision/backend, source and reference versions, test node or study identifier, requested/valid/failed counts, criterion, disposition and artifact provenance. Preserve `qualified`, `excess-error`, `inconclusive` and `not run` as distinct states.

Some preliminary spatial-survival numerical runs lack a pre-run source manifest. Keep that provenance limitation visible.

## Release evidence

The manual/tag workflow runs a bounded artifact preflight. A complete release
qualification additionally needs the exact installed-wheel studies, source reviews,
consumer acceptance and resource measurements selected by its frozen scope.
Neither a successful site build nor a preflight qualifies the release.

[Study mappings and retained reports](research.md) distinguish registration,
historical evaluations and explicitly selected release evaluations. Inspect each
download's wheel identity, scenarios, attempted/failed counts, uncertainty and
scientific disposition. An absent report is absent evidence. Human
source-fidelity review remains separate from Lean compilation.

## Known limits are part of the product documentation

Read the [known-limitations page](known-limitations.md) before using affected methods. Its records summarize the validation guide; consult the underlying study and artifacts for the exact scope.

For test commands, reference versions and campaign criteria, see the
[detailed validation guide](../validation.md).

:::{toctree}
:maxdepth: 1

Independent references <references>
Study mappings and retained reports <research>
Formal mathematics <../formal/index>
known-limitations
Validation commands and criteria <../validation>
Bibliography <../reference/bibliography>
:::
