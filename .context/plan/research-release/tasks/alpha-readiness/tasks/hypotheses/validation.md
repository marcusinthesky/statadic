---
type: reference
title: Hypothesis and explicit-order implementation evidence
description: Implementation decisions, actual checks and retained limits for the hypothesis refactor.
links:
  - .context/plan/research-release/tasks/alpha-readiness/tasks/hypotheses/README.md
---

# Hypothesis and explicit-order implementation evidence

Baseline: clean `5c8158fde57994364d95168359249203e1851047`.
The audit read all 541 lines / 31,348 bytes of the named 2026-11-10 transcript.
Its collapsed original excerpt is not represented as a separate complete read.

## Accepted boundaries

The core semantic marker becomes `Null`; host `Hypothesis` is a small immutable
value with explicit metadata. Named restrictions bind to actual parameter names
and contrast-derived restrictions also bind to the identical fitted design.
Diagnostics and parameter tests share `TestOutcome`; native numerical carriers
remain at their scientific owners. Influence keeps its individual masks.

RESET retains its rank-aware kernel. Hansen J preserves different two-step and
iterated eligibility. Geometric requests wrap `Reduced`, retaining their existing
permutation evidence. BG, DW and HAC require explicit `order_by`; a named finite,
unique real numeric key sorts ascending and explicit `None` retains row order.
Lags count observation positions, not calendar intervals. HAC ordering applies to
OLS, WLS, IV and SLX without refitting or replacing the retained fit.

Report snapshots preserve interpretation rather than executable inference.
Artifact schema 6, source-review status and existing scientific release gates
remain unchanged. No generic procedure framework, public auxiliary model or
opaque kernel field is introduced.

## Verification

All commands use the pinned devenv shell; numerical suites run serially.

- Named native checks: `tests/core/test_statistics.py`,
  `tests/inference/test__row_order.py`, `test_linear_diagnostics.py` and
  `test_linear_covariance.py`: **64 passed**. Includes unchanged diagnostic
  behavior, independent weighted HAC products and JIT/vmap invalid permutations.
- Focused live float64 references: `tests/inference/test_iv_reference.py` and
  `test_linear_diagnostics_reference.py`: **11 passed**. Dedicated IV two-step
  and iterated J match linearmodels under plain/centered bandwidth-zero weights;
  ordered BG/DW and OLS/WLS Bartlett HAC match statsmodels.
- Initial architecture checks stopped on unused dependency declarations while
  consumer migrations were incomplete; subsequent graph reconciliation passed.

The subsequent architecture gate passed, including all 13 negative fixtures.
The first strict type check found ten integration/test-annotation errors; those
were repaired. The final strict package check reported zero errors, and all
seven isolated native shape controls passed.

The first shared host run had 72 passing and 19 failing tests: the runtime
annotation hook rejected valid integer reference degrees declared as `float`.
Numeric statement fields now reuse `ReportNumber`, normalize stored values and
continue rejecting booleans. The final named shared/inference/geometric/regression/
static-contract/permutation selection passed **101 tests**. The diagnostic,
IV/GMM, explicit ordering, auxiliary-composition and SLX selection passed
**37 tests**. The float32 native/host ordering selection passed **15 tests**.

Independent review also exposed a new nonfinite-covariance metadata failure.
NaN/infinite options now retain null plus a reason, preserving native unavailable
status in outcomes and reports. A confirmed unsigned-index promotion defect was
repaired after a float64 gather failure: validated structural indices retain an
integer gather dtype. Its compiled control passes in both precision settings.

Ordering keys follow the existing `Observations` numeric normalization; no second
raw-source coercion policy is introduced. Calculation identities cross the host
boundary in one array transfer. Native fit arrays, weights and identity remain
unchanged. Independent NumPy IV Bartlett products now complement host plumbing
parity; the tests do not merely compare the adapter with its own covariance call.

The authored catalogue check passes with 2,408 public symbols and 213 method
records. It retains 51 procedures and 309 scientific variants. New and corrected
reference declarations do not upgrade source reviews, campaigns or dispositions.
The first strict documentation build stopped on 29 new cross-reference warnings.
Canonical type targets and alias resolution repaired them without suppressing
public types; the strict Sphinx-only retry passed with zero warnings and actual
rendered canonical links were inspected. The full native-lab build subsequently
passed as part of the aggregate check.

The full reference runner passed: **267** external-float64 comparisons, then
**23** precision-float32 and **23** precision-float64 comparisons. All 529
frozen input hashes still match after explicit formatting. This is editable-source
reference evidence; installed-wheel smoke checks are a separate scope.

## Final checks and retained evidence

The single required **`just check` passed**, including architecture, formatting,
strict typing, seven shape controls, **584 Testmon-selected native tests**,
evidence **15**, publishing **19**, studies **410 passed / 4 skipped**, performance
application **144**, examples in both precisions, docs **191 float32** and
**23 native float64** tests, strict Sphinx, seven native lab snapshots, LaTeX and
agent checks. The studies skips are retained; this is not a new scientific
qualification campaign. Full native suites remain CI's responsibility.

Fresh Lean certification reports **772 compiled declarations**, **153 literature
mappings** and **47 method families**. No statement pin, implementation pin,
source approval or mathematical claim was promoted by this host refactor.

The existing reference runner reconciled the authored declarations as follows:

| Lane | Comparisons | Executed mappings | Distinct mapped nodes | Other executed nodes |
| --- | ---: | ---: | ---: | ---: |
| external-float64 | 267 | 115 | 145 | 122 |
| precision-float32 | 23 | 5 | 4 | 19 |
| precision-float64 | 23 | 5 | 4 | 19 |

All lanes completed without failures, errors or skips. Precision mappings overlap
the main lane. **Nine** formula/fixture declarations have no runner lane and are
explicitly unexecuted by it. The catalogue has **124 declarations**, **153**
distinct declared nodes, **95/309** mapped scientific variants and **15** mapped
numerical method records; mappings never confer inherited scientific coverage.
An exact comparison against the baseline confirms that scientific procedure
records excluding reference declarations are unchanged.

`just python::statadic::test-artifacts` passed both isolated wheel routes, including
the new named restriction/outcome, explicit ordering and report snapshot smoke
checks. Standalone sdist lint/format/types passed, with **7** shape controls and
**58** typing tests. Its ordinary typing lane skipped the same seven shape tests
that passed in the dedicated environment; 3,514 other tests were deselected.
Installed NumPy controls and installed smoke checks are distinct from the
editable-source external comparisons above.

Artifacts use `SOURCE_DATE_EPOCH=315532800`:

| Archive | SHA-256 |
| --- | --- |
| Direct wheel | `3c3a958d5f19bde3dd9dff5ba68d2e2d04ded2443c5660a85e8907ddcba92282` |
| Sdist | `dd0b9ece865f7a84fc642629926203a976889eb2ff8af5358452af6140d2bf9f` |
| Independently tested sdist-built wheel | `a8b0aa62d27e7d64958d082a8494426deb50db167b128f9ec96b8aaf624e915c` |

The wheel archives are distinct: the direct locked build uses hatchling 1.27.0,
while the standalone declared-build route uses cached hatchling 1.33.0. Only
`METADATA`, `WHEEL` and `RECORD` differ; package payloads match. Both archives were
independently installed and tested; payload equivalence does not transfer a
scientific campaign between archive identities.

Browser checks passed **241 HTML targets**, **10 layouts**, **76 screenshots**
and zero JavaScript errors. The new hypothesis guide was visually inspected on
mobile; canonical API links were independently checked in rendered HTML.

Ignored retained evidence lives under `build/hypotheses-20261110/`: `checks/`,
`references/`, `artifacts/`, `artifact-checks/`, `docs/` and `browser/`.
`identity-audit.json` and `wheel-comparison.json` retain the scope and archive
distinctions. The final hash audit matched **529 reference inputs**, **699 docs
inputs**, all **three archives** and **76 screenshots** against current files.
The archive retains 29 check logs and 1,685 documentation files. Final authored
records were formatted, synchronized and workflow-checked without rerunning the
successful aggregate.

## Remaining release scope

This implementation task is complete. The alpha remains unqualified: the current
177-variant release selection still has 122 pending formula-source reviews,
required campaigns and full joint-survival qualification outstanding. Matched
performance reporting for the actual release candidate and hosted full CI remain
separate requirements. Historical consumer/campaign artifacts keep their own
identities; no earlier evidence is transferred to these archives.
