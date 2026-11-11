---
type: task
title: Repair the final alpha review and type fit evidence
description: Repair the confirmed findings of the final alpha review, give fit statistics one typed owner, close their reference gaps and type contrast estimands.
links:
  - .context/plan/api-cohesion/proposal-dispositions.md
  - .context/plan/api-cohesion/tasks/report-review/tasks/table-content/specification.md
status: in-progress
---

# Repair the final alpha review and type fit evidence

## Goal

Two inputs of 2026-11-11 meet here. The final alpha review of `4cc3077` reports
four reporting defects, R1–R4, each confirmed against the source. A separate audit of fit statistics found that HQIC is checked against a
peer only for OLS and WLS, and that several spatial likelihoods have no reference.
It also found a thin abstraction: ten results hand-count `k` and `n`, spatial
cores name their likelihood `loglik`, no Protocol describes either, private
helpers cross modules, and least-squares mathematics lives in `api/base.py`.
Finally, `core.statistics.Estimand` and `api.estimand.Estimand` share a name, and
contrast estimands are plain strings.

The owner decided on 2026-11-11:

- R1–R4 are accepted, as arbitrated in rows F01–F10 of the
  [dispositions](../../proposal-dispositions.md).
- The core estimand root becomes `PopulationEstimand`.
- Contrast estimands are typed now.
- `to_latex` gains footer selection.
- Permutation and GMS p-values gain independent witnesses.
- No AICc.

The work runs in waves of Sonnet sub-agents on disjoint files. The coordinator
owns records, inventories, the catalogue, Tach review, `just sync`, the typecheck
after each wave and the single final `just check`.

| Wave | Tasks |
| --- | --- |
| 1 | `coefficient-test-null`, `exact-fit-r-squared`, `report-record-contract`, `spatial-log-likelihood-name`, `pvalue-witnesses` |
| 2 | `fit-contract`, `contrast-estimand`, `criterion-peers`, `spatial-likelihood-references` |
| 3 | `regression-families`, `spatial-families`, `survival-families`, `latex-footer-selection` |
| 4 | `fit-cleanup-docs` |

These are deferred:

- **Caching fit summaries** to avoid repeated rank work. It has not been measured.
- **Splitting `reporting.py` internally.** No repeated cross-responsibility change
  has been observed, and the public `Report` stays stable.
- **Regression route invariants.** These are backlog in
  [route invariants](../followups/tasks/route-invariants/README.md).

## Acceptance

- [x] Complete every child within its declared file ownership.
- [x] After each wave: review every hand-back against its record and pass the package typecheck.
- [x] Refresh the inventory and statement pin, and reconcile the catalogue after waves 2 and 4.
- [x] Pass the external reference lane for every touched reference file, with no skips.
- [ ] Pass `just check` once at handoff, and report any check that could not run.

The continuation's [execution evidence](../../../compatibility-properties/evidence.md)
records the current no-skip reference reruns and final verification.

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

_No open children._

`0 open; 15 terminal`

<!-- insitu:end -->

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

- [Test invariants across Regression routes](../followups/tasks/route-invariants/README.md)
- [Qualify compatibility and strengthen API properties](../../../compatibility-properties/README.md)

<!-- insitu:end -->
