---
type: task
title: Add opt-in coefficient tests to inference reports
description: Let InferenceResult.report fill each coefficient's statistic and two-sided p-value only when the caller states a null value and a reference.
links:
  - .context/plan/api-cohesion/tasks/report-review/tasks/table-content/README.md
  - .context/plan/api-cohesion/tasks/followups/tasks/directional-tests/README.md
status: done
---

# Add opt-in coefficient tests to inference reports

## Goal

The owner accepted on 2026-11-11 that an explicit null value may add statistic and
p-value columns to the coefficient table (row P08 of the
[dispositions](../../../../proposal-dispositions.md)). Nothing is reported unless
the caller asks, so the report still chooses no null hypothesis.

Each coefficient test is the single-restriction Wald test that
`InferenceResult.test` already performs. The report shows its signed root, which
is the familiar z or t statistic, with the same p-value. One-sided tests remain
with [directional tests](../../../followups/tasks/directional-tests/README.md).

## Acceptance

- [x] `InferenceResult.report` takes `null: float | Mapping[str, float] | None = None`; `None` leaves every row as today.
- [x] A null without a `reference` raises `ValueError`; a boolean, nonfinite or unknown-name null is rejected.
- [x] Each tested row retains `statistic = (estimate - null) / std_error` and the two-sided p-value of the matching single-restriction test.
- [x] The p-value equals `InferenceResult.test` exactly: `chi2` for the normal reference, and `f` with the same degrees of freedom for the t reference.
- [x] A row whose test is unavailable keeps both values `None` and gains a warning naming the row and the native status.
- [x] Metadata retains the null, and one note states the null, the two-sided alternative and the reference.
- [x] An external-reference test matches statsmodels OLS t-values and p-values: model-based with the t reference, HC3 with the normal reference, and one nonzero null through `t_test`.
- [x] `tests/api/test_inference.py` and the new reference test pass; lint and strict typing pass for the touched files.

## Specification

- File: `src/python/packages/statadic/src/statadic/api/inference.py`, method `InferenceResult.report`.
  The coefficient tests belong to this method only. Other producers keep their reports.
- A float null applies to every row; a mapping tests only the rows it names.
- Per tested row `j`, call `linear_restriction_test(self.uncertainty, e_j, [null_j], ...)`
  with a one-row matrix selecting coefficient `j`, `reference="chi2"` for
  `"normal"` and `reference="f"` with the caller's `degrees_of_freedom` for `"t"`.
  Use its `test.pvalue` when `valid`. The p-value of a chi-square(1) or F(1, d)
  test of the square is the two-sided normal or t p-value of the root.
- Compute the statistic on the host from the retained estimate and standard error,
  only for rows whose standard error is available and whose test is valid.
- Add the native `finite_reference_outside_validated_range` flag of these tests to
  the existing warning.
- Metadata keys: `"null"` holds the float or the mapping as given, and is absent
  when no null is given. Note text: `Two-sided tests against 0; normal reference.`
  for a float, or `Two-sided tests of x = 1, z = 0; t reference.` for a mapping.
- Put the statsmodels comparison beside the existing OLS references in
  `tests/integration/api/test_structural_references.py` or
  `tests/integration/api/test_reference_walkthroughs.py`, marked like their
  neighbours, and run it with
  `cd src/python/packages/statadic && just test-external <file>` inside `devenv shell`.
- `api/reporting.py` is not edited here. Its renderer shows the statistic column
  whenever a row has a statistic, and `test-tables` heads it `$z$` or `$t$` from
  the retained reference. Do not assert rendered LaTeX in this task.

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

_No open children._

`0 open; 0 terminal`

<!-- insitu:end -->

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

- [Derive coefficient statistics and p-values from one canonical null](../../../final-review/tasks/coefficient-test-null/README.md)
- [Document coefficient tests, test tables and value export](../reporting-docs/README.md)

<!-- insitu:end -->
