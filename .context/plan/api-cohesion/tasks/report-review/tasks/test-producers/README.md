---
type: task
title: Report every calibrated test as a typed test row
description: Fill TestRow from TestOutcome, diagnostics, moment-condition and permutation reports, and keep integer test components exact.
links:
  - .context/plan/api-cohesion/tasks/report-review/tasks/test-tables/specification.md
status: done
---

# Report every calibrated test as a typed test row

## Goal

After [test tables](../test-tables/README.md) adds `TestRow`, every producer that
reports a test statistic with a calibrated p-value must fill one, instead of
`statistic` and `pvalue` metrics. Start when `test-tables` is done.

No statistic, p-value or reference law changes. Only where each is retained does.

## Acceptance

- [x] `TestOutcome.test_row()` returns its `TestRow`; `TestOutcome.components()` returns one metric per detail with an empty definition; `TestOutcome.metrics()` is removed.
- [x] `TestOutcome.report()` retains the row in `tests` and the components in `metrics`.
- [x] `Diagnostics.report()` retains one row per test outcome in request order, and components and influence metrics as metrics.
- [x] Every other producer under `api/` that reports a calibrated p-value as a metric reports a `TestRow` instead; none keeps a `pvalue` metric.
- [x] An integral test component, such as a rank or count of restrictions, is retained as an `int`.
- [x] No consumer in the repository reads a removed metric name.
- [x] The named test files for every touched module pass; lint and strict typing pass for the touched files.

## Specification

- `TestOutcome` in `api/hypotheses.py`. Map the reference to the row:

  | Reference | `law` | `law_parameters` |
  | --- | --- | --- |
  | `ChiSquare` | `chi-square` | `(df,)` |
  | `FisherF` | `f` | `(df, df_denominator)` |
  | `Normal` | `normal` | `(location, scale)` |
  | `ResampledReference` | `resampled` | `(draws,)` |
  | none | `None` | `()` |

  `reference` is `str(reference)` or empty; `hypothesis` is `self.hypothesis.describe()`.
- Components: `details` keep an `int` for a non-boolean integral input and a
  `float` otherwise. In `InferenceResult.test` (`api/inference.py`), retain
  `restriction_rank`, `covariance_rank` and `numerator_df` with `int(...)`, because
  their native arrays are integer. Do the same wherever a producer passes an
  integer native array as a detail.
- `Diagnostics.report` in `api/diagnostics.py` branches on `TestOutcome` and
  `InfluenceDiagnostic`. Its metadata, warnings and notes are unchanged.
- Known metric p-values to convert, found by searching for `"pvalue"` under `api/`:
  - the moment-condition inference report in `api/moments.py` (`statistic`,
    `degrees_of_freedom`, `pvalue`): one chi-square row named for the test the
    native result computes, with its degrees of freedom;
  - the permutation reports in `api/association.py` (Mantel),
    `api/comparison/_results.py` and `api/regression/_geometric.py`: one
    `resampled` row whose statistic is the observed statistic that the p-value
    calibrates, and whose draws are the null draws that enter the p-value.
    Describe the method and draw count in `reference`.
  In a permutation report, drop the metric that duplicates the row's statistic.
  Keep the descriptive metrics and the requested and failed draw counts.
- Search the whole repository, including `src/python/apps/` and the docs
  notebooks, for readers of the removed metric names and of `TestOutcome.metrics`,
  and update them.
- `api/reporting.py` and `api/artifacts/` belong to `test-tables`. If they need a
  change, report it instead of making it.

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

_No declared backlinks._

<!-- insitu:end -->
