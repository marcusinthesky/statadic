# Test row specification

Scope: `src/python/packages/statadic/src/statadic/api/reporting.py` and the analysis
snapshot in `src/python/packages/statadic/src/statadic/api/artifacts/`. Producers are not
changed here; [test producers](../test-producers/README.md) fills the rows.
`reporting.py` must not import `statadic.api.hypotheses`, which imports it.

## Record

```python
@dataclass(frozen=True)
class TestRow:
    """One calibrated or descriptive test statistic, ready for a table of tests."""

    __test__: ClassVar[bool] = False  # not a pytest class

    name: str
    statistic: ReportNumber | None
    pvalue: ReportNumber | None = None
    law: Literal["chi-square", "f", "normal", "resampled"] | None = None
    law_parameters: tuple[ReportNumber, ...] = ()
    reference: str = ""
    hypothesis: str = ""
    available: bool = True
    unavailable_reason: str | None = None
    status: str | int = "available"
```

| Field | Meaning |
| --- | --- |
| `law` | The calibrating law's kind, using the `kind` values of the hypothesis reference metadata |
| `law_parameters` | `chi-square`: `(df,)`. `f`: `(df, df_denominator)`. `normal`: `(location, scale)`. `resampled`: `(draws,)` |
| `reference` | Plain description of the law, such as `chi2(2)` or `random resampling (999 draws)` |
| `hypothesis` | Plain statement of the null hypothesis |

Validation in `__post_init__`, raising `TypeError` for a wrong type and
`ValueError` otherwise:

- `name` is nonempty text; `reference` and `hypothesis` are text; `available` is a `bool`.
- `statistic` and `pvalue` are finite numbers or `None` (`_finite`); a p-value lies in [0, 1].
- Available: `statistic` is not `None` and `unavailable_reason` is `None`.
- Unavailable: a nonempty reason, both numbers `None`; status `"available"` becomes `"unavailable"`, as for `Metric`.
- `pvalue is None` exactly when `law is None`. A descriptive statistic has neither.
- `law_parameters` becomes a tuple of finite numbers whose length matches the law:
  degrees of freedom and a normal scale are positive, and `draws` is a positive integer.
  With no law it is empty.
- `_status` checks `status` and the reason, as for the other records.

Export `TestRow` in `__all__`.

## Report

- `Report.__init__` gains `tests: tuple[TestRow, ...] = ()` as its last parameter,
  and the class gains the field. `__post_init__` checks the record type, owns a
  tuple and requires unique test names.
- `_repr_html_` bounds `tests` to the preview size like the other sections.

## Rendering

Labels from `labels=` apply to test names exactly as to parameter and metric names.

**Text.** After the parameter rows and before the metrics: a header line
`Test\tStatistic\tReference\tp-value`, then one line per row with `_display`
numbers and the plain `reference` (`unavailable` when there is no law). An
unavailable row adds `name: reason (status=...)`. A nonempty hypothesis adds
`name: hypothesis`.

**HTML.** A second table with columns Test, Statistic, Reference, p-value and
Status, escaped like the parameter table, followed by the hypotheses as paragraphs.

**LaTeX.** A table of tests follows the parameter table, separated by `\par\medskip`
when another table precedes it. A metrics-only table follows the tests the same way.

```latex
\begin{tabular}{@{}lS[table-format=1.5,group-digits=integer]cr@{}}
\toprule
Test & {Statistic} & Reference & $p$ \\
\midrule
jarque\_bera & 1.39938 & $\chi^2(2)$ & $0.497$ \\
breusch\_pagan & 5.03280 & $\chi^2(2)$ & $0.081$ \\
\bottomrule
\end{tabular}
```

- The Statistic column follows the table cell rules, including the computed
  `table-format`, braced scientific cells and `{---}` for an unavailable value.
- The p-value column uses the parameter p-value rule: `$<0.001$`, otherwise
  `f"${p:.3f}$"`, and `---` when absent.
- Reference column, by law. A number prints as its integer when it is integral,
  otherwise as `f"{value:.2f}"`.

  | Law | Cell |
  | --- | --- |
  | `chi-square` | `$\chi^2(d)$` |
  | `f` | `$F(d_1, d_2)$` |
  | `normal` | `$N(0, 1)$` when location 0 and scale 1, otherwise `$N(a, b^2)$` |
  | `resampled` | `_tex(reference)` |
  | none | `---` |

- Omit the Reference and $p$ columns when no row has a law.
- With `notes=True`, each nonempty hypothesis adds `\par name: hypothesis` and each
  unavailable row adds `\par name: reason`, before the report's own notes.

**Statistic header for coefficient tests.** In the parameter table, when
`metadata["reference"]` is `"normal"` the Statistic header is `{$z$}`, when it is
`"t"` it is `{$t$}`, and otherwise `{Statistic}`. Apply the same header in text and HTML.

## Snapshot

- `_report_state` writes `"tests": [asdict(row) for row in report.tests]`.
- `_restore_report` requires the `tests` key, restores `law_parameters` as a tuple
  and constructs each row through `TestRow`, so validation repeats on load.
- `_SCHEMA_VERSION` in `api/artifacts/_store.py` moves from 6 to 7. Update every
  literal 6 that names this version, including the one near line 359 of
  `_store.py`, and the tests that assert it (`tests/api/test_artifacts.py`,
  `tests/api/test_ordination_artifacts.py`).

## Compile gate

Extend `src/python/apps/docs/tests/test_reporting_workflow.py` with a report holding
chi-square, F, resampled and unavailable test rows, a descriptive row with no p-value,
a p-value below 0.001 and a hypothesis note. It compiles with `only_cached=True`
and no Overfull box or missing character.
