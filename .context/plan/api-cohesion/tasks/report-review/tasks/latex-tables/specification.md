# Table rendering specification

Scope: `Report.to_latex`, the new `Report.save_latex`, `_display` and `_tex` in
`src/python/packages/statadic/src/statadic/api/reporting.py`. `PlotSpec` and its renderers
belong to `pgfplots-output` and are not edited here. No dataclass field changes.

## Signatures

```python
def to_latex(
    self,
    *,
    digits: int = 6,
    decimals: int | None = None,
    labels: Mapping[str, str] | None = None,
    label_markup: Literal["plain", "tex"] = "plain",
    notes: bool = True,
) -> str: ...

def save_latex(self, path: str | Path, *, <the same keywords>) -> None: ...
```

`digits` must be a positive integer and `decimals` a nonnegative integer or
`None`; otherwise raise `ValueError`. `save_latex` writes `to_latex(...)` to
`path` as UTF-8 with `\n` newlines and creates no directory.

## Cell text

| Value | Text |
| --- | --- |
| `None` | Unavailable marker: `{---}` in an `S` column, `---` elsewhere |
| `int` (never `bool`) | Exact digits. Never rounded, never an exponent |
| `float`, `decimals` given | `f"{value:.{decimals}f}"` |
| `float`, otherwise | Start from `format(value, f"#.{digits}g")`, which keeps trailing zeros |

For the significant-digit case:

- A negative exponent gives a scientific cell: `$m \times 10^{e}$` with the mantissa as formatted.
- A positive exponent gives `f"{value:.0f}"`: every integer digit and no decimals.
- Otherwise use the text, removing a final `.`.

A scientific cell is braced in an `S` column, `{$1.20 \times 10^{-9}$}`, because
siunitx drops an exponent that the column's `table-format` does not declare.

The parameter p-value column ignores `digits` and `decimals`: below 0.001 print
`$<0.001$`, otherwise `f"${value:.3f}$"`.

## Columns

- Label column: `l`. Wrap the whole specification in `@{}` on both sides.
- Estimate, SE, Statistic, Lower and Upper: `S[table-format=F,group-digits=integer]`.
  `F` comes from that column's plain decimal cells: a leading `-` if any is
  negative, the largest count of integer digits, then `.` and the largest count
  of decimal digits when that count is positive. The Estimate column also counts
  the `float` metric cells it will hold as footers. A column with no plain
  decimal cell is `c`.
- p-value: `r`.
- A column other than Estimate is omitted when every row's value is `None`, as today.

## Header

Header text in an `S` column is braced: `{Estimate}`, `{SE}`, `{Statistic}`,
`{Lower}`, `{Upper}`. The p-value header is `$p$`.

When both interval columns are present and `metadata["confidence"]` is a real
number, write two header rows: the first spans the interval columns with
`\multicolumn{2}{c}{95\% CI}` (level from `_confidence_label`), then
`\cmidrule(lr){a-b}`, then a row holding only `{Lower}` and `{Upper}`.
Otherwise write the single header row.

## Metrics

- With parameters: after the parameter rows write `\midrule`, then one row per
  metric with its value in the Estimate column and the other cells empty. An `int`
  is `{\num{N}}`; a `float` follows the cell rules; unavailable is `{---}`.
- Without parameters: one table `@{}lS[...]@{}` headed `Metric & {Value}` with the
  same cell rules.

## Labels and text

`labels` maps a parameter or metric name to its display label. Entries that match
no row are ignored, so one mapping serves several reports. With
`label_markup="plain"` a label is escaped by `_tex`. With `"tex"` it is inserted
unchanged and a newline in it raises `ValueError`. A name without an entry is
always escaped.

`_tex` also rewrites characters the pinned fonts lack. Use math mode or ASCII
only, never `\textsuperscript` or a font-size command:

| Character | TeX |
| --- | --- |
| `—`, `–`, `…` | `---`, `--`, `\ldots{}` |
| `−`, `×`, `·`, `±` | `$-$`, `$\times$`, `$\cdot$`, `$\pm$` |
| `≤`, `≥`, `≠`, `≈`, `∞`, `→` | `$\leq$`, `$\geq$`, `$\neq$`, `$\approx$`, `$\infty$`, `$\to$` |
| `²`, `³` | `$^{2}$`, `$^{3}$` |
| Greek letters | `$\alpha$` … `$\omega$`, `$\Gamma$` … `$\Omega$`; omicron becomes `o` |

## Notes and comments

With `notes=True`, keep today's lines after the table: one `\par` line per
unavailable reason, metric definition, report note and warning. With
`notes=False`, omit them all. The first line stays `% <title>` and the last stays
the `%` metadata comment.

## Text and HTML

`_display` prints an `int` exactly and a `float` as today (`.6g`).

## Example

Illustrative; the rules above are normative.

```python
report = Report(
    "Poisson inference",
    parameters=(
        ParameterRow("intercept", -0.49421, std_error=0.060761,
                     ci_lower=-0.61330, ci_upper=-0.37512),
        ParameterRow("load", 0.29980, std_error=0.016360,
                     ci_lower=0.26773, ci_upper=0.33187),
    ),
    metrics=(Metric("n_observations", 400),),
    metadata={"confidence": 0.95},
    notes=("95% intervals; normal reference; covariance HC1",),
)
report.to_latex(digits=4, labels={"load": "Facility load",
                                  "n_observations": "Observations"})
```

```latex
% Poisson inference
\begin{tabular}{@{}lS[table-format=-1.4,group-digits=integer]S[table-format=1.5,group-digits=integer]S[table-format=-1.4,group-digits=integer]S[table-format=-1.4,group-digits=integer]@{}}
\toprule
Parameter & {Estimate} & {SE} & \multicolumn{2}{c}{95\% CI} \\
\cmidrule(lr){4-5}
 & & & {Lower} & {Upper} \\
\midrule
intercept & -0.4942 & 0.06076 & -0.6133 & -0.3751 \\
Facility load & 0.2998 & 0.01636 & 0.2677 & 0.3319 \\
\midrule
Observations & {\num{400}} & & & \\
\bottomrule
\end{tabular}
\par 95\% intervals; normal reference; covariance HC1
% {"confidence": 0.95}
```

## Compile gate

- Add `\usepackage{siunitx}` to `src/python/apps/docs/templates/report.tex`.
- `_compile` in `src/python/apps/docs/scripts/compile_report.py` also fails on
  `Missing character` or `could not represent character` in the engine output.
- Extend the compile test in `src/python/apps/docs/tests/test_reporting_workflow.py`
  with one report holding: a count of at least 1,000,000, a negative estimate, an
  unavailable row, a p-value below 0.001, retained confidence with both interval
  columns, a scientific cell, a Greek-letter name and a `labels` entry. It
  compiles with `only_cached=True` and no Overfull box.
- The vendored bundle holds only fonts that current documents use. If a fixture
  needs another font, change the fixture. Do not run `just latex::bundle-refresh`.
