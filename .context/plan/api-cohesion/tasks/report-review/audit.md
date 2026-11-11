# Report API audit

Every output below came from running the public API in float64 at clean
`9c501be`. LaTeX was compiled offline with the pinned Tectonic and the vendored
bundle. Row IDs refer to the [dispositions](../../proposal-dispositions.md).

## What a manuscript receives today

Fixture: OLS on 12,345 rows with terms `x` and `log_income_2020`, HC3 covariance,
normal reference, `to_latex(digits=4)`.

| Row | Observed | Consequence |
| --- | --- | --- |
| P01 | `n_observations` and `df_resid` print as `$1.234 \times 10^{4}$`. At the default six digits, any count from 1,000,000 is rounded. | A table misstates its sample size. |
| P01 | One column holds `$1.975$` and `$2.01$`: `g` formatting strips trailing zeros and cells are right-aligned. | Precision is misstated and decimals do not align. |
| P01 | An unavailable row prints the word `unavailable` in every numeric cell. | Four words where a dash and one note suffice. |
| P01 | Interval columns are headed `Lower` and `Upper`. The level appears only in a trailing note. | The header does not state the level. |
| P01 | A Wald p-value that underflows prints `$0$`. | No p-value is zero. |
| P02 | Rows are labelled `log\_income\_2020`, `n\_observations` and `wald.restriction\_rank`. No display label can be supplied. | Identifiers stand where manuscript labels belong. |
| P02 | `χ`, `≥`, `—` and `…` in a name or note give `Missing character` under the pinned engine, which `tools/check-log` rejects. | A Greek parameter name fails the manuscript gate. |
| P03 | Both shipped examples write `Path.write_text(report.to_latex())`. | The crucial step has no verb. |
| P04 | The inference table shows no sample size. It appears only in the `%` metadata comment. | A coefficient table without N. |
| P04 | A test outcome prints four lines of `Test component.` beneath its table. | Boilerplate reaches the manuscript. |
| P05 | Nine classes define `summary()` as `return self.report()`. The package `AGENTS.md` names `summary`; the protocol and reference page name `report`. | Two verbs for one act. |
| P05 | `GraphBootstrap.report(level=)` against `InferenceResult.report(confidence=)`. | Two keywords for one quantity. |
| P06 | Coordinates print as `0.10000000000000001`. Empty `xlabel={}`, `ylabel={}` and `title={}` are emitted. Colours are fixed as `blue`, `red` and so on. | Noisy files that a manuscript cannot restyle without editing them. |

## What a Report cannot express

- **P08 Coefficient tests and fit statistics.** `InferenceResult.report` never
  fills `statistic` or `pvalue`; only dyadic regression does. A Wald test of one
  coefficient returns a separate metrics-only report. `examples/ols_analysis.py`
  derives centred R² by hand. The omission of default p-values is deliberate.
- **P09 Test tables.** `TestOutcome.report` and the diagnostics report flatten each
  outcome into metrics named `name.statistic`, `name.pvalue` and `name.component`.
  No row carries statistic, reference law and p-value together, so no renderer
  can set a table of tests.
- **P10 Figures.** The software paper's coverage figure is four separate
  `tikzpicture`s joined by `\hfill`, with a shared range forced through
  `\pgfplotsset`. Sample sizes 25 to 1600 sit at categorical positions because no
  axis is logarithmic. The permutation null is a scatter against draw index where
  a histogram is conventional. Coefficient names are x tick labels, which
  collide when names are long. The vendored bundle lacks the `groupplots` library.
- **P11 Model tables.** `analysis/generate.py` assembles three of the paper's four
  tables by hand. The OLS listing computes HC0 and HC3 inference and cannot set
  them side by side.
- **P12 Inline values.** The same script binds 92 values through 53 hand-written
  `put(...)` calls.
- **P13 Evidence.** The study catalogue holds 309 variants: 123 with citation keys,
  116 with reference mappings and 282 with study mappings. It lives in the studies
  application and is not in the wheel. `TestOutcome.sources` is set for three
  diagnostics only. The publishing example types its claim block by hand.

## Constraints found by compiling the repairs

- The vendored bundle holds only the fonts current documents use. `\footnotesize`
  and `\textsuperscript` fail offline. Generated fragments must not select a font size.
- A plain `article` without `fontenc` and `lmodern` lacks script-size Computer Modern
  (`cmr7`, `cmsy7`), so `10^{-9}` fails there. The docs template's T1 and Latin
  Modern preamble compiles the full target table.
- siunitx drops an exponent silently when the column's `table-format` has none.
  Exponent cells must be braced math, outside the `S` parser.
- `\addplot+` with `forget plot` takes the cycle entry of the series that follows,
  and `cycle list shift` advances it past an unlabelled series. Ribbons filled
  with `fill=.` match their series. Both were compiled and inspected.
- Snapshots keep `int` and `float` apart, so integer counts survive a round trip.
