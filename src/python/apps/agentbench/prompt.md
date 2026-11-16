# Adapted StatLLM research task

Read task.json for the scientific question, variables, adaptations and required
figures. Use the assigned package for fitting, prediction and native inference.
NumPy/pandas may prepare data, construct designs and serialize results; SciPy
quantiles and explicit contrast arithmetic are permitted and must be disclosed.
Do not substitute another estimator, hardcode results, use external services or
install packages. Use double precision. Explore installed official docstrings,
package source and `/docs` when present; no repository agent instructions, gold
solutions, hidden tests or network are available.

The inputs under `/input` are immutable. Column names are lowercase, category
whitespace is stripped, filters and complete cases have been fixed identically
for both packages. Preserve supplied row_id and row order. Do not silently drop
influential observations or add regularization.

## Deliverables

Write **one self-contained `/work/submission/solution.py`**. With `/work` as its
working directory and no arguments it must read `/input/data.csv` and regenerate
`submission/results.json`, `submission/report.tex`, and requested plots. Do not
rely on existing output or helper scripts: clean replays receive only solution.py
and the public inputs. Use absolute `/input` paths, relative `submission` outputs.

JSON must satisfy `/input/schema.json`. The schema gives shapes and names, NOT
expected numerical values. Preserve canonical `parameter_names` in task.json:
intercept, continuous columns, squares, non-reference factor levels in explicit
factor_order, then interaction products. A factor's first listed level is its
reference. Reparameterize equivalent library results before export if needed.
Numbers must be finite, unrounded JSON numbers; no NaN/Infinity or fabricated
fallback values. Fail clearly when a valid result is unavailable.

Use classical OLS covariance SSE/(n-p) times inverse X'X, or inverse information
for unpenalized logit/Poisson. Coefficient intervals: 95% Student t with n-p df for
OLS, normal/Wald for GLMs. `fitted` means the training response means; `residuals`
are observed minus fitted. For a log-transformed response these stay on log scale.
OLS r_squared is centered. Diagnostic arrays use the retained row order.

For ANOVA, test equal-weight factor marginal means over the other factor, not
main-effect dummies at just the reference cell. Interaction tests jointly set
interaction coefficients to zero. Report F, numerator df, denominator n-p, and
p-value. Marginal arrays follow the declared levels; cells use Cartesian product
order over factor_order. Pairwise arrays contain `[difference, lower95, upper95]`
in index-pair order (0,1), (0,2), ..., first minus second, with unadjusted intervals.
Neuralgia instead requires `[odds_ratio, lower95, upper95]` for treatment marginal
log-odds differences, exponentiated with Bonferroni correction for three pairs.

`report.tex` is a LaTeX fragment, not a full manuscript. Include a labeled results
table, required figures and 100–250 words interpreting estimates, uncertainty
and limitations without unwarranted causal or normality claims. Native exporters
are welcome. The fixed wrapper supplies amsmath, booktabs, siunitx, graphicx,
pgfplots and hyperref. Tectonic has a frozen offline cache; no shell escape or
package downloads. Figures may be PNG or PDF under `submission/`.

## Public rubric

Clean reproducible contract: 10; estimates: 40; inference/contrasts: 25;
LaTeX compilation: 10; interpretation/report fidelity: 10; assigned-package
compliance: 5. Numerical comparison uses rtol=1e-4, atol=1e-6 per declared array
or scalar, with equal weight per quantity rather than per row. Human review
checks prose, figures, report/JSON agreement and actual package use. Compilation
alone does not establish any of those. Integrity violations receive zero.
