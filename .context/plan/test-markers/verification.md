# Marker migration verification

The baseline is `dadeeccf4b53d303d61f81c63aa37bf8e540ae05`. Native collection
through `just python::statadic::test-fresh --collect-only -n 0 -m ''` found 3,092
cases before the migration. The registry at `1f77ba2ad0` still identifies all
1,711 core definitions across 206 modules. It supplies starting classifications;
eight source-backed reviews adjudicate the final marker mapping.

The new policy keeps one scope and optional assertion purposes. It retains the
external-reference, qualification, typing and resource selectors independently.
Source-span edits preserve parameterization, test bodies, dependency requirements
and test paths except for the explicitly reviewed assertion corrections.

## Baseline failure

The planning-stage `just check` failed at
`tests/estimation/spatial/test_observed_panel.py::test_observed_information_matches_independent_finite_difference`
for seed `1015`: the generated sample produced an unavailable fit, while the
test asserted `fit.valid`. The marker migration does not alter that assertion
or the estimator. Validation results below distinguish this existing failure
from policy and collection checks.

## Scientific corrections

- The candidate-energy diagonal is within-cloud mean ground distance, including
  self-pairs with divisor `n²`. The test now checks an independent NumPy mean;
  zero is not its general contract.
- The shifted-cloud pairwise-max fixture checks shapes and finite positive
  outputs. A numerical probe found tied maximum scores, so it does not establish
  unique outlier ranking or repeated-sample statistical performance.
- The selected-period spatial-survival report checks a validity flag and retains
  error diagnostics. It is not a gated reference comparison. The neighbouring
  budget test retains its explicit `0.001` error criterion.

## Execution results

The migration preserves every original definition and parameterized case. It
adds six policy definitions expanding to 21 cases. The source audit finds no
missing or repeated scopes or retired markers. It preserves every original
execution marker and dependency requirement, and checks function ASTs to limit
behavioral edits to the reviewed assertion and subprocess-fixture changes.

The baseline contained 360 missing scopes, one conflicting declaration and two
redundant inherited declarations. All are resolved. The reviewed mapping is a
historical migration snapshot; source markers and the test ownership guide own
the continuing policy.

| Native collection | Original cases retained | Current cases |
| --- | ---: | ---: |
| All | 3,092 | 3,113 |
| Fast | 2,928 | 2,949 |
| Live reference | 248 | 248 |
| Qualification | 47 | 47 |
| Typing | 57 | 57 |

All original lane selections match the baseline, including the unchanged slow
classification. Collection does not establish pass/fail results.

- `just python::fmt` passed after addressing the new fixture lint findings.
- The named policy file, energy diagonal, PCoA transform and shifted-cloud checks
  passed together: **51 cases**. The invocation used `just python::statadic::test-fresh`
  with explicit paths, `-m ''` and `-n 0`.
- The named Bartlett and Newey–West comparisons passed the mandatory reference
  gate: **2 cases**, using the prepared locked reference environment and
  `just python::statadic::test-external tests/migration/test_t75_3_statsmodels.py`.
- A temporary copy of the tests and configuration collected all **57 typing
  cases** with the layout plugin disabled and marker policy active. This checks
  copied-harness portability using the development installation; it does not
  establish installed-wheel origin or distribution qualification.
- `just sync` and `git diff --check` passed.

The final `just check` passed repository format/link/schema checks, Lean checks,
Python architecture, test layout, Ruff, Pyrefly and all seven isolated shape
tests. Its Testmon-selected fast suite finished with **2,941 passed, 7 skipped
and 1 failed**, in 521.04 seconds. The sole failure is the unchanged
observed-information case above, again at seed `1015`, where `fit.valid` is
false. The source audit confirms its function body is unchanged. No new
marker-policy or corrected-assertion failure occurred.

The check stopped at that failure. The later Python build, evidence, publishing,
studies, performance and documentation app checks, plus LaTeX and agent checks,
were not reached. No full suite was run outside the repository check, and no
long-running statistical qualification or fresh installed-distribution check
was executed.
