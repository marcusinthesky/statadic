# Test layout and live-reference verification

The enforcer fork is pinned at
`4a1911f444e30328e7e01a0c885f24486807021b`, whose tree matches the reviewed local
changes. Forty tests passed on Python 3.11, 3.12, 3.13 and 3.14; pytest 8 also
passed on Python 3.11 and 3.13. Both entry points, xdist, wheel/sdist construction
and non-editable metadata were exercised. No PyPI release is required.

The [46-file migration](moves.json) preserves all 2,886 original collected
nodes and adds nine availability/evidence regressions. All 246 original live
comparison nodes remain and completed successfully. The configured gate checks
219 source modules and 141 focused test files; six explicit cross-module/support
categories retain their independent responsibilities.

## Scientific corrections

- Observed-panel seed `744` converges to the spatial boundary `-0.999`, both at
  40 and 256 iterations. It is identified but unavailable by the existing fit
  contract. The regression retains that failure state; the interior seed `19`
  tests covariance and impacts, and generated cases test availability propagation.
- KKP seed `1886`, under spreg 1.9.0's default approximate-gradient L-BFGS-B,
  produced between variance `11.776848451673336` versus STATADIC's
  `11.76861513723185`. Tightening only the oracle optimizer's stopping criteria
  gives `11.768615044654455`. The test retains this seed, both weighting modes,
  spreg's own moments/whitening and the original comparison tolerances.
- The release runner explicitly selects float64 for its outer check. The GLM
  no-import-hook regression now explicitly selects float32 for its child instead
  of incorrectly inheriting that setting. The Wolak face-enumeration witness
  retains `[0, 1e-9]` as an example and uses the actual `1e-10` binding boundary,
  rather than classifying the free positive coordinate as binding at `1e-9`.
  Both failures reproduced and their targeted corrections passed with a float64
  parent environment. Their statistic and existence assertions remain intact.
- A nearly rank-one PCoA cloud exposed an unconditional exact-reconstruction
  assertion despite the documented spectral cut. Its literal counterexample is
  retained. The witness now independently forms the centered-point Gram and its
  spectrum, verifies the full eigenvalue output, and bounds each squared-distance
  error by twice the discarded spectral norm plus eigensolver backward error.
  This preserves the rank policy instead of requiring discarded axes to survive.
- No numerical implementation changed. The MMD source diff updates only a moved
  test's path in a docstring; its API inventory and AST correspondence pin were
  explicitly refreshed after review. Formal verification claims are unchanged.

## Completed reference gate

`just setup-references` prepares a separate locked environment;
`just validate-references` then runs offline and retains evidence under
`build/reference-validation`. Every selected comparison completed without a
skip or xfail. [Collection results](collection.json) retain the node comparison
and declared comparator counts, which are not coverage denominators.

| Lane | Passed |
| --- | --- |
| Complete float64 live references | 246 |
| Float32 precision references | 23 |
| Float64 precision references | 23 |

The full lane includes 35 spreg, 23 statsmodels, 60 linearmodels and 12 lifelines
comparisons, plus the remaining declared independent peers. Component tests can
declare more than one comparator. Forty-one policy/audit tests, both typing lanes
and seven native shape regressions also passed. Both wheel build routes and
extracted-sdist development passed, including 50 typing tests.

The complete fast suite passed: 2,558 tests, with 166 optional skips. Evidence,
studies, performance and documentation tests also passed in the repository
check. The strict site build then identified a link to the test README outside
the Sphinx source tree; it now uses an explicit downloadable reference.

Full repository and installed-wheel validation will be retained by the existing
`just release-preflight` workflow. Its source-free harness omits the Git-only
layout development dependency from the hashed export and disables only that
plugin, after checking source layout before building. Earlier concurrent JAX
runs exhausted RAM/swap; the overlapping repository check was deliberately
stopped and final validation is sequential. An ad-hoc installed-wheel runner's
automatic approval timed out twice before execution; the existing repository
release workflow is the reviewed alternative.

## GitHub merge requirement

The separate CI jobs run on the existing Linux x86-64 and ARM64 runner matrix.
They retain their reports and fail on missing comparator groups, unsuccessful
comparisons or empty selections. Enabling them as mandatory merge checks is an
external administration step: the GitHub ruleset API returned HTTP 403,
`Upgrade to GitHub Pro or make this repository public to enable this feature.`
After resolving that account/repository limitation, require both
`Reference validation (ubuntu-24.04)` and
`Reference validation (ubuntu-24.04-arm)` statuses. No repository visibility or
account subscription was changed.

The long-running statistical qualification limitations remain separate and
unchanged; live parity success does not resolve those studies.
