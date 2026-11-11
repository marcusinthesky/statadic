# Validation evidence and execution

Validation belongs to a method, configuration, output and reference version.
Agreement with an external library, an independent numerical witness, an internal
invariance and repeated-sample statistical qualification establish different
claims. A green default suite does not certify every method.

## Numerical and reference checks

Tests retain visible Arrange–Act–Assert phases. Neutral inputs and strict numeric
assertions may be shared; independent references do not reuse STATADIC centering,
weighting, residual, covariance, projection or p-value calculations. Family tests
state intercepts, retained samples, weight meanings, covariance corrections,
degrees of freedom, objective and null-law conventions explicitly.

`reference` identifies a comparison with independent software, an analytic
result, a published value, or a trusted fixture. `external_reference` identifies
a live comparison with an optional software dependency and also carries
`reference` and `requires_optional(name)`. An independent oracle does not by
itself make a test integration scope; scope follows the boundary exercised by
the assertions. These markers describe assurance and select tests, not a
coverage denominator. Missing optional dependencies skip locally. Broken
installed dependencies fail. The required lane rejects absent comparators,
empty selections, skips, xfails and unexpected passes of expected-failure
comparisons.

From the repository root:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.cache/statadic-reference-venv"
uv sync --locked --package statadic --group reference --project src/python
just python::statadic::test-external
just python::statadic::test-fresh
```

The locked lifelines AFT oracle is 0.30.0, tested with the existing pandas 3
dependency. Lifelines 0.30.3 comparisons also passed in an isolated pandas 2
environment. That newer release caps pandas below 3; the pinned Nix resolver
does not support the resulting project-conflict fork. The qualified 0.30.0
reference avoids a production downgrade or a package-metadata override. Keep
reference environments separate to avoid pruning the shared workspace.

The test process retains the legacy x64 default only when `JAX_ENABLE_X64` is
unset. Run matched, strict references in separate caller-configured processes:

```bash
JAX_ENABLE_X64=false just python::statadic::test-precision-references --junitxml=precision-float32.xml
JAX_ENABLE_X64=true just python::statadic::test-precision-references --junitxml=precision-float64.xml
```

The precision lane compares linear fitting, Student-t intervals and their
zero-exclusion decisions, tied Pearson/Spearman correlations, and weighted energy
distance with SciPy. Statsmodels supplies the WLS/HC3 and Poisson coefficient and
offset/exposure prediction comparisons, using the same quantized inputs.
The WLS comparison mixes float32 design/weights with the process-selected outcome
precision. Scaled exact transport is compared with SciPy assignment on the same
represented costs. A scaled GMM coefficient and covariance witness uses SciPy
least squares and independently formed residual variability; both isolated
processes exercise the formerly truncated coefficient direction.
The same processes run Weibull/lognormal AFT comparisons with lifelines:
coefficients, total log likelihood, full joint observed-information covariance,
survival, cumulative hazard, CDF quantiles and independently integrated RMST.
Right censoring, delayed entry and unbalanced repeated-spell fixed-effect designs
are explicit fixtures. Spatial log-time survival additionally has complete-event
spreg reductions, fixed-zero-rho reductions and independent joint Gaussian-tail
witnesses; lifelines does not implement that joint dependent likelihood.
The lane attempts bfloat16 correlation and the actual fitting backend: unsupported
QR dtype is recorded in JUnit rather than skipped; a supporting backend must
produce the checked result. Precision jobs report separate selected-node counts,
not coverage denominators. These finite examples do not establish statistical
calibration, all-dtype support, or GPU behavior. The full locked reference lane
still runs with its legacy x64 default; precision runs do not replace it. Caller
configuration is respected without library-side precision mutation or hidden casts.

The full reference lane covers additional method-specific configurations. These
entry points show the actual comparisons and their conventions; each test's
scope remains narrower than its family name.

The required CI reference jobs use the same locked dependencies in an isolated
environment, independently of the ordinary fast gate. Run their complete local
equivalent from the repository root:

```bash
just setup-references
just validate-references --output build/reference-validation
```

The output directory must be empty. Each of the full float64 reference lane and
the float32/float64 precision subsets retains its log, JUnit report and selected
nodes. The manifest records dependency versions, the pinned layout-enforcer Git
origin, caller precision and comparison counts by comparator. An empty selection,
failed/skipped comparison or missing spreg, statsmodels, linearmodels or lifelines
comparison fails validation. Precision subsets require statsmodels and lifelines
comparisons separately. These are live parity checks, not statistical qualification.

The repository command also supplies the authored catalogue. Each declared
comparison requires its exact parameterized node and comparator; the manifest
retains unmapped executions separately. Source, tests, executed recipes, lock and
catalogue are fingerprinted before and after the run. Independent-formula
mappings outside these lanes remain explicitly unexecuted by this command.

Repository branch protection should require both
`Reference validation (ubuntu-24.04)` and
`Reference validation (ubuntu-24.04-arm)` CI statuses.

The generated [reference index](validation/references.md) is the configuration-to-
test map for constructors and explicit compositions. It retains exact checked
outputs and conventions, links the current test sources, and exposes unmapped
scientific variants separately from method-level numerical comparisons.

Constant-correlation shrinkage comparisons use PyPortfolioOpt with its sample
covariance normalized to the T divisor **before fitting**. They assert target,
intensity and covariance; sklearn's scaled-identity target is a different method.
Panel peers require linearmodels 7.1. RE conditional-mean effects are checked
independently rather than equated with quasi-demeaned reference effects. Panel AC
uses actual integer time distances; compressed observed-row lags are a different
contract on panels with interior gaps.

Dedicated IV Hansen-J comparisons match the 2SLS initial weight, one estimated
update or converged iteration, plain/centered bandwidth-zero variability, actual
generating W, coefficients, J, degrees of freedom and tail with linearmodels.
The separate affine-GMM tests do not grant coverage to that producer. Ordered
BG/DW controls compare sorted complete rows with statsmodels; BG retains all n
rows with zero pre-sample residuals. Ordered OLS/WLS HAC compares complete
influence rows and an explicitly matched correction. These numerical controls
do not establish dependence assumptions, source fidelity or Monte Carlo size.

Property profiles are selected with native `--hypothesis-profile`: `statadic`
preserves each `@examples(N)` cost tier, `statadic-reference` caps it at 20, and
`statadic-explore` multiplies it by five. Untiered defaults are 100, 20 and 500.
Profiles print replay blobs and leave deadlines disabled and health checks active.
Required semantic modes have separate test nodes; random generation searches
within valid domains. Important counterexamples remain literal regressions.
Hypothesis examples are not independent Monte Carlo draws.

## Statistical qualification

Run the scheduled study set independently of the fast tier:

```bash
just python::statadic::test-qualification -k 'not dyadic' --junitxml=qualification.xml
```

Dyadic qualification is manual-only and remains unexecuted. Its two node-size
cells each require 10,000 independent trials with 999 production node-bootstrap
draws per trial. Warm measurements project about 8 hours per cell; allow at least
that time. Two benchmark trials establish runtime, not interval coverage. Run
the complete canonical study explicitly:

```bash
just python::statadic::test-qualification \
  tests/qualification/test_panels.py::test_dyadic_complete_node_bootstrap_coverage \
  --junitxml=qualification-dyadic.xml
```

Each scenario specifies its DGP independently of the implementation and uses
10,000 independent trials. Stable scenario/trial/stream IDs separate data,
resampling and optimization randomness. Randomized inference uses 999 inner
draws unless a documented family contract requires otherwise. Neither worker
identity nor Python's randomized hash chooses seeds.

Default significance is 0.05. A one-sided 95% exact binomial upper bound for
failure-inclusive rejection must be at most 0.06; the failure-rate upper bound
must be at most 0.005. For nominal 95% intervals, the coverage lower bound must
be at least 0.94, counting unavailable intervals as uncovered. Results distinguish
qualified, excess-error and inconclusive scenarios. Inconclusive results do not
satisfy qualification. Bounds apply per scenario, not simultaneously across all
methods. Power is reported with uncertainty without imposing sampled monotonicity
or an unadvertised power guarantee.

JUnit properties retain requested/valid/failed counts, conditional rejection,
failure-inclusive bounds and qualification status. Sampler-law checks additionally
use independent marginal/joint distributions, so a constant or underspread sampler
cannot qualify merely by being conservative. Covariance risk studies report loss
uncertainty rather than assuming every sample benefits from shrinkage.

### Ordinary AFT qualification

The prespecified [survival study](../tests/qualification/test_survival.py) ran
10,000 independent trials per cell in float64 on CPU. The two normal covariates
have coefficients `(0.4, -0.3)`, with intercept `0.2`, Weibull shape `1.5` or
lognormal scale `0.7`. Positive entry is Uniform(0,1), with event times sampled
conditional on surviving entry; censoring is entry plus Exponential(rate=0.5).
The full fitted observed-information Wald procedure counts failures as
rejections and uncovered intervals.

| Family | n | Entry | Valid / requested | Rejected | Coverage lower bound | Disposition |
| --- | --- | --- | --- | --- | --- | --- |
| Weibull | 128 | Zero | 10000 / 10000 | 551 | 0.941000 | Qualified |
| Lognormal | 128 | Zero | 10000 / 10000 | 589 | 0.937081 | Inconclusive |
| Weibull | 512 | Zero | 10000 / 10000 | 544 | 0.941722 | Qualified |
| Lognormal | 512 | Zero | 10000 / 10000 | 515 | 0.944717 | Qualified |
| Weibull | 128 | Positive | 9999 / 10000 | 564 | 0.939556 | Inconclusive |
| Lognormal | 128 | Positive | 10000 / 10000 | 557 | 0.940381 | Qualified |
| Weibull | 512 | Positive | 10000 / 10000 | 504 | 0.945853 | Qualified |
| Lognormal | 512 | Positive | 10000 / 10000 | 525 | 0.943684 | Qualified |

These are cell-specific results, not a blanket survival qualification. Both
inconclusive assertions remain failing; no tolerance, seed or protocol was
changed to turn them green. Fixed-effect repeated-spell and dependent spatial
Wald inference have not been statistically qualified by this iid study.

## Scientific limits and CI

The 500-node, twenty-period spatial-survival diagnostics use an independent
one-factor Gaussian quadrature oracle. At eight replicates of 1,024 draws, both
10% censoring fixtures meet the 1e-3 per-period log-probability target; the 50%
and 80% fixtures fail it for both changing populations and retained full graphs
with missing outcomes. Maximum period errors are about 0.07 and 1.03 respectively.
For a selected 50% period, two independent keyed streams still miss the target at
65,536 draws per replicate (errors about -0.0061 and -0.0027). The failing
assertions remain in the numerical stress suite. These preliminary runs lack a
pre-run source manifest and do not establish final-source qualification.

A separate informative 500-node cycle, twenty-period, 10%-censored fit converged
in 86 iterations with fixed 8x1024 draws. It deliberately omitted information;
convergence of that candidate does not establish integration accuracy or
spatial Wald calibration. High-dimensional censoring needs further numerical
qualification before statistical qualification is meaningful. Use explicit
refinement evidence; finite output and replicate error estimates are insufficient.

Eager/JIT/vmap agreement is internal consistency, not independence. Gradient tests
apply only to advertised smooth domains. Eigenvectors, transport plans and
optimizer weights need not be unique; compare identified subspaces, feasibility
and objectives. PSD is required only for estimators that promise it. Solver
convergence, parameter identification and inferential validity remain distinct.

PRs retain the complete fast suite, locked external comparisons and separate
float32/float64 precision-reference processes. Weekly
and manually dispatched Actions runs execute the non-dyadic statistical studies
in the locked environment; both exclude dyadic studies. Current-reference
exploration runs in an isolated package copy, records its environment and does
not rewrite the reviewed workspace lock. No GPU qualification or recovery of
retired simulation/paper experiments follows from these CPU tests.

The October qualification records expose finite-sample limitations rather than a
blanket certification. Native Gaussian SAR Wald inference at 48 units exceeds
the agreed size budget; the 96-unit cell is inconclusive. A stationary-bootstrap
mean interval at T=256 is inconclusive under IID normal data and undercovers an
AR(1) process with coefficient 0.3. These cells retain their failing qualification
assertions. Numerical peer agreement does not remove these inference limits.

The directly invoked public Sharpe CI at T=128 covered 9,312/10,000 Gaussian,
9,335/10,000 skewed and 8,711/10,000 dependent-return trials. The dependent cell
uses stationary AR(1) returns with coefficient 0.5. Each study used 999 inner
draws and had zero unavailable trials; all three reported excess error under
the prespecified coverage criterion.

### Interpreting the adverse cells

The adverse-cell follow-up separates implementation correctness from the
finite-sample performance of the retained methods. Its numerical witnesses and
paired diagnostic comparisons do not change the qualification thresholds or
choose new defaults.

For SAR, Gaussian maximum likelihood uses `SSE / N` for the error variance and
joint expected information for asymptotic uncertainty. An independent optimizer
and matched spreg fits check those conventions. Small-sample downward scale
estimation, parameter bias and the Wald approximation can still produce narrow
intervals. The existing optional debiasing setting is a different inference
configuration; a small diagnostic improvement does not qualify it.

For stationary-bootstrap means, the exact conditional variance follows from the
circular Markov resampling law. Under the T=256 Gaussian AR(0.3) design, its
expected variance can be substantially below the true finite-sample mean
variance even with a population-based block length and infinitely many draws.
That establishes a finite-sample variance mechanism, not an exact coverage bound.
Increasing the number of bootstrap draws alone cannot remove that variance
attenuation.

The public Sharpe interval is a percentile interval for sample mean divided by
sample standard deviation (`ddof=1`). It has no studentizing or HAC correction
in its endpoints. Independent fixed-schedule NumPy endpoints check arithmetic;
the Gaussian noncentral-t control checks the population target. Automatic block
selection uses returns, rather than the Sharpe influence series, and can choose
lengths above the observation count. Paired comparisons with other selectors
or fixed lengths diagnose sensitivity; they do not establish a replacement
method or default.

These exact designs lack a demonstrated coverage guarantee under the agreed
budget. The two inconclusive results remain uncertified; they are not proof of
either adequate calibration or excess error. No universal minimum sample size,
dependence cutoff or broader unsupported region follows from the seven cells.
Their qualification assertions remain active. Diagnostic studies use separate
held-out trial IDs and are not marked as qualification.

From the repository root, replay the seven original cells separately:

```bash
just python::statadic::test-qualification tests/qualification/test_panels.py -k gaussian_sar
just python::statadic::test-qualification tests/qualification/test_resampling.py -k stationary
just python::statadic::test-qualification tests/qualification/test_returns.py -k direct_sharpe
```

An assertion failure from these commands remains a statistical qualification
failure, including after the investigation explains or bounds the method's
scope. It must not be converted to a skip or expected failure.

Run the separate held-out mechanism diagnostics with the reference group
installed:

```bash
just python::statadic::test-all tests/qualification/test_sar_diagnostics.py \
  tests/qualification/test_stationary_diagnostics.py \
  tests/qualification/test_sharpe_diagnostics.py -n 0 -o junit_family=legacy \
  --junitxml=adverse-diagnostics.xml
```

The bootstrap diagnostics compare six methods on 1,000 held-out datasets per
original DGP, with shared randomness and 999 resamples. Their reports retain
unavailable lanes, directional misses and paired differences; passing a
diagnostic test does not qualify any of those methods. SAR uses 256 held-out
datasets per design to examine bias and scale, with sixteen separate matched
software/likelihood witnesses.

Exactly constant returns now produce zero sample Sharpe and undefined (NaN)
skewness/kurtosis, even when roundoff leaves a tiny computed standard deviation.
Stable callable identities and module-level kernels also reuse compilation in
Sharpe stationary bootstrap, mixture/equivalence resampling and active-set
polishing. These execution changes retain the existing statistics, random-key
schedules and inference rules; the finite-sample limits above remain explicit.
