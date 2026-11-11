# Performance workload definitions

These links describe executable CPU workloads, not measured speedups. Run
`just python::performance::smoke --output PATH` for a bounded check, or
`just python::performance::bench PATH` for a retained campaign. Each output
directory must be new. The runner executes the listed families together, with
explicit precision and matched measurement conditions.

(performance-discrepancy)=

## Distribution discrepancies

Unequal-support point clouds use the canonical `PackedClouds` carrier. Timed
variants construct that carrier from fixed-shape inputs inside the compiled
call; these measurements include that construction.

(performance-discrepancy-mmd-value-gradient)=

### RBF MMD value and input gradient

`discrepancy/mmd-value-gradient` evaluates V-statistic MMD squared and its
coordinate gradient, with RBF bandwidth one. It does not measure a fitted-model
derivative or qualify the signed U-statistic.

(performance-discrepancy-energy-euclidean)=

### Raw Euclidean energy

`discrepancy/energy-euclidean` evaluates raw V-energy with explicit Euclidean
ground. It does not measure the rooted host `Energy` workflow.

(performance-ols)=

## Least squares

All variants fit the same complete-data least-squares problem.

(performance-ols-full-fit)=

### Full fit

`ols/full-fit` retains the complete native fitted state and diagnostics.

(performance-ols-parameters-projection)=

### Parameters only

`ols/parameters-projection` returns only coefficients. Compiler elimination of
unused outputs can change its cost; this is not a separate least-squares solver.

(performance-ols-fit-and-inference)=

### Fit and classical covariance

`ols/fit-and-inference` returns coefficients and their classical covariance.
Other covariance estimators and reference laws are outside this workload.

(performance-row-refits)=

## Supplied-row refits

Every variant uses the same frozen row-resampling schedule. Chunk size changes
execution grouping, not draw identities or the statistical procedure.

(performance-row-refits-chunk-1)=

### One draw per chunk

`row-refits/chunk-1` is present in smoke and retained benchmark campaigns.

(performance-row-refits-chunk-2)=

### Two draws per chunk

`row-refits/chunk-2` is a smoke-only variant.

(performance-row-refits-chunk-8)=

### Eight draws per chunk

`row-refits/chunk-8` is present in retained benchmark campaigns.

(performance-row-refits-chunk-32)=

### Thirty-two draws per chunk

`row-refits/chunk-32` is present in retained benchmark campaigns.

(performance-graph-profile)=

## Projected Wasserstein graph likelihood

The full workload has 100 firms, 128 equal-mass articles per firm and four
outcome periods. Each two-dimensional article vector is projected onto a line.
Both variants sort the projected samples, form squared Wasserstein distances,
apply off-diagonal RBF weights and profile the Gaussian SAR mean and variance
at a fixed spatial coefficient.

(performance-graph-profile-profile-value)=

### Profile log likelihood

`graph-profile/profile-value` returns the Gaussian profile value.

(performance-graph-profile-profile-value-angle-gradient)=

### Profile value and projection derivative

`graph-profile/profile-value-angle-gradient` differentiates the full calculation
with respect to the projection angle. An independent NumPy calculation checks
the value and derivative away from sorting ties.

## Methods exercised

<!-- statadic:performance-catalogue -->

## Current-wheel matched references

The separate `just python::performance::reference` lane compares an installed
candidate wheel with independent peer processes. It prepares identical inputs
before timing, then measures the complete numerical consumer outputs below.
Prepared empirical carriers keep coordinates and masses dynamic. The public
target-projection and ordered dyadic-rank consumers remain eager; the runner
does not introduce a different traced procedure to improve their timing.

The executable definitions are
[`reference_workloads.CASES`](https://github.com/marcusinthesky/statadic/blob/main/src/python/apps/performance/src/statadic_performance/reference_workloads.py),
[`reference_fixtures`](https://github.com/marcusinthesky/statadic/blob/main/src/python/apps/performance/src/statadic_performance/reference_fixtures.py)
and [`reference.QUALIFICATION_CASES`](https://github.com/marcusinthesky/statadic/blob/main/src/python/apps/performance/src/statadic_performance/reference.py).
They freeze exact inputs, sizes, conventions, output vectors and required
subcases. These workload contracts do not cover every statistical variant in
their named families.

| Required workload | Enumerated consumers and independent references |
| --- | --- |
| `linear-fit-and-inference` | OLS and relative-precision WLS coefficients plus HC3 covariance; statsmodels |
| `canonical-glm` | Grouped-binomial logit with offset and Poisson with exposure/offset: fit, model covariance, likelihood and means; statsmodels |
| `dispersion-likelihood` | Gamma and NB2 joint estimated dispersion: coefficients, physical dispersion, likelihood and physical-coordinate observed information; independent SciPy likelihood/analytic score and information differences |
| `covariance-shrinkage` | Sample covariance and constant-correlation Ledoit–Wolf target, weight and estimate; independent NumPy moments. This is not sklearn's identity target. |
| `dyadic-node-refits` | Supplied multinomial node counts with and without node effects: ordered rank selection, focal coefficient, SSE, overall/within R² and active counts; independent weighted SVD refits |
| `affine-gmm` | Fixed non-efficient actual W and three centered Bartlett-HAC updates: fit, retained actual W, final HAC covariance and criterion; independent SciPy SVD/Bartlett calculations |
| `weighted-mmd-energy` | Unequal weighted RBF V-MMD and Euclidean exponent-one V-energy values plus coordinate gradients; sklearn kernels and independent analytic gradients |
| `empirical-transport` | Unequal weighted real-line W1 and squared W2; SciPy W1 and independent monotone coupling. Assignment, sliced and Sinkhorn performance are outside this contract. |
| `measure-projections` | Energy and RBF target-simplex mixtures: weights and original objective; independent SciPy SLSQP over separately calculated distance/kernel means |
| `qp-gmv` | Capped-simplex quadratic weights and objective; SciPy SLSQP |
| `supplied-row-refits` | Initial OLS and all supplied aligned row-pairs coefficient refits; independent NumPy SVD solves |
| `spatial-operators` | Fixed-graph SLX fit/classical covariance/fitted values and Gaussian SAR fit/expected covariance/fitted/reduced means; statsmodels and spreg |
| `spatial-panel` | Time-varying-W unit-effect SAR fit and expected information, plus changing-roster/full-graph-missing pooled SAR fit and observed information; independent period-block Gaussian calculations |
| `ordinary-aft` | Right-censored Weibull/lognormal, ordinary and delayed entry: fit, likelihood and observed information; independent SciPy likelihoods/analytic derivatives |
| `joint-censored-sar-aft` | **Unsupported matched reference:** full 500-unit ×20-period fit and identified observed information. Small integration or objective-only pilots do not replace it. |
| `ordination-fit-transform` | Kernel PCA and Isomap training and held-out coordinates; scikit-learn with matched centering, bandwidth or union-neighbor paths and frozen queries |
| `geometric-regression-permutations` | Observed and supplied Freedman–Lane partial statistics and p-value; independent NumPy projections |

Registration enables measurement; it establishes no parity, scientific or
resource outcome. The current alpha selects `report`: completed matched sessions
for every requested workload, numerical agreement and the actual executed case
list, with all adverse ratios visible. A bounded case does not establish
full-family coverage. The legacy/default `superiority` policy additionally
requires every registered subcase, three independent sessions, five warmups,
thirty samples and warm-time and total-process RSS ratios at most `0.90` in each
session. Process RSS includes imports, compilation and retained buffers; it is
not import-subtracted workload memory. Cold/first-call costs remain separate.
The unsupported joint workload still leaves the full roster incomplete under
either policy.

## Interpret retained measurements

Compilation, first-call latency, synchronized warm execution and host transfer
are separate measurements. Process peak RSS includes imports and compilation;
compiler buffer estimates are not an allocator peak. Independent numerical
witnesses run outside timing. Compare only matched campaigns with
`just python::performance::compare BASELINE CANDIDATE`; results do not transfer
to other hardware, shapes, precisions or statistical configurations.
