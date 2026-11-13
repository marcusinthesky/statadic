# Projected spatial graph evidence

## Independent fitted-state comparison

The `test_graph_likelihood_reference.py` tests compare the dynamic graph estimator against actual PySAL `spreg` 1.9.0 for matched Gaussian ML problems. They use a fixed two-dimensional coordinate draw, 24 units, off-diagonal RBF weights on squared Euclidean distance at bandwidth 0.7, and the same outcomes and covariates on both sides. STATADIC constructs the weights with `rbf_weights`; PySAL receives an independently constructed SciPy row softmax. Row normalization makes the weights generally asymmetric. A distinct directed ring supplies an exogenous lag of the covariate as a second design column in both estimators.

| Design | PySAL reference | Quantities checked |
| --- | --- | --- |
| Single cross section, common intercept, SAR or SEM | `ML_Lag`, `ML_Error` | Intercept, slopes, spatial coefficient, ML innovation variance, log likelihood |
| Five balanced periods, unit fixed effects, time-invariant W, SAR or SEM | `Panel_FE_Lag`, `Panel_FE_Error` | Slopes, spatial coefficient, ML innovation variance, log likelihood |

The joint panel likelihood is evaluated in time-major order. STATADIC includes unit dummies and PySAL demeans each unit. The panel oracle's reported likelihood uses `log(SSE)`, requiring the additive `NT log(NT)/2` conversion to STATADIC's `log(SSE/NT)` convention. All four comparisons passed in float64, with absolute tolerances of 0.000002 for spatial coefficients and 0.0000002 for likelihoods. These comparisons cover point fits with exogenous supplied graph geometry. PySAL's single fixed graph does not provide an oracle for a continuously estimated projection, multiple endogenous lags or observed-subvector marginalization. See [PySAL spatial regression](https://pysal.org/spreg/api.html) and [Elhorst (2003), *Specification and Estimation of Spatial Panel Data Models*](https://doi.org/10.1177/0160017603253791).

The repository's `just validate-references` command selects these four tests in its full float64 lane. Each test declares live `spreg` and `libpysal` dependencies. The required-comparator checks turn missing packages, skips and expected failures into gate failures. The precision-only lanes cover different comparators.

## Conditional size and power

The registered study `spatial.graph-selection-power` uses 20 units with four equal-mass articles per unit. Candidate projections retain coordinate one or two. Sorting the four coordinates gives exact one-dimensional squared Wasserstein distances. Off-diagonal RBF weights use one quarter of the median off-diagonal squared distance. The first coordinate generates the SAR field. A separate directed ring supplies an SLX mean effect. All dates share the geometry; innovations are independent Gaussian and covariates are fixed across simulation replicates.

Each replicate makes a two-sided likelihood-ratio decision on four inference dates. One procedure prespecifies the generating projection. The other chooses between both projections using four earlier independent dates, then freezes that choice. Both refit spatial coefficients, nuisance slopes and variance on the inference dates. The nominal 5% cutoff uses the asymptotic one-degree-of-freedom chi-square reference. The test concerns the SAR coefficient conditional on the calibration decision. A projection selected on the inference outcomes would require a different reference distribution. See [Wilks (1938)](https://doi.org/10.1214/aoms/1177732360).

The evaluation used seed 20261107 and 10,000 replicates per scenario. The campaign verifier checked all 30,000 retained records. Parentheses give one Monte Carlo standard error in percentage points. The last column compares rejection decisions within jointly valid pairs on the same simulated data.

| Generating rho | Prespecified projection, rejection % | Training-selected projection, rejection % | Prespecified minus selected, percentage points |
| --- | ---: | ---: | ---: |
| 0 (size) | 5.74 (0.23) | 5.83 (0.23) | −0.08 (0.23) |
| 0.25 (power) | 31.47 (0.46) | 24.93 (0.43) | 6.53 (0.29) |
| 0.50 (power) | 89.38 (0.31) | 87.18 (0.33) | 2.20 (0.15) |

Both procedures exceeded the nominal 5% size in this design. Choosing the graph on four independent dates reduced power most at the weaker alternative. All prespecified fits returned valid decisions. One selected-graph replicate at rho zero and two at rho 0.25 failed the convergence criterion. The size column counts a failure as a rejection and the power column counts it as a nonrejection. Valid pairs numbered 9,999, 9,998 and 10,000 respectively. These are conditional results for two discrete candidates and this fixed geometry. They do not establish size for a continuous projection optimized on the same outcomes.

## End-to-end gradient cost

The `graph-profile` workload projects two-coordinate article embeddings onto a learned line, sorts the projected articles, constructs all pairwise squared Wasserstein costs, row-normalizes RBF weights, and evaluates pooled Gaussian SAR log likelihood at rho 0.35. The full fixture has 100 firms, 128 articles per firm and four periods. The two-coordinate input isolates the projection and graph machinery. An independent NumPy implementation checks the profile value and analytic derivative through sorting, normalization, residualization and the log determinant.

| CPU precision | Median profile value | Median value and projection-angle gradient | Absolute derivative discrepancy against NumPy float64 |
| --- | ---: | ---: | ---: |
| float32 | 1.68 ms | 3.15 ms | 6.58e-7 |
| float64 | 1.95 ms | 6.55 ms | 1.78e-15 |

Timings are from one CPU process per precision on an AMD EPYC 9V74, Python 3.13.5 and JAX 0.11.0. Each median summarizes 30 synchronized executions after five warmups, excluding host transfers, tracing and compilation. This is a single-machine measurement for rank one, including the Gaussian profile. Encoder tuning, high-dimensional OT, complete estimation and bootstrap refits incur additional costs.

## Catalogue

The research catalogue links the scientific procedure `spatial.projected-graph-regression` to one study with distinct prespecified and training-selected methods. The continuous graph estimator, the chronological bandwidth selector, the time-block bootstrap and likelihoods with multiple endogenous operators or missing outcomes each retain a separate uncovered entry. The performance catalogue links both `graph-profile` variants to the Gaussian likelihood and RBF construction. The release qualification process separately checks retained campaign provenance and source review.

## Reproduction

With the repository's reference dependency group installed, run from the repository root:

```bash
(cd src/python/packages/statadic && JAX_ENABLE_X64=1 just test-external tests/estimation/spatial/test_graph_likelihood_reference.py)
JAX_ENABLE_X64=1 just python::studies::run build/studies/graph-selection-evaluation \
  --study spatial.graph-selection-power --profile evaluation --trials 10000 --seed 20261107
just python::studies::verify build/studies/graph-selection-evaluation
```

To reproduce each focused benchmark worker in the performance application environment:

```bash
for dtype in float32 float64; do
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
    python -m statadic_performance.worker graph-profile --dtype "$dtype" \
      --output "/tmp/statadic-graph-profile-$dtype.json"
done
```
