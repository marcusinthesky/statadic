# Using a proof to justify a method

Start with the conclusion you want to report. A claim page gives the mathematical
assumptions, the exact conclusion and the observations that help assess whether
that result applies. Record which conditions the data or solver can check and
which depend on the study design. Cite the claim ID alongside the paper, the
STATADIC revision, the diagnostic values and the threshold used in the analysis.

## When a diagnostic rules out a method

The shortest-path distances around a four-cycle satisfy the metric triangle
inequality. Classical scaling nevertheless cannot reproduce all six pairwise
distances as ordinary Euclidean distances. Its centered Gram matrix has an
eigenvalue of −1. For example, run this with JAX 64-bit mode enabled:

```python
import jax.numpy as jnp
from statadic.geometry.embedding import pcoa_kernel

distances = jnp.array([
    [0., 1., 2., 1.],
    [1., 0., 1., 2.],
    [2., 1., 0., 1.],
    [1., 2., 1., 0.],
])
fit = pcoa_kernel(distances, tolerance=1e-8)
print(bool(fit.valid), float(fit.minimum_eigenvalue), float(fit.spectral_scale))
```

The result is `False`, approximately −1, and approximately 2. The permitted
negative roundoff is only $2\times 10^{-8}$. The failed check rules out the
exact Euclidean interpretation for this input. For a passing matrix, the
squared-distance truncation bound uses the same spectral cutoff. Removing
additional positive axes for a two-dimensional plot requires accounting for
the discarded spectrum. [Euclidean realization](claims/STATADIC-EMBEDDING-002.md) ·
[Truncation error](claims/STATADIC-EMBEDDING-003.md)

## When a diagnostic quantifies what remains uncertain

For a capped-simplex quadratic, the threshold $K u\ge 1$ tells you whether any
$K$ weights can sum to one under cap $u$. Once feasibility and convexity on the
zero-sum directions are established, the linearization gap bounds the objective
improvement available over a candidate. The size of the change in weights on
the last iteration has a different interpretation.

```python
from statadic.optimize.simplex import simplex_qp_result

fit = simplex_qp_result(2 * jnp.eye(2), jnp.array([0., 1.]), maximum_weight=0.6)
print(fit.solver.value, fit.feasible, fit.convex, fit.optimality_gap)
```

The exact optimum is $(0.6,0.4)$. A 64-bit run returned approximately
$(0.59999987,0.40000013)$, both flags true, and an objective gap of about
$7.6\times10^{-8}$. Report the cap, precision, gap and feasibility residual,
and retain the dual and complementarity residuals when using the KKT argument.
The theorem bounds objective loss; bounding weight error also requires
curvature away from zero. [Feasible cap](claims/STATADIC-SIMPLEX-003.md) ·
[KKT certificate](claims/STATADIC-SIMPLEX-001.md) · [Objective gap](claims/STATADIC-SIMPLEX-002.md)

## When the assumption comes from the research design

| Method | Evidence available | What the researcher must justify |
| --- | --- | --- |
| Two-way clustered covariance | Cluster labels, scores and the shared-entity/shared-time inclusion–exclusion calculation | Dependence outside both labels must satisfy the stated cross-moment restriction. An omitted common shock can violate it even with correctly encoded labels. [Population criterion](claims/STATADIC-PANELCOV-002.md) |
| Delayed-entry survival | Entry and exit ordering, event indicator, time scale, fitted density and survival contributions | State how entry and censoring occurred and why the conditional likelihood applies to the observed risk set. [Conditional likelihood](claims/STATADIC-SURVIVAL-001.md) · [Time transformation](claims/STATADIC-SURVIVAL-002.md) |
| Gaussian probability integration | Effective counts within replicates and relative standard error across replicates | Between-replicate accuracy requires a randomized design with the stated target expectation, variance and cross-replicate covariance. Reusing identical deterministic point sets can produce zero dispersion. [Effective count](claims/STATADIC-INTEGRATION-001.md) · [Accuracy bound](claims/STATADIC-INTEGRATION-002.md) |
| Rooted energy distance | Ground and moment descriptors, energy scheme, numeric root | Supply the required moments and a separating mean embedding on the intended class of laws. Document the ground and sampling scheme before reading zero distance as equality of distributions. [Metric conditions](claims/STATADIC-ROOTENERGY-001.md) |
| Simulation qualification | Attempted, failed, rejected and covered trial counts | Retain failed trials. The extremal completion bounds are deterministic; interpretation of binomial uncertainty also depends on how the attempted trials were generated. [Failure-inclusive bounds](claims/STATADIC-QUALIFICATION-001.md) |

## Choosing the claim for a reported result

**Affine GMM and IV.** Suppose an overidentified fit returns a small moment
residual and an instrument matrix of full sample rank. Exogeneity requires the
population moment to vanish at the scientific target. A counterexample attains
zero residual at a different target. Another example keeps the matrix full rank
while making coefficient sensitivity arbitrarily large. Report the instrument
rationale, singular values on the stated scale, fitted weight and choice of
score covariance. A Hansen J reference with $m-p$ degrees of freedom also
requires the efficient weight and the stated Gaussian limit. Independent
square-integrable rows furnish one route to that limit.
[Identification](claims/STATADIC-GMM-004.md) · [Wrong target](claims/STATADIC-IV-002.md) ·
[Weak instruments](claims/STATADIC-IV-003.md) · [Score CLT](claims/STATADIC-GMM-011.md) ·
[Hansen J](claims/STATADIC-JTEST-003.md)

**Permutation inference.** With 19 independent uniform transformations and the
observed statistic added to the reference set, the smallest possible upper-tail
p-value is $1/20$. If all 19 statistics tie the observed statistic, the inclusive
count gives $20/20$. State the units that can be exchanged under the null,
the group action, how the transformations were sampled and how ties were
counted. Full enumeration attains each nominal grid point under the distinct-rank
assumption. A data-dependent choice of transformation or statistic needs its
own argument.
[Sampled level](claims/STATADIC-RANDOMIZATION-004.md) ·
[Ties](claims/STATADIC-RANDOMIZATION-005.md) · [Full-grid equality](claims/STATADIC-RANDOMIZATION-006.md)

**Long-run covariance.** For a series of length $N$ and a Bartlett bandwidth
$b$, report $b$, the time index, the centering convention and the construction
of each score. The growing-window consistency result requires summable
covariances, a bound on dependence among lag products and
$(b+1)^2/N\to0$. For fitted scores the replacement error must remain small
after summing over the included lags. Compute sensitivity to nearby bandwidths
and inspect persistence. The decay assumptions concern the underlying process
and must be justified from the sampling design or a process model.
[Centered Bartlett limit](claims/STATADIC-HAC-007.md) · [Score substitution](claims/STATADIC-COVBRIDGE-001.md)

**Learned spatial graphs.** Record which observations, outcomes and timestamps
determined the weights. If the analysis conditions on an externally fixed
graph, give the reason its moment or invariance assumptions still hold after
conditioning. An independent training split can justify averaging a valid
fixed-choice test over graph selection. Adjacent blocks from the same dependent
panel need a separate independence or dependence argument. On a shared graph,
zero-mean lag and error filters can also yield exactly the same covariance after
swapping the two coefficients.
[Held-out selection](claims/STATADIC-SELECTION-001.md) ·
[Shared-graph alias](claims/STATADIC-SPATIAL-006.md) · [Global moment roots](claims/STATADIC-SPATIAL-004.md)

**Geometric median and transport.** A median of a noncollinear Euclidean
sample is unique. With just two distinct points the whole joining segment
minimizes the distance sum. Report an optimality residual separately from the
geometry that warrants uniqueness. For finite transport, specify both
marginals, the cost matrix and its units. A feasible coupling and feasible
dual potentials bracket the optimum. At equal objective values they certify
that plan, including when several plans tie. When the implementation returns
a rooted cost, compare certificates on the powered objective first.
[Median uniqueness](claims/STATADIC-MEDIAN-004.md) · [Collinear example](claims/STATADIC-MEDIAN-005.md) ·
[Primal and dual certificate](claims/STATADIC-TRANSPORT-001.md) ·
[Finite rectangular attainment](claims/STATADIC-TRANSPORT-005.md)

For a manuscript, `\statadicassumptions{ID}` and `\statadicdiagnostics{ID}` reproduce the
reviewed text that accompanies a specific claim. Add the values observed in
your analysis and your reasons for accepting the study-design assumptions.
[Manuscript guide](manuscripts.md) · [All families and their reference calculations](family-sources.md)
