---
type: reference
description: Definitions and measurement scope of the CPU performance workloads.
---

# Performance workload definitions

## Usage

Inspect a matched operation and its independent reference from a repository checkout:

```python
from statadic_performance.reference_workloads import CASES

case = CASES["ols-hc3"]
print(case["operation"])
print(case["reference"])
```

## Description

Each workload specifies the calculation being measured, including its inputs,
returned outputs and scope of execution. Compare timings only after matching
these choices to the analysis you intend to perform, since a faster result can
reflect a smaller set of requested outputs.

## Options

(performance-discrepancy)=

### Distribution discrepancies

For clouds with unequal support sizes, the workload constructs a `PackedClouds`
carrier from fixed-shape coordinates within the compiled calculation. Its timing
therefore includes construction of the carrier, along with the calculation on
the clouds it represents.

(performance-discrepancy-mmd-value-gradient)=

#### RBF MMD value and input gradient

`discrepancy/mmd-value-gradient` evaluates V-statistic MMD squared and its
coordinate gradient, with RBF bandwidth one. Its derivative concerns the input coordinates rather than a fitted model, and
the signed U-statistic lies outside this measurement.

(performance-discrepancy-energy-euclidean)=

#### Raw Euclidean energy

`discrepancy/energy-euclidean` evaluates raw V-energy with explicit Euclidean
ground. It does not measure the rooted host `Energy` workflow.

(performance-ols)=

### Least squares

Each variant fits the same complete-data least-squares problem, while retaining
a different selection of results from the numerical calculation.

(performance-ols-full-fit)=

#### Full fit

`ols/full-fit` retains the complete native fitted state and diagnostics.

(performance-ols-parameters-projection)=

#### Parameters only

`ols/parameters-projection` returns only coefficients, allowing the compiler to
eliminate calculations for outputs that the caller does not use. Its cost can
therefore differ from a full fit even though the same least-squares solver is used.

(performance-ols-fit-and-inference)=

#### Fit and classical covariance

`ols/fit-and-inference` returns coefficients and their classical covariance.
Other covariance estimators and reference laws are outside this workload.

(performance-row-refits)=

### Supplied-row refits

Every variant refits the outcome against selected rows from the same design,
following one frozen resampling schedule. Varying the size of each chunk changes
the grouping of the calculations without changing the identities of the draws.

(performance-row-refits-chunk-1)=

#### One draw per chunk

`row-refits/chunk-1` is present in smoke and saved benchmark runs.

(performance-row-refits-chunk-2)=

#### Two draws per chunk

`row-refits/chunk-2` is a smoke-only variant.

(performance-row-refits-chunk-8)=

#### Eight draws per chunk

`row-refits/chunk-8` is present in saved benchmark runs.

(performance-row-refits-chunk-32)=

#### Thirty-two draws per chunk

`row-refits/chunk-32` is present in saved benchmark runs.

## Results

Timing the first call includes compilation for that particular shape and
precision, while synchronized warm execution measures subsequent calls. Host
transfer measures movement between host and device memory. Process peak RSS
includes memory for imports and compilation, whereas estimates of compiler
buffers concern the allocation for the compiled calculation.

Independent numerical comparisons run outside timing. Compare measurements
with matching hardware, shapes, precision and statistical configurations, using
the commands in the repository performance application.

A complete fit and a projection of its coefficients have different outputs.
A compiler may remove work whose results are unused, so compare consumers that
return the same quantities before interpreting a difference in execution time.

## Methods and formulas

(performance-graph-profile)=

### Projected Wasserstein graph likelihood

The full workload contains observations from 100 firms, with 128 equal-mass
articles for each firm over four outcome periods. Each article is represented
by a two-dimensional vector, which is projected onto a line before comparison.
Both variants sort the projected samples, form squared Wasserstein distances,
apply off-diagonal RBF weights and profile the Gaussian SAR mean and variance
at a fixed spatial coefficient.

(performance-graph-profile-profile-value)=

#### Profile log likelihood

`graph-profile/profile-value` returns the Gaussian profile value.

(performance-graph-profile-profile-value-angle-gradient)=

#### Profile value and projection derivative

`graph-profile/profile-value-angle-gradient` differentiates the full calculation
with respect to the projection angle. An independent NumPy calculation checks
the value and derivative away from sorting ties.

## Assumptions and limits

Measurements describe a particular implementation, device, precision and shape.
Numerical agreement, statistical behavior and resource use require their own
evidence. A bounded example supports conclusions about that example, while the
full joint censored SAR/AFT workload still lacks a matched independent reference.

## Evidence

<!-- statadic:performance-catalogue -->

The [performance application](https://github.com/marcusinthesky/statadic/blob/main/src/python/apps/performance/README.md)
records the exact matched consumers, independent references and measurement
policies. Read those definitions alongside the results of the particular run.

## See also

- [Transformation support](transformation-support.md)
- [Independent references](../validation/references.md)
- [Known limitations](../validation/known-limitations.md)
