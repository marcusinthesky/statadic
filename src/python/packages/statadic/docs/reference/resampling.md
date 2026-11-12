---
type: reference
description: Generate paired row-index schedules, evaluate statistics and refits over them, and summarize the retained draws.
---

# Resampling

## Usage

```python
import jax
import jax.numpy as jnp
from statadic.inference.resampling.schedule import (
    ReplicateValue,
    evaluate_replicates,
    iid_row_schedule,
    percentile_summary,
)

data = jnp.array([1.0, 3.0, -2.0, 8.0])
schedule = iid_row_schedule(jax.random.key(42), data.shape[0], 199)


def statistic(indices):
    sample = data[indices]
    return ReplicateValue(
        values=jnp.array([jnp.mean(sample), jnp.mean(sample**2)]),
        valid=jnp.ones((), dtype=bool),
        status=jnp.array(0, dtype=jnp.int32),
    )


draws = evaluate_replicates(schedule, statistic, chunk=16)
summary = percentile_summary(draws, confidence_level=0.95)
print(draws.valid_count, summary.lower, summary.upper)
```

## Description

`statadic.inference.resampling` separates four objects. A row-index schedule says
which rows enter each draw, an evaluator or refitter computes the requested
values, a result retains every draw with its status, and a summary or test
applies its own failure policy. Because one schedule can feed several
statistics, the draws stay paired by ID across all of them, and the same
indices reach every statistic in the same order.

Schedules hold row indices only. Node multiplicities, sampled cluster IDs and
wild-bootstrap multipliers are different designs, and the dyadic and Mantel
procedures keep their own. A schedule does not choose a dependence model, so you
decide whether independent rows, stationary blocks or a permutation group
matches the way the data arose.

## Options

| Function | Purpose |
| --- | --- |
| `iid_row_schedule(key, n, draws)` | Rows sampled with replacement |
| `stationary_row_schedule(key, n, draws, block_length)` | Circular blocks with geometric lengths |
| `row_resample_schedule(indices, n_observations=..., draw_ids=...)` | Your own integer `(draws, observations)` array |
| `evaluate_replicates(schedule, fn, chunk=...)` | Evaluates a statistic over every draw |
| `refit_replicates(schedule, refit_fn)` | Refits one frozen model on each draw |
| `linear_pairs_replicates(fit, schedule)` | Row-pairs refits of a least-squares fit |

A statistic returns a `ReplicateValue` whose values are real floating scalars or
vectors, or nonempty tuples and string-key dictionaries of them. Shapes and
structure stay fixed across the draws. Keys can be typed JAX keys or `uint32[2]`
arrays, and `chunk` bounds the working memory of an evaluation in the loop over
draws.

Each entry point has its own failure policy, and this table is the single
statement of them.

| Entry point | Policy |
| --- | --- |
| `percentile_summary` | Needs two or more draws and no failed draw, otherwise bounds and SE are NaN with a status |
| `add_one_p` | Discards nothing, and a non-finite value or an empty reference gives NaN |
| `permutation_test_kernel` | Counts ties as extreme, with p-value $(1+b)/(1+B)$ |
| `bootstrap_estimate`, `stationary_bootstrap_estimate` | Scalar results from scheduled evaluation, with SE computed at $\text{ddof}=1$ and set to zero for a single draw |
| Eager Mantel bootstrap adapters | Discard degenerate draws, and the paired gap needs both arms valid |

## Results

`ReplicateResult` keeps every requested draw in schedule order, with `draw_ids`,
`valid`, `status` and the counts `requested_count`, `valid_count` and
`failed_count`. The raw values of failed draws stay available for inspection, and
a draw is invalid when its value is not finite, even if the callback reports
success. `PercentileSummary` carries `lower`, `upper`, `standard_error`, `valid`
and a `status` taken from `ReplicateStatus`.

Refitting a least-squares model over a stationary schedule shows how the pieces
of the workflow fit together:

```python
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.inference.resampling.schedule import (
    linear_pairs_replicates,
    stationary_row_schedule,
)

t = jnp.arange(30.0)
x = jnp.column_stack([jnp.ones(30), t])
y = 0.5 * t + jnp.sin(t)
fit = LeastSquares().estimate(LinearProblem(x, y, parameter_names=("intercept", "slope")))
blocks = stationary_row_schedule(jax.random.key(0), 30, 99, 4.0)
slopes = linear_pairs_replicates(fit, blocks)
print(slopes.valid_count, percentile_summary(slopes).valid)
```

`linear_pairs_replicates` gathers the design, response and relative precisions
with one set of indices and keeps the order of the columns. Any other model
works through `refit_replicates`, whose closure receives the scheduled indices
and returns an object with `.parameters`, `.valid` and `.solver`. The closure
states which of the steps of preprocessing, selection and tuning are frozen or
repeated, and a draw counts as valid only when the fit is valid and the solver
converged.

## Methods and formulas

Percentiles use linear interpolation for each component of the output, and SE is
the sample standard deviation with $\text{ddof}=1$. Bounds for a vector are
marginal intervals, so they carry no simultaneous coverage.

<!-- statadic:math STATADIC-RANDOMIZATION-001 -->

`permutation_test_kernel(statistic, null, B, alternative)` returns
$p = (1 + b)/(1 + B)$, where $b$ counts null draws at least as extreme as the
observation. The rule is `null >= observed` for `greater`, `null <= observed`
for `less` and `abs(null) >= abs(observed)` for `two-sided`. The last differs
from doubling the smaller tail when the null is asymmetric.
`PermutationPlan(unit=..., blocks=...)` names the group that is permuted. Whole
assignment units move together and labels stay inside blocks.

## Assumptions and limits

- A schedule does not establish exchangeability or weak dependence. A permutation
  group larger than the one in the design, such as free rows for repeated visits
  of randomized subjects, can overstate the significance of a result.
- The add-one p-value is a randomized quantity. An exhaustively enumerated null
  does not switch it to the exact $b/B$ rule.
- Indices out of range invalidate their own draw, and duplicate draw IDs
  invalidate the paired schedule. Host integers must fit within the active JAX
  integer width.
- Block lengths below one are treated as one, and nonfinite lengths invalidate
  every draw.
- Equal seeds under different PRNG implementations need not give equal streams.
- Stationary-bootstrap mean intervals have no general finite-sample coverage
  guarantee.
- Schedules and results store $O(\text{draws} \times \text{observations})$
  values, whatever the chunk size.
- Functions transform under `jit` and outer `vmap` with fixed counts, chunk size
  and confidence level. No derivative of a fitted solution follows from this.

## Evidence

- Mathematical results: the {ref}`finite full-group level result <formal-STATADIC-RANDOMIZATION-001>`
  for randomization tests over a finite group.
- Studies: {ref}`bootstrap of the mean <research-procedure-bootstrap.mean>` and
  {ref}`mixture row refits <research-procedure-bootstrap.mixture-row-refits>` in
  the study catalogue.
- Finite-sample coverage of stationary-bootstrap intervals is recorded in the
  [known limitations](../validation/known-limitations.md).

## See also

- [Resampling guide](../concepts/resampling-and-failures.md) and [hypotheses](hypotheses.md)
- [Graph resampling guide](../guides/resample-panel-graphs.py)
- [Method catalogue](method-catalogue.md)
