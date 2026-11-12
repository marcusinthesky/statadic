---
type: reference
description: Find dependent-data resampling and track-record inference operations.
---

# Resampling and performance

## Usage

```python
import jax.numpy as jnp
from statadic.sample.packed import pack

clouds = pack([jnp.array([[0.0], [1.0]]), jnp.array([[2.0], [3.0]])])
```

## Description

Resampling constructs repeated datasets or schedules to describe uncertainty. Performance procedures assess return statistics and account for choices among strategies.

The choice of schedule determines which relationships among observations are preserved when the procedure creates repeated samples.

For a dependent time series, the length of a block controls how much of its local dependence can be represented in a resampled sequence.

## Options

| you want | call |
|---|---|
| a Sharpe ratio | `statadic.estimation.sharpe.sharpe_ratio` |
| probabilistic / deflated Sharpe | `statadic.inference.sharpe.probabilistic_sharpe_ratio`, `deflated_sharpe_ratio` |
| a paired Sharpe-difference test | `statadic.inference.sharpe.sharpe_difference_test` |
| Romano–Wolf / Reality Check across strategies | `statadic.inference.sharpe.multiple_sharpe_test` |

| You want | Call | Notes |
| --- | --- | --- |
| A block length for a dependent series | `statadic.inference.resampling.block_length.optimal_block_length` | The heuristic in the same module is a different estimator |
| A stationary bootstrap interval | `statadic.inference.resampling.block_length.stationary_bootstrap_ci` | Select assumptions and block length for the data |
| A transformable bootstrap | `statadic.inference.resampling.bootstrap.bootstrap_estimate` | Uses a PRNG key and returns an `EstimateResult` |
| A scalar bootstrap interval | `statadic.inference.resampling.bootstrap.bootstrap_confidence_interval` | Eager adapter returning two floats |
| Scheduled row draws and refits | `statadic.inference.resampling` | Preserves draw identities and failure status |
| Circular block indices | `statadic.operators.longrun.circular_block_bootstrap_indices` | Builds indices for a dependent sequence |

## Results

Bootstrap estimates retain their draw information and status. Performance tests report their selected reference and uncertainty.

## Assumptions and limits

The sampling design must match the dependence in the data. A supported executor does not guarantee coverage, and failed draws must remain visible.

## Evidence

[Independent comparisons and studies](../validation/index.md) describe the scope of evidence for each method.

## See also

- [Resampling](resampling.md)
- [Known limitations](../validation/known-limitations.md)
