# Resampling: schedules, failures and denominators

Choose the resampling design for the statistical problem before choosing a computational executor. A shared schedule gives paired draw identities and repeatable computation; it does not establish exchangeability or a valid dependence model.

## Separate four objects

A row-index schedule specifies which rows enter each draw. An evaluator or refitter computes the requested values. A retained result records every requested draw and its status. A summary or test then applies its own failure and calibration policy.

Row schedules are not node multiplicities, sampled cluster identities or wild-bootstrap multipliers. A dyadic analysis and an ordinary row-pairs bootstrap do not become interchangeable because both use repeated indices somewhere internally.

## Failure policy is entry-point specific

| Entry point | Documented policy | Reporting implication |
|---|---|---|
| Preview `percentile_summary` | At least two draws; any failed draw makes the summary unavailable; bounds and SE become NaN with status | Report requested, valid and failed counts; do not delete failures to obtain an interval |
| Existing `add_one_p` | Discards no draws; a non-finite observation or null draw, or no draws, gives NaN | Report NaN as unavailable, never as a p-value from the remaining draws |
| Eager Mantel bootstrap adapters | Historical conditional-on-valid-draw summaries; paired gap requires both arms valid | Explain conditioning and retained draw counts |

A blanket statement that STATADIC never drops failed draws would be wrong. The strict scheduled API offers a stronger policy, but legacy adapters retain distinct contracts.

## Refits and preprocessing

State which operations are frozen and which are repeated. The linear-pairs adapter gathers the retained design, response and relative precisions together; it does not redo selection, imputation or tuning. A rank-deficient resample remains failed even if a solver returned an iterate. Shape/programming errors are not silently converted to statistical failure.

## Calibration and memory

The permutation kernel's randomized add-one calculation uses `(1+b)/(1+B)`. Passing an exhaustively enumerated null does not automatically switch it to the exact `b/B` rule. Its two-sided extremeness rule uses absolute statistics, not necessarily twice a one-sided tail.

Vector percentile intervals are marginal, not simultaneous. Chunking reduces concurrent working space, but schedules and retained results still grow with draw count. A random seed does not promise identical streams across PRNG implementations.

Consult [known finite-sample limits](../validation/known-limitations.md) before treating a stationary-bootstrap mean interval as calibrated.
