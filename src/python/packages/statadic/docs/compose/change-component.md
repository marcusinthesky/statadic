# Change one component: covariance

Run this independently after [installation](../start/installation.md):

```bash
.venv/bin/python src/python/packages/statadic/examples/change_covariance.py
```

```{literalinclude} ../../examples/change_covariance.py
:language: python
```

Both retain the same point fit and normal reference. HC3 adjusts covariance for
leverage. Inspect availability before interpreting either report. Neither establishes
causal identification or exact small-sample coverage.

<!-- statadic:covariance-comparison -->

See [inference choices](../guides/inference.md).
