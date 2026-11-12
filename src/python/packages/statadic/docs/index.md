# STATADIC

## Composable statistics. Inspectable mathematics. Reproducible evidence

Fit a familiar model on JAX. Change supported components. Inspect the assumptions,
numerical diagnostics and scientific evidence behind your result.

**Align → Specify → Estimate → Inspect → Diagnose → Predict or infer → Report.**

[Get started](start/index.md) · [Explore methods](methods.md)

```python
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Regression

data = {"x": [0.0, 1.0, 2.0, 3.0], "y": [1.0, 2.1, 2.8, 4.2]}
fit = Regression.ols("y", Intercept() + Linear("x")).estimate(data)
print(fit.report())
```

Continue with [diagnostics, inference and reporting](tutorials/first-regression.md),
or [distribution-valued observations](tutorials/distribution-workflow.md).

API reference: [Regression](reference/regression.md) ·
[Ordination](reference/ordination.md) · [Estimand](reference/estimand.md) ·
[Hypothesis](reference/hypotheses.md) ·
[Report](reference/reporting.md).

[Transformations](capabilities.md), [machine-checked mathematics](formal/index.md)
and [statistical studies](validation/index.md) answer different questions.
Read the [known limits](validation/known-limitations.md) before interpreting results.

:::{toctree}
:hidden:
:maxdepth: 2

Start <start/index>
Tutorials <tutorials/index>
How-to <guides/index>
Concepts <concepts/index>
Reference <reference/index>
Evidence <validation/index>
Blog <blog/index>
:::
