# U or V: where does bias come from?

**Learning objective:** compare the finite-sample bias of two squared-MMD
estimators, and distinguish Monte Carlo evidence from a mathematical theorem.

Predict whether including within-sample self-pairs changes the expected result.
Then run the registered `mmd.finite-bias` study: it compares U and V reductions
for the same data in four fixed scenarios. The linear-kernel population target is
known for each scenario. The table reports estimated bias and Monte Carlo standard
error conditional on valid estimates, alongside valid and failed counts. More trials
reduce simulation uncertainty; they do not change the population target.

The lab creates a temporary source-only smoke campaign through the same study engine
used for research. Its JSON download contains the frozen manifest, provenance and
summary. Individual trial records are not retained in this teaching download, and
these runs cannot qualify a method. For retained evaluation evidence, see
[research results](../validation/research.md).

Compare the [checked MMD mathematical result](../formal/claims/STATADIC-MMD-001.md)
with the [MMD API](../reference/api.rst): the Lean result concerns its stated
mathematical assumptions, while the experiment calls the Python implementation.
Neither alone proves the floating-point implementation correct.

```{raw} html
<a href="snapshots/study-results.html">View the executed static snapshot</a> · <a href="downloads/study-results.py">Download the notebook</a> · <a href="snapshots/study-results.provenance.json">Snapshot provenance</a>
```

[Run this lab locally](index.md)
