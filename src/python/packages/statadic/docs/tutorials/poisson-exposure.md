# Poisson regression with exposure

This example generates synthetic counts in an Arrow table and fits a Poisson
model with exposure. Run `.venv/bin/python poisson_exposure.py --output results` after
[installation](../start/installation.md); omitting the destination uses a temporary
directory.

```{literalinclude} ../../examples/poisson_exposure.py
:language: python
:start-after: "# tutorial-start"
:end-before: "# tutorial-end"
```

{download}`Complete executable script <../../examples/poisson_exposure.py>`

The fitted rate is per unit of exposure, so its prediction line needs only the
load grid. Mean count predictions also require exposure. HC1 with a normal
reference describes uncertainty under the chosen assumptions. The script writes
the report and figure for LaTeX and checks both saved artifacts after reloading.
