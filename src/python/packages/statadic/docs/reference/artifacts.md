---
type: reference
description: Save and reload fitted predictors and analysis snapshots, and read what each file format restores.
---

# Saved artifacts

## Usage

```python
from pathlib import Path

from statadic.api.artifacts import load_analysis, load_predictor, save_analysis
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Regression

data = {"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "y": [1.1, 1.8, 3.2, 3.7, 5.3, 5.8]}
fit = Regression.ols("y", Intercept() + Linear("x")).estimate(data).require_valid()

output = Path("results")
output.mkdir(exist_ok=True)
fit.save_predictor(output / "predictor")
save_analysis(output / "analysis", fit.report())

predictor = load_predictor(output / "predictor")
print(predictor.predict({"x": [6.0, 7.0]}))
print(load_analysis(output / "analysis").report.to_text())
```

## Description

STATADIC saves two kinds of artifact. A predictor keeps what a fitted model
needs to predict or transform new rows, and an analysis snapshot keeps a
computed [report](reporting.md) for rendering. Each artifact is a directory
with two files. The manifest records the schema version, the purpose, the family
and the declared capabilities of the artifact, and the binary file stores
little-endian arrays with a digest for each of them.

Loading restores only what an artifact declares. A loaded predictor applies its
frozen design to new rows, and a loaded snapshot renders the report in every format. Neither
contains a covariance or a way to refit, so rerun your analysis script on the
original data when you need new inference.

## Options

| Function | Saves | Restores |
| --- | --- | --- |
| `fit.save_predictor(path)` and `load_predictor(path)` | A linear, Poisson or logistic regression predictor | A {py:class}`~statadic.api.artifacts.PredictionArtifact` with `predict(data, kind=None)` |
| `save_ordination_predictor(path, mapping=..., metadata=...)` and `load_ordination_predictor(path)` | A stored ordination coordinate map | An `OrdinationArtifact` with `transform` and `transform_distances` |
| `save_graph_predictor(path, result)` and `load_graph_predictor(path)` | A graph regression result | A `FrozenGraphPredictor` whose `predict(observations)` solves the spatial mean |
| `save_graphs(path, graphs, parameters=...)` and `load_graphs(path)` | Distance-graph recipes with shared fitted parameters | A `GraphArchive` |
| `save_analysis(path, report)` and `load_analysis(path, trust_tex=False)` | Parameters, metrics, tests, metadata, warnings, notes and plots | An `AnalysisSnapshot` whose `report` renders every format |

Fitted results with a `save_predictor` method accept a `path` only. The
module-level `save_predictor(path, design=..., parameters=..., family=...)`
takes the pieces directly, with optional `offset`, `exposure`, `trials` and
`metadata`.

## Results

A loaded `PredictionArtifact` exposes `capabilities`, which is `("predict",)`,
and `parameter_names`, the ordered feature schema. Prediction needs the
predictor columns in your new rows, plus any offset, exposure or trials column
required by the requested prediction kind. For example, a Poisson rate needs no
exposure and a binomial probability needs no trials. Nested ordination terms keep their training maps inside the
regression design, so a loaded regression predictor transforms new rows the way
the fit did.

A snapshot with TeX markup in a plot needs `trust_tex=True` at the call to
`load_analysis`, because the markup passes to LaTeX unchanged. Without it, `load_analysis`
raises `ArtifactError`.

## Methods and formulas

Current artifacts use schema 7. The manifest holds closed JSON records, and
loaders validate the required fields, discriminants and values of each record
before rebuilding any numerical state. The prediction kind follows from the declared family rather
than from the saved state. Kernel bandwidths keep their scalar payload, dtype
and JAX weak-scalar status, so loading preserves how they promote against
feature and measure arrays. A reader accepts only its own schema version, and
the error message names the version found and the version expected.

## Assumptions and limits

- A reader rejects any other schema version, including artifacts from earlier
  prerelease versions. There is no migration, so re-save from the original
  analysis. The manifest records the writer version for diagnosis only.
- Executable classes, closures and pickle are unsupported. Restored state that
  fails validation raises `ArtifactError` with the underlying cause.
- Saving and loading arrays with 64-bit precision require JAX x64 to be enabled.
  Artifacts are not silently downcast when the setting is disabled.
- Predictors restore neither inference nor refitting, and snapshots restore
  neither a fit nor sampling uncertainty.
- A snapshot preserves reproducibility of the table, not of the computation.
  Keep the analysis script, lockfile and data identity, or use the
  `statadic-publishing` companion, to rebuild a result.

## Evidence

Round trips for every artifact family are tested for exact prediction and
rendering, and malformed or tampered files are tested to raise `ArtifactError`.
See the [composition reference](composition.md) for which models save a predictor.

## See also

- [Reports](reporting.md) and [save and share results](../guides/save-and-share-results.py)
- [Regression](regression.md) and [composition](composition.md)
