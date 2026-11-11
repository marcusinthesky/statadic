# Research-workflow evidence

Measured on 2026-11-05 at commit `36168ea` in `devenv shell`: Python 3.13.15, JAX
0.11.0 on CPU with x64 off, Polars 2.1.0, PyArrow 24.0.0, pandas 3.0.3, Tectonic
0.17.0 and the `tlextras-2022.0r0` bundle. Probes ran outside the repository and were
not retained. Rerun them from the descriptions and fixture below. HoloViews and Orbax
were resolved or installed in a throwaway environment and never added to the lock.

## Poisson fixture

```python
import jax.numpy as jnp
import numpy as np
from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear
from statadic.api.inference import HC
from statadic.api.regression import PoissonRegression

rng = np.random.default_rng(42)
load, hours = rng.uniform(0, 5, 200), rng.uniform(1, 4, 200)
incidents = rng.poisson(hours * np.exp(-0.5 + 0.3 * load))
fit = PoissonRegression(
    response="incidents", terms=Intercept() + Linear("load"), exposure="hours"
).fit(Observations.from_arrays({"load": load, "hours": hours, "incidents": incidents}))
report = fit.infer(covariance=HC("HC1")).report(reference="normal")
```

Before `reject-narrowing` this runs in float32 without a warning. Afterwards it needs
x64 enabled or explicit casts, which is the intended change.

## Prediction inputs

- `kind="rate"` without an `hours` column raises `KeyError: 'hours'` on live and reloaded
  predictors, though `poisson_prediction` ignores exposure for that kind.
- `{"load": ...}` and a Polars frame passed to `predict` raise `AttributeError`, because
  they have no `numeric` method.
- A scalar column in `from_arrays` raises `Numeric column 'hours' must have shape (3,)`.
- Reloaded Poisson and linear predictors accept `kind="linear"`. The live Poisson
  predictor rejects it, and live `LinearResult.predict` takes no `kind`.

## Precision with x64 off

| Input | Result |
| --- | --- |
| NumPy float64 through `Observations.from_arrays` | Stored as float32 with no warning; `1.0` and `1.0 + 1e-12` became equal |
| Default Polars `Float64` through `from_polars` | `ValueError`: requires caller-enabled JAX x64 or an explicit source cast |
| Default Polars `Int64` through `from_polars` | The same `ValueError` |
| Polars cast to `Float32` through `from_polars` | Accepted as float32 |
| Wide arrays passed to artifact saving or loading | `ArtifactError` |

## Arrow boundary

- `jnp.from_dlpack(pa.array(float32_values))` shares the Arrow buffer: the pointers matched.
- `__arrow_c_stream__` is present on Polars frames, PyArrow tables and pandas 3.0.3 frames.
- `pa.table(polars_frame)` works through that protocol.
- `api/tables.py:14` imports Polars at module level for annotations.

## Artifact codec

The fixture's `arrays.bin` is 8 bytes (two float32 coefficients). `_encode` output was
byte-identical to `array.tobytes()`.

| Elements | `_encode` | `tobytes` | Bundle write | Bundle read |
| --- | --- | --- | --- | --- |
| 10,000 | 0.031 s | under 0.001 s | 0.001 s | 0.060 s |
| 1,000,000 | 0.053 s | 0.001 s | 0.024 s | 0.335 s |
| 5,000,000 | 0.141 s | 0.003 s | 0.146 s | 1.540 s |

The Phase 4 record also states that no new checkpoint codecs were added.

## Dependency closures

- HoloViews 1.23.2 with Matplotlib: 38 packages and 318 MB installed, including pandas,
  Bokeh and Panel. It resolves against the locked set.
- Orbax 0.12.9: 58 packages, including Google Cloud storage clients and TensorStore.
  Unconstrained it selects JAX 0.12.0.1, while `pyproject.toml` pins `<0.12`. It resolves
  to JAX 0.11.0 when constrained by the lock.

## LaTeX report

Compiling the fixture's `report.to_latex()` in a plain `article` with `booktabs`:

- The Statistic and $p$ columns print the word `unavailable` in every row. This is by
  design in the data, because `InferenceResult.report` omits default p-values.
- Metadata prints as a JSON paragraph below the table.
- Overfull boxes: 43pt for the table and 72.9pt for the JSON paragraph.
- The metadata has no confidence level, so a table cannot state its interval level.

The docs gate passes the same content only because `apps/docs/templates/report.tex` uses
a landscape page and `\footnotesize`.

## Plot output

| Route | Result |
| --- | --- |
| `PlotSpec.to_matplotlib().savefig("x.pgf")` | `RuntimeError: 'xelatex' not found`; leaves a truncated file |
| HoloViews `hv.render` then `savefig("x.pgf")` | The same error and a truncated file |
| HoloViews `hv.save(curve, "x.pgf")` | The same error |
| `PlotSpec.to_pgfplots()`, repo cache only | `pgfplots.sty` not found |
| `PlotSpec.to_pgfplots()`, bundle primed | Compiles with pgfplots 1.18.1 and stays editable as `\addplot` |
| Polygon ribbon using `coordinates {...} -- cycle;` | Compiles and renders with no extra PGFplots library |
| Trusted `$\hat\beta_1$` axis label | Renders when not escaped |

- `render_pgfplots` currently escapes `$\hat\beta_1$` so it prints literally. Matplotlib
  parses the same string as math, so the two renderers disagree.
- A 12 cm figure with a legend overflowed a portrait `article` by 9.7pt.
- `devenv shell` provides `tectonic` but no `xelatex`, `pdflatex` or `lualatex`.
- No test compiles `to_pgfplots()` output; existing tests assert on strings.
- The bundle host returned HTTP 429 and then a transient failure before succeeding.

## Linear MMD

- `linear_gram` is uncentred: `jnp.dot(x, y.T)` (`geometry/ground/similarities.py:124`).
- The mean route subtracts a reference (`discrepancy/mmd/scalar.py`,
  `discrepancy/mmd/weighted.py::_linear_mmd_cross`).
- Algebraically the routes differ by the reference times the difference in total mass, so
  they agree when both measures have equal total mass. No Lean theorem covers this yet.

## Formal status

- `src/lean/registry/claims.json` has 7 claims. Every review field records that human
  review is pending.
- `src/lean/registry/methods.json` has 53 obligations: 50 `source-unresolved` and 3 `open`.
- `implementation-pins.json` has status `source-bound-research-prototype` and five
  unresolved gates.

## Publication facts

- The repository was private, created 2026-11-01, with 14 commits and no tags.
- Source was about 55,800 lines and tests about 58,900 lines of Python.
- [JOSS](https://joss.readthedocs.io/en/latest/submitting.html) requires more than six
  months of public history, evidence of research use and an AI usage disclosure.
- [JSS](https://www.jstatsoft.org/about/submissions) requires replication materials and a
  GPL or GPL-compatible licence. Apache-2.0 is compatible with GPL-3; confirm with editors.
- SoftwareX requires freely available code. Its fees and length limits differ across
  third-party listings, so check its own guide.
- A search found no statistics or economics journal requiring Lean proofs.
- [MathEconLib](https://github.com/matheconlib/matheconlib) lists econometrics only as a
  long-term goal.

## Not verified

SJR scores, Lean `vcgen` benchmark figures and the Anthropic/Theorem proof claims were not
checked. GPU behaviour was not measured.
