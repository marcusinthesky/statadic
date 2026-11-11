# Verification

Commands use the pinned `devenv shell` and locked uv projects.
Results below describe completed command runs. Concurrent edits landed after
the full scientific-suite run and prevent a clean current-workspace gate.

| Check | Evidence |
| --- | --- |
| Architecture | Native Tach internal/external checks, source guard and 13 rejection fixtures pass |
| Typing | Full strict check initially passed; final scoped check has zero errors, excluding the three concurrently edited prediction/panel files listed below; isolated shape check and seven fixtures pass |
| Derivatives and numerical ports | 94 regression/panel/inverse-normal/PSD tests pass, five optional skips; repeated-eigenvalue JVP/VJP and generalized-vertex witnesses pass |
| Cloud/artifact/spatial fixes | 173 focused tests pass, one optional skip; independent little-endian codec bytes, corruption metadata and squared-MMD curvature are included |
| Mixture distances | 149 consumer tests and 26 additional barycentre/parity/reference tests pass, three skips in each group; weighted-atom flattening witness and a JIT/vmap dense-grid oracle pass |
| Qualification | Native float32/float64 binomial bounds agree with SciPy; large-ID typed-key witnesses pass |
| Studies | 287 tests pass, three skips; updated public-symbol catalogue passes its consistency check |
| Performance consumer | 24 tests pass, including precision-isolated native workloads |
| Installed artifacts | Both wheel routes and standalone source-distribution development pass; 50 typing tests pass, seven shape tests delegated to their isolated environment |
| Documentation | 83 float32 and 16 float64 native tests pass; site and notebook snapshots build |
| Final codec check | 35 artifact tests pass after canonical payload ordering |
| Zero-mass reference | 93 scalar, matrix, weighted, component, barycentre and dynamic-strategy MMD tests pass; independent angular expectations verify inert zero-mass atoms |
| Exact transport | 43 tests pass, one skip; direct broadcast cloud axes agree with independent sphere-chord distances |
| Full fast suite | 2,581 tests pass, 166 skips; no failures in that run |
| Full repository | `just check` ran and stopped at the research catalogue after a concurrently added `LogisticResult.save` method; current full typing also reports concurrent errors |
| Remaining repository gates | Standalone LaTeX and AgentSync checks pass |

The latest full strict check reported six errors in concurrent edits to
`api/artifacts/_prediction.py`, `estimation/spatial/panel.py` and
`tests/estimation/spatial/test_panel.py`. Scoped typing excluded exactly those
three files through command arguments; repository typing policy was not changed.
The missing catalogue symbol is `statadic.api.regression.LogisticResult.save`.
These edits were preserved for their owning work. The full-workspace acceptance
item remains open rather than claiming a clean `just check` result.

The source guard rejects known dynamic-loader, reflection, FFI/process and host
serialization bypass patterns. It is not a sandbox for malicious Python and does
not remove NumPy/SciPy from JAX's own transitive dependencies.

Empirical-gradient witnesses differentiate floating coordinate/logit parameter
trees with integer counts fixed. They do not promise whole-carrier integer AD,
smooth transport assignments, or smooth Euclidean norms at coincident points.
The PSD cone vertex uses an explicit generalized derivative. The GEL envelope
exception is first-order and requires a converged interior multiplier.

Local cached publications were re-exported through `just python::studies::export`
from their retained trials using caller-default float32 accounting. Summaries now
name `clopper-pearson-betainc-v2` and their dtype. Original exports remain intact
under ignored `build/studies/pre-jax-accounting/`; trial source provenance and
simulation outcomes were not changed or represented as current-source trials.
