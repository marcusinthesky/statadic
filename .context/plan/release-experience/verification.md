# Implementation verification

Implemented against base revision `52268aa5078d94c9bf8b092c7655fbf3ae54031a`.
The working tree is uncommitted. No release, deployment, push or human semantic
approval was performed.

## Executed checks

| Check | Observed result |
| --- | --- |
| `devenv shell -- just check` | Passed again on the final shared tree, including the concurrent Lean additions |
| Fresh scientific fast suite | 2,677 passed, 161 skipped, five warnings |
| Documentation tests | 64 passed; separate x64 native process: 14 passed, 50 deselected |
| Formal integrity/registry tests | 26 passed; compiled proofs and blueprint HTML/PDF built |
| Installed artifact qualification | Retained wheel, rebuilt-sdist wheel and standalone sdist development passed; 45 typing tests |
| Default standalone artifact command | Both build routes passed again after adding artifact-race protection |
| Installed-wheel precision references | 23 passed in float32 and 23 passed in float64, each in an isolated process outside the checkout |
| Evidence/budget behavioral tests | Nine passed; tampering, stale sources, failed commands, incomplete lanes and transclusions exercised |
| Formatting | Root formatting executed; disposable 700-file checkout produced no source diff |
| Strict fresh Sphinx/lab build | Passed again with six executed snapshots, source downloads and the final proof/source freshness check |
| Browser inspection | 74 HTML targets; 1,440- and 390-pixel layouts, light/dark, rendered equations, API/proof/lab pages, no JavaScript errors or page overflow |

The rebuilt Lean 4.35.0 toolchain occupies 3.1 GiB locally, excluding dependencies
and provisioning overhead. This is not a hosted-runner peak-storage measurement.
The native-library fix also allowed NumPy 2.4.6 and JAX 0.11.0 to import without
manual environment overrides. The installed STATADIC version remains 0.4.1.

The first-use source budget is **1,168 words versus 6,166**, an **81.06% reduction**,
below the frozen 1,541-word cap. It includes code/text transclusions and retained
compatibility targets. Comprehensive reference prose remains outside this cap;
the check reports its size separately.

## Retained local evidence

Logs, JUnit XML, wheel hashes and screenshots are in ignored `build/ci/`, rather
than temporary directories. The final root check log is `build/ci/check-final.log` (the earlier pass remains
in `build/ci/check.log`); artifact
results are under `build/ci/default-distribution/`; precision reports are
`build/ci/installed-precision-float32.xml` and `build/ci/installed-precision-float64.xml`.
The maintained browser command writes `build/ci/browser/report.json` and PNGs.
The site's own provenance and per-lab sidecars are under the docs app's generated
HTML output. They distinguish origin revision from current build identity.

The CI workflow uploads diagnostics, the built site and certification. The
manual/tag release workflow retains artifacts, source/dependency identities,
JUnit results, scientific dispositions, proof scope and generated site. Retention
uses the repository-configured artifact period; no automatic publication occurs.

## Remaining evidence and concurrent work

Cold-cache and warm-cache hosted x86-64/ARM64 runs have not been observed for
these changes. The complete locked external-reference and expensive Monte Carlo
qualification lanes have not been rerun as a release campaign. The new workflow
executes them; the local precision subset is not equivalent to that campaign.

During final verification, separate changes appeared in the Lean library,
claim registry, blueprint and Gretton literature record, together with a new
finite-MMD proof task. They were not authored or reverted by this implementation.
The final root check also passed with those additions present. This establishes
compilation and automated integrity checks, not human semantic approval; that
review belongs to the formal-literature task.

Human source-fidelity review and the all-method proof programme remain distinct
from compilation and unresolved as documented by their owning records. The
publishing companion is an accepted design and backlog only; no paper-building
runtime API was added.
