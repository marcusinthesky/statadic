# Formal mathematics implementation and verification

The accepted scope is a literature-first, all-method programme. The bridge and
initial finite proof batch are implemented; the all-method endpoint remains open.

The initial results are restricted specializations or proof steps of identified
published sources, including an explicitly identified online proof-book entry.
They are not the complete general source theorems or verified Python/JAX code.
Independent agent source comparisons are retained in [reviews](reviews/).
Human semantic review remains pending.

## Remaining scientific work

The [method registry](../../../src/lean/registry/methods.json) records unresolved
source and proof obligations. These include continuous-law extensions and the
remaining conclusions of Gauss–Markov, WLS and robust inference, GLM, GMM/GEL,
panel/spatial/survival guarantees, geometry and transport, bootstrap and random
permutation validity, optimization and specialized inference. Hybrid models
retain component guarantees only until a published combined result is sourced.

`just lean::programme-check` must fail until all required mapped obligations
are proved. Registry integrity passing is not proof-programme completion.

## Executed checks

- `devenv shell -- just check` passed. Complete no-Testmon scientific fast suite: 2,677 passed, 161 skipped, five warnings (466.76 seconds). Documentation: 55 passed; separate x64 native process: 14 passed, 41 deselected. Strict Ruff/typechecking, wheel build, Sphinx, six executed notebook snapshots, blueprint HTML/PDF, LaTeX and agent configuration gates passed.
- The compiled audit checked 88 first-party declarations in ten modules, including private helpers and unimported-source discovery. Six principal native labels map to four identified sources; one label is a published proof step and five are restricted specializations. All transitive axioms are within the three approved standard axioms. Additional OLS supporting results prove full-rank normal-equation/minimizer equivalence and specialized uniqueness.
- All 26 registry/integrity/inventory behavioral tests passed. They exercise admissions, private custom axioms, placeholder propositions, missing modules, source races, changed native LaTeX/types, stale evidence, missing mappings and invalid coverage closure.
- Real-browser verification passed for 70 local HTML targets, native TeX download hashes, seven pages with rendered MathJax equations, docstring-to-proof navigation, bibliography and a 390-pixel mobile viewport. No JavaScript or MathJax errors occurred.
- LeanArchitect's generated native TeX is built and bound to the corresponding declarations. Statement and proof text is extracted from the actual TeX artifacts, rendered in the documentation, and accompanied by downloadable source fragments. The three-page PDF companion was visually inspected; no overfull boxes or undefined references occurred. Additional cached math fonts were provisioned explicitly; later compilation passed offline.
- The 421-file scientific source/test baseline was checked: only four Python docstrings changed; their executable ASTs match the source repository. Runtime package metadata is unchanged. Documentation-only dependencies were locked separately in the shared workspace lock.
- The completion gate returned nonzero as intended for remaining all-method obligations; read-only checking preserves the existing certification artifact.

The API inventory is static source coverage: inherited and dynamic runtime members are not expanded. The 46 method-family records are an initial obligation catalogue, not a claim that every scientific proposition has already been discovered. The original 32 bundle candidates remain explicitly unreviewed and are not counted as completed published formalizations.

Slow/GPU, dedicated expensive statistical qualification, optional external-reference and standalone artifact qualification suites were not rerun. No deployment, publication, commit or push occurred. Human semantic review remains pending. Logs and browser/PDF artifacts are under `/tmp/statadic-formal-*`.
