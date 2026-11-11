# STATADIC documentation application

The canonical site source is `../../packages/statadic/docs/`; this app owns the
Sphinx configuration and seven native marimo labs. The STATADIC wheel does not depend
on documentation tools. From the repository root:

```bash
just python::docs::build
just python::docs::serve
just python::docs::lab ground-geometry
just python::docs::check
just python::docs::check-size
just python::docs::check-browser
```

The build writes HTML, lab snapshots and notebook downloads under the ignored
`build/` directory. The snapshots run locally with default inputs; use the lab
command for interactive controls. They do not require an online notebook host.
Dependencies are prepared by `just python::setup`, including the pinned TeX
resources used by report tests. `just python::docs::setup` prepares those TeX
resources explicitly and may use the network; checks compile with cached inputs.
Report compilation uses the repository Tectonic tool and shared template.
External link checks remain separate from the offline build.

Study reports enter the site only through explicit selection, for example
`just python::docs::build --study-export /absolute/path/to/publication`.
Ordinary selections are labelled historical. The release orchestrator first
verifies exact wheel/scope correspondence, then passes `--release-evidence` and
the matching repeated `--study-export` paths. There is no ambient campaign scan.
An empty release selection explicitly supplies no current-release Monte Carlo
evidence. Build provenance records selected paths and export-manifest hashes;
rendering does not independently grant scientific qualification.
When selected reports contain operating-characteristic bounds, the site plots
their retained failure-inclusive rates, uncertainty and frozen thresholds.
Missing bounds produce an explicit absence note. Resource qualification still
requires the separate matched-reference campaign; no chart implies a memory win.

The formal mathematics pages require the prepared Lean workspace. The build
runs `just lean::certify` and the blueprint build before Sphinx, then publishes
source-linked statements, the shared bibliography, Lean sources and proof
evidence. `just lean::programme-check` reports remaining literature and proof
obligations with a nonzero exit; ordinary checks do not claim all-method proof
completion. See the [Lean guide](../../../lean/README.md).

The Shibuya site uses local serif fonts, black text, Delft-blue links and the
supplied animated curve-fit logo/favicon; asset provenance is documented in
[the branding notes](static/brand/README.md).
Navigation separates Start, Tutorials, How-to, Concepts, Reference and Evidence.
The short first regression leads to a four-part OLS chapter, whose executable
recipe also powers the native regression marimo lab. Site snapshots retain
fixed settings; local notebooks provide reactive controls. `word-budget.json`
freezes the approved first-use scope
and 6,166-word baseline; the checker includes authored text/code transclusions
and caps the replacement path at 1,541 words. Detailed references are outside
this cap, and their total is reported separately.

Browser inspection is an explicit command after building: install Chrome or
Chromium, or pass `--browser /path/to/executable`. Its Python dependencies are
locked in the docs development group. It checks responsive white layouts under
light/dark system preferences and saved theme choices, local serif fonts,
readable articles, nested navigation, search, local targets and rendered mathematics. Screenshots
and their hashes are retained under `build/ci/browser/`.

`build/html/build-provenance.json` identifies the current source revision,
dependency versions, notebook/document hashes and requested runtime settings.
Each lab snapshot has a companion provenance JSON. `labs.json` records the
notebooks' origin revision separately; it is not current verification evidence.

From the repository root, `just release-check` qualifies retained distributions
and writes `build/release/manifest.json`, logs, JUnit reports, documentation and
hashes. Keep that directory empty before a new run by retaining/moving the
previous bundle. Long scientific campaigns belong on persistent local machines.
Qualification requires a complete reviewed method catalogue and one retained
installed-wheel evaluation export per required study, supplied through repeated
`--campaign PATH` options. Each must include all scenarios, at least 10,000
replicates and the evaluation resampling budget. Source/ownership gaps stop the
command before expensive execution. Evaluations must match the release wheel;
adverse scientific results remain visible in their per-cell summaries.
Release and installed-campaign preparation use the same sdist-to-wheel build
route and a fixed archive epoch (`315532800`). Qualification compares exact wheel
archives; equal implementation payloads with different ZIP timestamps are not
silently treated as the same release artifact.
The manual/tag workflow runs `just release-preflight` on both supported CPU
architectures, retains a complete tar archive and verifies its transport. This
bounded preflight deliberately does not claim scientific qualification. Verify
a full bundle with the locked app runner:

```bash
uv run --project src/python --offline --frozen --package statadic-docs python src/python/apps/docs/scripts/release.py --verify build/release
```

Successful test execution does not turn an inconclusive study into qualified
evidence. Review the recorded per-cell scientific dispositions and formal scope.
Use `--verify-archive PATH --allow-preflight` to check a preflight archive's
integrity without promoting its scope to qualification.

The pre-release downstream application adapter is
`statadic_docs_labs.consumer_acceptance.run_acceptance`. Supply the read-only consumer
repository, a new output directory, the candidate wheel and its already prepared
isolated interpreter. It reads the exact locked application revision, retains
original/adapted sources and runs the actual deterministic
`numerical_qualification(64)` witnesses with the canonical `estimate` verb.
Only frozen application sources enter the child import path; STATADIC must match the
installed wheel byte for byte. Bounded dyadic/frontier fixtures, actual factor-SAR
fits (spherical and diagonal innovations) and innovation/total-risk
function complete the consumer matrix, with independent dense-Gaussian and
least-squares checks. This evidence does not replace statistical or performance
qualification, a full retained-data DVC replay, or the post-alpha production migration.

Use `--scope PATH` for an explicit bounded qualification request. The resolver
retains the complete catalogue and freezes exact selected variants; missing
source reviews, adapters or adverse required cells still block release. The
default command continues to require the complete all-method catalogue.
Scoped qualification also consumes `--consumer-repo PATH` for the adapted
installed-wheel fixtures and `--performance PATH` for matched-reference evidence.
Its raw records must match the same wheel, scope and lock. The current alpha's
explicit `report` policy accepts completed bounded measurements for every requested
workload, retaining the actual case list and all adverse ratios. Legacy/default
`superiority` scopes still require all registered subcases and ratios at most
0.90 in each of three independent sessions. Numerical JUnit lanes cover
explicit named witnesses; the large censored-panel accuracy/refit obligation
remains separate and cannot inherit small Gaussian-control results.

Retained scoped bundles require the same explicit `--scope PATH` on verification.
The verifier re-derives contract records from their retained producers instead of
accepting a passing manifest status alone. Source correspondence, human fidelity,
statistical qualification, downstream acceptance and resource results remain
independent evidence dimensions.
