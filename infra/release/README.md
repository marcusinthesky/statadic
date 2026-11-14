# Releases

The [release workflow](../../.github/workflows/release.yml) follows Insitu's
tag-driven release process. Only the public `statadic` package is published.
Internal applications and companion packages keep their separate versions.

The documentation site has a separate [GitHub Pages workflow](../../.github/workflows/pages.yml).
It builds on pull requests and publishes successful builds from `main`.
See [GitHub Pages](#github-pages) for setup.

Before publishing, configure these account settings:

1. Create the GitHub environment `release` in `marcusinthesky/statadic`.
   Restrict deployments to release tags and configure required reviewers if desired.
2. Register a [PyPI Trusted Publisher](https://docs.pypi.org/trusted-publishers/)
   for project `statadic`, owner `marcusinthesky`, repository `statadic`,
   workflow filename `release.yml`, environment `release`. Use a pending publisher
   if the project has not been created. No PyPI API token is needed.

For each release:

1. Update `src/python/packages/statadic/pyproject.toml`, the changelog and the
   matching `statadic` version in `src/python/uv.lock`. Use the locked uv tooling
   to refresh the lock. Review and refresh the Lean public API inventory with
   `just lean::inventory-refresh` so its package version agrees.
2. Open a pull request. CI validates a canonical wheel and sdist on x86-64 and
   ARM64, with four complete package-test shards per architecture. Dependency
   preparation runs before expensive tests. Repository and documentation checks
   retain their separate scopes.
3. After merging, wait for **Package candidate** for the exact `main` commit,
   then push an exact package-version tag, for example `v0.5.0a2`.
   Supported prerelease suffixes are `aN`, `bN` and `rcN`; they are marked as
   prereleases on GitHub. Versions are explicit; commits do not trigger bumps.

The version job rejects mismatched package, lock, registry and tag versions.
The reusable package workflow checks both supported architectures. The tag
workflow selects its exact-commit `package-candidate` artifact and verifies
producer metadata, package quality, complete shard coverage, installed comparison
reports, source/lock identities and distribution hashes. PR candidates cannot be
published. PyPI receives those exact files without rebuilding or repeating tests.
The wheel is platform independent; dependencies such as JAX supply platform wheels.

Candidates remain available for 30 days. An absent or expired candidate stops
publication early. Dispatch CI at the tagged commit, wait for its package
candidate, then retry Release with that producer run ID:

```bash
gh workflow run ci.yml --ref v0.5.0a2
gh workflow run release.yml --ref v0.5.0a2 -f producer_run_id=RUN_ID
```

Dispatched commits must be reachable from `main`. To create the next release:

```bash
git tag -a v0.5.0a2 VALIDATED_MAIN_SHA -m "STATADIC 0.5.0a2"
git push origin refs/tags/v0.5.0a2
```

Retries require existing PyPI and GitHub files to match the validated bytes and
upload only missing files. Conflicting published files stop publication.
Leave `v0.5.0a1` unchanged: its failed pipeline skipped PyPI publication and the
existing GitHub prerelease had no distribution assets when reviewed. Check the
current publication state independently before retrying any version.

The package candidate does not depend on docs or workspace application checks.
Repository CI still reports those checks separately and its aggregate requires
both scopes. Scientific qualification remains a separate command and claim.

The legacy documentation/scientific preflight can consume a complete repository
receipt through `--preflight-check-receipt PATH`. Its retained reports must prove
complete, non-overlapping execution and successful repository checks. The release
archive verifies that evidence again. This command remains separate from package
publication; a package-only receipt cannot replace a full repository receipt.

The legacy manual preflight and local `just release-preflight` run fresh checkout
checks by default. They are separate from publication. Qualification rejects
imported CI receipts.

After PyPI succeeds, the workflow creates a GitHub release with those same
distributions and `SHA256SUMS`. Detailed preflight evidence remains in each
workflow run's `release-evidence-*` artifacts. A pull request or a manual preflight
never publishes. Hosted preflight does not establish full scientific qualification;
retain the separate `just release-check` evidence before making that claim.

Run the lightweight version gates locally in the prepared environment:

```bash
uv run --project src/python --offline --frozen --package statadic-docs python infra/release/check_version.py --tag v0.5.0a2
uv run --project src/python --offline --frozen --package statadic-docs python -m unittest discover -s infra/release -p 'test_*.py' -v
```

## Offline consumer installation

Before the expensive validation jobs, prepare and check the source-free publishing
environment with the pinned Python runner:

```bash
devenv shell --profile python-ci -- uv run --project src/python --offline --frozen --all-packages --group reference python infra/release/check_offline_install.py --prepare --output build/ci/offline-install
```

The helper exports the production dependency closure to `pylock.toml`, retaining
the reviewed lock's exact artifact URLs and hashes. Only `--prepare` downloads
missing artifacts, and uv selects wheels for the current interpreter and
architecture. When building the workspace wheels, it uses hashed build constraints
exported from the workspace's existing development lock; `--prepare` also fetches
those build inputs. The helper then discards the prepared environment and repeats
the install in a fresh environment with network and registry resolution disabled. It
installs the newly built workspace wheels without dependency resolution, checks
their dependency metadata, and verifies imports come from that isolated
environment. `build/ci/offline-install` retains the exported lock and successful
install receipt. Run without `--prepare` to require all artifacts to be cached.

The publishing reconstruction test uses the same installer before exercising the
installed wheels. A successful workspace `uv sync` or a cached package index alone
does not establish that this independent offline consumer installation works.

## Release evidence

The early CI gate uses the locked workspace backend to build a fresh sdist
offline and checks its required qualification inputs before the main matrix.
Run that bounded check in the pinned workspace with
`just python::statadic::test-artifacts --inputs-only`. It does not qualify wheels.
The complete `just python::statadic::test-artifacts` check installs both wheel
build routes, exercises Polars and Arrow through the wheel's `[data]` extra in
float32 and float64, and checks standalone sdist development. Its smoke driver
ships in the sdist; documentation notebooks have their own execution lane.

The legacy manual workflow runs a bounded artifact preflight. A complete release
qualification additionally needs the exact installed-wheel studies, source reviews,
consumer acceptance and resource measurements selected by its frozen scope.
Neither a successful site build nor a preflight qualifies the release.

The [study mappings and retained reports](../../src/python/packages/statadic/docs/validation/research.md)
distinguish registration, historical evaluations and explicitly selected release
evaluations. Inspect each download's wheel identity, scenarios, attempted and failed
counts, uncertainty and scientific disposition. An absent report is absent evidence.
Human source-fidelity review remains separate from Lean compilation.

## Release qualification

From the repository root, `just release-check` qualifies retained distributions
and writes `build/release/manifest.json`, logs, JUnit reports, documentation and
hashes. Keep that directory empty before a new run by retaining or moving the
previous bundle. Long scientific campaigns belong on persistent local machines.
Qualification requires a complete reviewed method catalogue and one retained
installed-wheel evaluation export per required study, supplied through repeated
`--campaign PATH` options. Each must include all scenarios, at least 10,000
replicates and the evaluation resampling budget. Source and ownership gaps stop
the command before expensive execution. Evaluations must match the release wheel,
and adverse scientific results stay visible in their per-cell summaries.

Release and installed-campaign preparation use the same sdist-to-wheel build route
and a fixed archive epoch (`315532800`). Qualification compares exact wheel
archives, so equal implementation payloads with different ZIP timestamps are not
treated as the same release artifact.

The legacy manual workflow runs `just release-preflight` on both supported CPU
architectures, retains a complete tar archive and verifies its transport. This
bounded preflight does not claim scientific qualification. Verify a full bundle
with the locked app runner:

```bash
uv run --project src/python --offline --frozen --package statadic-docs python src/python/apps/docs/scripts/release.py --verify build/release
```

Successful test execution does not turn an inconclusive study into qualified
evidence. Review the recorded per-cell scientific dispositions and formal scope.
Use `--verify-archive PATH --allow-preflight` to check a preflight archive's
integrity without promoting its scope to qualification.

Use `--scope PATH` for an explicit bounded qualification request. The resolver
retains the complete catalogue and freezes exact selected variants. Missing source
reviews, adapters or adverse required cells still block release. The default
command continues to require the complete all-method catalogue. Scoped
qualification also consumes `--consumer-repo PATH` for the adapted installed-wheel
fixtures and `--performance PATH` for matched-reference evidence. Its raw records
must match the same wheel, scope and lock.

The alpha's explicit `report` policy accepts completed bounded measurements for
every requested workload, retaining the actual case list and all adverse ratios.
Legacy and default `superiority` scopes still require all registered subcases and
ratios of at most 0.90 in each of three independent sessions. Numerical JUnit
lanes cover explicit named witnesses. The large censored-panel accuracy and refit
obligation remains separate and cannot inherit small Gaussian-control results.

Retained scoped bundles require the same explicit `--scope PATH` on verification.
The verifier re-derives contract records from their retained producers instead of
accepting a passing manifest status alone. Source correspondence, human fidelity,
statistical qualification, downstream acceptance and resource results remain
independent evidence dimensions.

## Consumer acceptance

The pre-release downstream application adapter is
`statadic_docs_labs.consumer_acceptance.run_acceptance`. Supply the read-only
consumer repository, a new output directory, the candidate wheel and its already
prepared isolated interpreter. It reads the exact locked application revision,
retains original and adapted sources and runs the deterministic
`numerical_qualification(64)` witnesses with the canonical `estimate` verb. Only
frozen application sources enter the child import path, and STATADIC must match the
installed wheel byte for byte. Bounded dyadic and frontier fixtures, actual
factor-SAR fits (spherical and diagonal innovations) and an innovation and
total-risk function complete the consumer matrix, with independent dense-Gaussian
and least-squares checks. This evidence does not replace statistical or performance
qualification, a full retained-data DVC replay, or the post-alpha production
migration.

## GitHub Pages

The [GitHub Pages workflow](../../.github/workflows/pages.yml) builds the
documentation site on pull requests and pushes to `main`. Pull requests retain the
`github-pages` artifact without deploying. Successful `main` builds deploy to the
`github-pages` environment, and manual runs deploy only from `main`. The build
prepares the locked Python and Lean environments, certifies proofs, builds the
blueprint and exports the lab snapshots before uploading HTML. Pages configuration
supplies `STATADIC_DOCS_BASE_URL` for Sphinx's canonical URLs, and the site's
relative asset links support the `/statadic/` project path.

In **Settings, Pages, Build and deployment**, select **GitHub Actions** as the
source. Configure the `github-pages` environment to allow deployments from `main`.
The expected default address is `https://marcusinthesky.github.io/statadic/`, and
an existing custom domain is read from Pages configuration. These account settings
need repository administration access. Build logs and proof certification are
retained in `pages-diagnostics` for seven days.

Study reports enter the site only through explicit selection, for example
`just python::docs::build --study-export /absolute/path/to/publication`. Ordinary
selections are labelled historical. The release orchestrator first verifies exact
wheel and scope correspondence, then passes `--release-evidence` and the matching
repeated `--study-export` paths. There is no ambient campaign scan. An empty release
selection supplies no current-release Monte Carlo evidence. Build provenance
records selected paths and export-manifest hashes, and rendering does not
independently grant scientific qualification. When selected reports contain
operating-characteristic bounds, the site plots their retained failure-inclusive
rates, uncertainty and frozen thresholds. Missing bounds produce an explicit absence
note. Resource qualification still requires the separate matched-reference
campaign, and no chart implies a memory win.

The formal mathematics pages require the prepared Lean workspace. The build runs
`just lean::certify` and the blueprint build before Sphinx, then publishes
source-linked statements, the shared bibliography, Lean sources and proof evidence.
See the [Lean guide](../../src/lean/README.md).

`build/html/build-provenance.json` identifies the source revision, dependency
versions, canonical notebook and document hashes and requested runtime settings.
The renderer executes the notebook sources into native Sphinx pages and exports
their source downloads. This build provenance identifies the rendered inputs,
while scientific evidence retains its own method-specific acceptance rules.

## Validation lanes and CI

Validation belongs to a method, configuration, output and reference version. A
green default suite does not certify every method.

Tests keep visible Arrange, Act and Assert phases. Neutral inputs and strict numeric
assertions may be shared, but independent references do not reuse STATADIC
centering, weighting, residual, covariance, projection or p-value calculations.
Family tests state intercepts, retained samples, weight meanings, covariance
corrections, degrees of freedom, objective and null-law conventions explicitly.

`reference` marks a comparison with independent software, an analytic result, a
published value or a trusted fixture. `external_reference` marks a live comparison
with an optional dependency and also carries `reference` and
`requires_optional(name)`. An independent oracle does not by itself make a test
integration scope; scope follows the boundary the assertions exercise. These
markers describe assurance and select tests. They are not a coverage denominator.
Missing optional dependencies skip locally, and broken installed dependencies fail.
The required lane rejects absent comparators, empty selections, skips, xfails and
unexpected passes of expected-failure comparisons.

From the repository root:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.cache/statadic-reference-venv"
uv sync --locked --package statadic --group reference --project src/python
just python::statadic::test-external
just python::statadic::test-fresh
```

The locked lifelines AFT oracle is 0.30.0, tested with the existing pandas 3
dependency. Lifelines 0.30.3 comparisons also passed in an isolated pandas 2
environment, but that release caps pandas below 3 and the pinned Nix resolver does
not support the resulting project-conflict fork. Keep reference environments
separate to avoid pruning the shared workspace.

The test process keeps the legacy x64 default only when `JAX_ENABLE_X64` is unset.
Run matched, strict precision references in separate caller-configured processes:

```bash
JAX_ENABLE_X64=false just python::statadic::test-precision-references --junitxml=precision-float32.xml
JAX_ENABLE_X64=true just python::statadic::test-precision-references --junitxml=precision-float64.xml
```

The precision lane compares linear fitting, Student-t intervals, tied
Pearson and Spearman correlations and weighted energy distance with SciPy, and
WLS/HC3 and Poisson fits with statsmodels. It also compares scaled exact transport
(SciPy assignment), a scaled GMM witness (SciPy least squares), Weibull and
lognormal AFT (lifelines) and spatial log-time survival (spreg reductions and
independent Gaussian-tail witnesses). The lane attempts bfloat16 correlation and the
actual fitting backend, and records an unsupported QR dtype in JUnit rather than
skipping it. Precision jobs report separate selected-node counts. The full locked
reference lane still runs with its legacy x64 default, and precision runs do not
replace it.

The required CI reference jobs use the same locked dependencies in an isolated
environment, independently of the ordinary fast gate. Run the complete local
equivalent from the repository root:

```bash
just setup-references
just validate-references --output build/reference-validation
```

The output directory must be empty. The full float64 reference lane and the
float32 and float64 precision subsets each retain a log, JUnit report and selected
nodes. The manifest records dependency versions, the pinned layout-enforcer Git
origin, caller precision and comparison counts by comparator. An empty selection,
a failed or skipped comparison, or a missing spreg, statsmodels, linearmodels or
lifelines comparison fails validation. Precision subsets require statsmodels and
lifelines comparisons separately. The repository command also supplies the
authored catalogue. Each declared comparison requires its exact parameterized node
and comparator, and the manifest retains unmapped executions separately. Source,
tests, executed recipes, lock and catalogue are fingerprinted before and after the
run. Independent-formula mappings outside these lanes stay unexecuted by this
command. These are live parity checks, not statistical qualification.

Branch protection should require both `Reference validation (ubuntu-24.04)` and
`Reference validation (ubuntu-24.04-arm)` CI statuses. The generated
[reference index](../../src/python/packages/statadic/docs/validation/references.md)
is the configuration-to-test map for constructors and explicit compositions.

Constant-correlation shrinkage comparisons normalise PyPortfolioOpt's sample
covariance to the T divisor before fitting, and panel peers require linearmodels
7.1. Property profiles are selected with `--hypothesis-profile`: `statadic`
preserves each `@examples(N)` cost tier, `statadic-reference` caps it at 20, and
`statadic-explore` multiplies it by five. Untiered defaults are 100, 20 and 500.
Profiles print replay blobs, leave deadlines disabled and keep health checks
active. Required semantic modes have separate test nodes, and important
counterexamples stay as literal regressions.

### Statistical qualification

Run the scheduled study set independently of the fast tier:

```bash
just python::statadic::test-qualification -k 'not dyadic' --junitxml=qualification.xml
```

Dyadic qualification is manual-only and unexecuted. Its two node-size cells each
need 10,000 independent trials with 999 production node-bootstrap draws per trial.
Warm measurements project about 8 hours per cell. Run the canonical study explicitly:

```bash
just python::statadic::test-qualification \
  tests/qualification/test_panels.py::test_dyadic_complete_node_bootstrap_coverage \
  --junitxml=qualification-dyadic.xml
```

Each scenario specifies its DGP independently of the implementation and uses 10,000
independent trials. Stable scenario, trial and stream IDs separate data,
resampling and optimization randomness. Randomized inference uses 999 inner draws
unless a family contract requires otherwise. Neither worker identity nor Python's
randomized hash chooses seeds.

The default significance is 0.05. A one-sided 95% exact binomial upper bound for
failure-inclusive rejection must be at most 0.06, and the failure-rate upper bound
at most 0.005. For nominal 95% intervals, the coverage lower bound must be at least
0.94, counting unavailable intervals as uncovered. Results are qualified,
excess-error or inconclusive, and inconclusive results do not satisfy
qualification. Bounds apply per scenario. JUnit properties retain requested, valid
and failed counts, conditional rejection, failure-inclusive bounds and
qualification status.

Pull requests run the complete fast suite, locked external comparisons and the
separate float32 and float64 precision-reference processes. Weekly and manually
dispatched Actions runs execute the non-dyadic statistical studies in the locked
environment. Current-reference exploration runs in an isolated package copy,
records its environment and does not rewrite the reviewed workspace lock.

Replay the original adverse cells separately. An assertion failure from these
commands is a statistical qualification failure and must not be converted to a skip
or expected failure.

```bash
just python::statadic::test-qualification tests/qualification/test_panels.py -k gaussian_sar
just python::statadic::test-qualification tests/qualification/test_resampling.py -k stationary
just python::statadic::test-qualification tests/qualification/test_returns.py -k direct_sharpe
```

Run the held-out mechanism diagnostics with the reference group installed. Passing
a diagnostic does not qualify any method, and diagnostic studies use separate
held-out trial IDs.

```bash
just python::statadic::test-all tests/qualification/test_sar_diagnostics.py \
  tests/qualification/test_stationary_diagnostics.py \
  tests/qualification/test_sharpe_diagnostics.py -n 0 -o junit_family=legacy \
  --junitxml=adverse-diagnostics.xml
```

The paper's validation records live in
`src/latex/projects/01_statadic/analysis/validation-records.yaml`.
