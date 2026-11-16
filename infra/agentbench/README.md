# AgentBench environment

This is the **controller** development shell and two devenv-built OCI treatments.
There is no Dockerfile, model proxy or custom container engine. Pydantic AI's
DockerSandbox owns execution/cleanup; Pydantic Evals owns cases and scoring.
The model client and credentials stay outside the sandbox. This evaluates
**Pydantic Coder powered by the configured model**, not the official Codex CLI.

## Setup (Linux)

Use a dedicated worker with Nix, devenv and a local Docker daemon (or rootless
Podman). Do not expose a Docker socket, host credentials or the checkout to agents.
Containers share the host kernel; they are not a VM security boundary.

```sh
cd infra/agentbench
devenv shell
just setup
just check
just tex-cache
just images
cp config.example.json config.local.json
```

Set an explicit supported Pydantic model identifier in config.local.json (e.g.
the `openai-responses:` provider plus your chosen API model snapshot), and the two
immutable `sha256:...` image IDs printed by `just images`. No default model means
no accidental charged runs. Configure authentication on the controller through
the provider's normal environment variables, never in JSON or image layers.

For an existing ChatGPT/Codex subscription, set `model` to
`openai-codex:gpt-6-luna` and leave `pricing` null. Pydantic's native provider
reads the existing `~/.codex/auth.json` (or `$CODEX_HOME/auth.json`) on the
controller; run `codex login` first if needed. No API key or model proxy is
required. Subscription usage is not a per-token USD invoice, so dollar costs
remain unavailable. Never mount the login file or its directory into a sandbox.

For Podman, load the built images with devenv's registry/copy mechanism into its
local store, set `executable` to `podman`, and inspect IDs with `podman image inspect`.
The supplied convenience `images` recipe uses Docker. Use the same local runtime
for building/loading and executing. Remote Docker bind mounts are not supported.

## Freeze before measuring

The new controller has an independent uv project so model-provider dependencies
cannot alter STATADIC's scientific environment. `just setup` resolves its initial
`uv.lock`; retain and review that lock and this directory's generated `devenv.lock`
before collecting results. Do not update either during a campaign. Subsequent
commands require locked, offline dependencies. Preparation copies the controller
lock into the campaign and checks versions and hashes before execution.

Treatment images reuse `src/python/uv.lock` through uv2nix, installing only the
selected package plus common I/O/plotting utilities. Nix inputs are revision-pinned
in devenv.yaml. The local TeX cache is primed in an explicit network step, then
copied into the immutable image. Archive image IDs/digests, OCI exports, both
lockfiles, the repository commit and the TeX cache with a published campaign.
Image IDs pin the actual built runtime, including docs and TeX assets.

The image layer explicitly replaces devenv's default source-copy layer. It must
contain **no checkout, developer instructions, test suite, grading code or gold
answers**. The runtime uses a read-only root, no network, non-root user, bounded
memory/processes, read-only public inputs and one writable work directory.
The Pydantic backend overrides image entrypoints, so PATH and binaries are
provided explicitly, rather than relying on a devenv shell startup script.

## Pilot and full run

```sh
just bench list
just bench prepare config.local.json campaigns/pilot --acknowledge-source-terms
just bench run campaigns/pilot                # plan only
just bench run campaigns/pilot --execute      # explicit model charges
just bench summarize campaigns/pilot --automatic
```

The initial config is two tasks × two packages × one repetition. A complete
20-task evaluation with five repetitions and optional effort levels is a
separately configured campaign, started only after assessing the pilot and
authorizing execution. A new configuration requires a new campaign.
No silent retries or overwrites; retain failures and incomplete usage.

`pricing`, when provided, contains `input`, `cached_input`, and `output` in USD per
million tokens, plus `as_of` and `source`. It is an estimate, not a subscription
invoice. Reasoning is already included in provider output tokens; cached input
is a subset of input. Never add either category twice.

## Documentation A/B pilot: 48 attempts

This exploratory pilot uses the first catalog entry in each family: PD0003,
PD0007, PD0119 and PD0036. Run three repetitions for each task, library and
documentation condition, with the fixed example model/effort/seed and budgets.
Use the existing Codex subscription provider; no paid API fallback is configured.
The historical 20-task baseline is retained separately and is not the new minimal
arm. Do not launch a full 20-task follow-up automatically.

```sh
# From infra/agentbench, in the pinned devenv shell after setup and tex-cache:
just images-all
docker image inspect statadic-agentbench-statadic:pilot statadic-agentbench-statsmodels:pilot statadic-agentbench-statadic-reference:pilot statadic-agentbench-statsmodels-reference:pilot --format '{{.RepoTags}} {{.Id}}'
cp config.docs-ab.example.json config.docs-ab.local.json
```

Replace the four placeholders in the ignored local configuration with the
corresponding immutable image IDs. `conditions.minimal` uses the two original
container names, and `conditions.reference` uses the two `-reference` names.
`just images` still builds only minimal images; `just images-reference` builds
only reference images. `just images-all` builds/copies all four serially.
Do not include the legacy top-level `images` key in a condition configuration.
The explicit model is `openai-codex:gpt-6-luna`; leave `pricing` null.

```sh
just bench prepare config.docs-ab.local.json campaigns/docs-ab-pilot --acknowledge-source-terms
just bench run campaigns/docs-ab-pilot
# After inspecting the complete 48-attempt plan and authorizing model execution:
just bench run campaigns/docs-ab-pilot --execute
just bench summarize campaigns/docs-ab-pilot --automatic
```

Every image exposes `/docs/INDEX.md`, root-package pydoc, runtime identity,
license and a hashed corpus manifest. Reference images add six grouped public
API references from explicit installed-object allowlists; STATADIC also includes
selected unmodified reference pages. The corpus is generated offline from pinned
sources without adding scientific dependencies. Corpus byte counts differ
between libraries and must be retained, rather than presented as matched.
No task-specific cookbook, expected answer, tests, repository agent instructions or private
calibration solution belongs in the corpus.

The same initial instruction in both conditions requests reading `/docs/INDEX.md`
before implementation. Both use `Coder(unrestricted_filesystem=True)`, giving
file tools access to `/input` and `/docs` inside the OCI workspace. This access
setting is shared; package isolation, read-only mounts,
network restrictions, credentials boundary and grading remain unchanged. Skills
are not injected. Changes to docs, prompts, packages or budgets require a new
frozen campaign rather than a mid-pilot adjustment.

The summary directory contains condition-level `summary` outputs, `per-task`
and `attempts` tables, and reference-minus-minimal `condition-differences` and
`task-condition-differences`, each in JSON/CSV/Markdown. Available token/cost
quality plots label all conditions. Means average repetitions within tasks before
equal task weighting; failures remain included. Completion, replay, TeX, numerical
checks, failure categories and descriptive documentation trace counters remain
separate. Trace counters do not establish actual reading or package compliance.
Observed input, cached-input and output token subtotals can be partial; unknown
full means and differences remain null. Subscription costs remain unavailable
with null pricing. Human-review fields and final grades remain null until
qualified review with actual evidence. Assess the pilot before deciding any
larger evaluation; these four previously observed tasks support descriptive
comparisons only.

Native request hooks append events to each run's `requests.jsonl`. Complete
usage requires all prepared attempts to return measured usage with matching
native response counts. Agent failure alone does not make usage incomplete:
a post-response budget failure can be fully measured. Unresolved requests,
unmeasured responses and count mismatches retain explicit incompleteness reasons;
observed subtotals are preserved without filling unknown full totals.

## Validation boundaries

The draft includes deterministic tests and an opt-in OCI smoke test. It does not
claim that building these Nix images, fetching all source datasets, or running a
paid model has already been validated on every host. Run the smoke test before
spending model tokens; validate native STATADIC and statsmodels solutions on each
source task before publishing performance claims. See the application README.

References: [devenv containers](https://devenv.sh/containers/),
[uv2nix](https://pyproject-nix.github.io/uv2nix/usage/getting-started.html),
[Pydantic DockerSandbox](https://pydantic.dev/docs/ai/harness/docker-sandbox/),
[Pydantic Evals lifecycle](https://pydantic.dev/docs/ai/api/pydantic_evals/lifecycle/).
