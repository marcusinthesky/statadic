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

The initial config is two tasks × two packages × one repetition. After calibrating
the contracts and reviewing the pilot, use `tasks: "all"`, five repetitions and
optionally multiple effort levels. A new configuration requires a new campaign.
No silent retries or overwrites; retain failures and incomplete usage.

`pricing`, when provided, contains `input`, `cached_input`, and `output` in USD per
million tokens, plus `as_of` and `source`. It is an estimate, not a subscription
invoice. Reasoning is already included in provider output tokens; cached input
is a subset of input. Never add either category twice.

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
