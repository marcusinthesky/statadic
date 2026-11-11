# Contributing

## Setup

Use Linux x86-64 or ARM64 with Git and Nix. Keep uv available for Copier updates.
Initial setup downloads locked dependencies; later checks use prepared inputs.

```bash
git init -b main
bash infra/nix/install-devenv.sh
devenv shell
just setup
```

Run `just --list` to discover this project's commands.

## Your first plan

```bash
just plan-new project "Project plan"
just task-new .context/plan/project first-step "Make the first useful change"
just sync
just check
```

Open [ROADMAP.md](ROADMAP.md) and follow the plan link. Edit the task's frontmatter
and synchronize to refresh its parent's table. Each parent shows immediate
children; its own status is authored separately.

## Your first reference

```bash
just reference-new first-source "First source"
just sync
just check
```

Edit the source's flat CSL metadata in
`.context/reference/first-source/README.md`. Its directory is the citation key.
Synchronization derives `src/latex/refs.bib` from these records.

## Try the code

```bash
just python::run
just lean::build
just lean::blueprint-web
just latex::build
```

## Find your way

- [Project context](.context/README.md) explains the installed workflows.
- [The Python workspace](src/python/README.md) explains code and commands.
- [The Lean workspace](src/lean/README.md) explains code and commands.
- [The Latex workspace](src/latex/README.md) explains code and commands.
- [AgentSync](.agents/README.md) manages shared instructions and native configuration.
- [Architecture](ARCHITECTURE.md) explains ownership and updates.

Edit source, run `just sync`, then `just check`.
Formatting and synchronization are explicit writes. External links run separately
through `just links`.

## Update

Commit local work, then run:

```bash
env -u PYTHONPATH -u PYTHONHOME uvx --from copier==9.16.0 copier update --vcs-ref v0.1.0-alpha
```

Review the diff and conflicts, run setup, sync and check. The explicit reference pins
the alpha release. Adding capabilities preserves saved selections. Removing them
requires manual migration because Copier can remove previously owned paths.
Actual records remain project-owned.

This project started with [Exordio](https://marcusinthesky.github.io/exordio/); see the [alpha release](https://github.com/marcusinthesky/exordio/releases/tag/v0.1.0-alpha).
