---
description: Project-owned source and explicit maintenance.
---

# Architecture

Copier owns shared configuration and starter code. This project owns its purpose
and authored records. Creation commands never overwrite existing paths, and actual
records never occur in template output. Read adjacent READMEs for commands.

When selected, Insitu derives marked views from source metadata. Task paths supply
identity; parents show immediate children. Status belongs to each authored item.
Declared links are canonical repository-relative files; ordinary body links retain
URL and fragment syntax. Schematter validates small contracts and entry budgets.

Devenv pins system tools. Workspaces and PEP 723 repository scripts commit frozen dependency locks.
Nix definitions and static document contracts live under infra/. Setup may
download inputs and prepare ignored environments; checks never repair tracked
files. Synchronization and formatting are explicit writes. Insitu runtime preparation preserves immutable caches. AgentSync is built
from pinned Rust source by Nix and needs no npm installation.

AgentSync, when selected, checks all destinations before projecting shared sources.
Authored files and unexpected links stop apply. SecretSpec, when selected, resolves
credentials only for an explicitly invoked subprocess. Nix is the shared
infrastructure; no cloud resources are created.

Keep Copier answers in Git. Tagged updates preserve selections or add features;
review conflicts before committing. Removing a feature or adopting an existing
repository requires manual migration: Copier can delete previously owned paths.

The distribution is [Exordio](https://github.com/marcusinthesky/exordio).
