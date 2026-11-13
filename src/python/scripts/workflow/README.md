---
description: Locked planning and reference commands, independent of the Python application workspace.
---

# Document workflows

The root justfile exposes plan/task/reference creation, `sync` and `check`.
These commands run this PEP 723 script with its pinned dependencies and sidecar
`script.py.lock`; selecting workflows does not select the Python application example.

`just setup` explicitly prepares the script environment and its native Insitu
executable. Preparation changes only the installed executable in uv's disposable
script environment, preserving shared package caches. Run setup again after
clearing caches or changing dependencies. Ordinary commands use frozen, offline
dependencies and never prepare native executables implicitly.

Creation refuses existing destinations. Schematter validates document contracts;
Insitu owns generated regions. `check` reports invalid records and stale views
without repairing them. With references and LaTeX, `sync` derives the bibliography
through Pandoc. Keep canonical metadata in the authored records.
Synchronization checks entry budgets after refreshing generated views, so a
compact template can repair an oversized old index without manual region edits.
