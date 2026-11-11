---
type: task
title: Remove confirmed dead code across the package
description: Remove confirmed dead code across the package
links: [".context/plan/api-ergonomics/tasks/research-workflow/execution.md"]
status: done
---

# Remove confirmed dead code across the package

## Goal

Remove all confirmed dead code across the STATADIC package, including unrelated
findings, after integrating the accepted research-workflow changes. Preserve
intentionally supported public and dynamic interfaces with reviewed evidence.

## Acceptance

- [x] Add Vulture to the locked development environment and expose a repository audit command.
- [x] Audit package source, tests and scripts at confidence 60; review references, exports, protocols, fixtures, reflection and serialization before deleting candidates.
- [x] Delete confirmed dead implementations, obsolete callers, tests and configuration without deprecated aliases or blanket suppressions.
- [x] Record reasons for retained audit findings in separate verification evidence.
- [x] Pass affected tests, the integrated audit, distribution tests and `just check`, reporting incomplete checks.

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

_No open children._

`0 open; 0 terminal`

<!-- insitu:end -->

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

_No declared backlinks._

<!-- insitu:end -->
