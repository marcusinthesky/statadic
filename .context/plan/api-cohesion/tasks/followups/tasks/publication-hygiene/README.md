---
type: task
title: Decide repository hygiene before publication
description: Owner decisions on private paths, history, scans and internal identifiers before the repository becomes public.
links:
  - .context/plan/api-ergonomics/tasks/research-workflow/tasks/publication-hold/README.md
status: backlog
---

# Decide repository hygiene before publication

## Goal

Ledger rows H05, H07 and H08 left these outside the API repair, and the
[publication hold](../../../../../api-ergonomics/tasks/research-workflow/tasks/publication-hold/README.md)
does not list them. Read at `00e7d8f`:

- Ten home-directory paths remain in three tracked verification records, under
  `public-surface`, `release-guarantees` and `research-release`.
- 23 tracked files name the private downstream repository.
- Internal task identifiers such as `t46` remain in 29 package source and
  documentation files.
- The branch has 96 commits and the remote has 14 branches. Publishing exposes
  all of them; a file deleted now stays in history.
- No secret, history or licence scan is recorded as executed.

Row H09 in the [dispositions](../../../../proposal-dispositions.md) owns this.

Owner: repository owner. No agent takes a public or irreversible step under this
task. See [deferred work](../../deferred.md).

## Acceptance

- [ ] Owner chooses existing or squashed history before the first public push.
- [ ] Owner decides whether historical records keep private paths and the downstream name, or are redacted.
- [ ] Run and record a secret and licence scan of the tree and the history to be published.
- [ ] Remove internal task identifiers from shipped source and documentation, or record why they stay.

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
