---
title: Project context
description: Optional project plans and reference records, owned by this repository.
---

# Project context

- Planning is optional. Start a new plan with `just plan-new SLUG TITLE`.
- [References](reference/README.md): keep bibliographic metadata and source context together.
- [Retained evidence](evidence/index.md): source reviews preserved from the initial import.

Records are project-owned; Copier never creates or updates their contents.
Create one explicitly, edit its frontmatter and body, then run `just sync` and
`just check`. Supporting files may live beside the record that explains them.

Declare backlinks with `links`, using canonical repository-relative file paths.
Ordinary Markdown links can still point to sections and external URLs.
