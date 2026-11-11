---
title: Planning
description: Project-owned plans and nested tasks with honest completion and local views.
---

# Planning

Create a plan, then its first task:

```bash
just plan-new project "Project plan"
just task-new .context/plan/project first-step "Make the first useful change"
just sync
```

The [root roadmap](../../ROADMAP.md) lists plans. Each plan or task shows only
its immediate open children and counts terminal children. A deeper task lives
in its parent's `tasks/SLUG/README.md`; its path supplies identity and parentage.
Changing status never moves a file or breaks its links.

Edit `status` in frontmatter: `backlog`, `in-progress`, `blocked`, `done` or
`closed`. `done` requires at least one nonempty active acceptance criterion,
all checked. Fenced examples and struck-through criteria do not count.
`closed` stops work without claiming completion and requires `resolution.reason`.
Terminal parents cannot retain open children. Reopen a parent before adding work.

Every work item keeps one generated children region, including leaves. Edit
outside the markers; `just sync` refreshes views, while `just check` reports
invalid records or drift without repairing them. Put longer supporting detail
in linked files rather than growing the 1,500-token entry point.
