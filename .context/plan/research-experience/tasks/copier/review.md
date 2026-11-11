# Exordio alpha migration review

GitHub's release and tag API identify `v0.1.0-alpha` as the published prerelease at commit `4c641448957366aae760e39e87b9c14640f62645`. This commit has no parent. Exordio does not contain the currently recorded old template commit `81c54a7`; changing only `_src_path` breaks Copier's old-baseline lookup. The requested documentation site did not load in the web tool, so its availability is unverified. The release URL itself is confirmed by GitHub's API.

I rendered the old and alpha commits into `/tmp/statadic-copier-old` and `/tmp/statadic-copier-alpha`, respectively, using the saved STATADIC choices, Copier 9.16.0, and the old template's frozen uv project. The alpha local checkout exactly matches the published tag commit. Their whole rendered-tree diff consists of seven paths:

- `.copier-answers.yml`: source, commit and trailing blank line.
- `.github/workflows/ci.yml`: Nix installation, hosted runner disk capacity and Tectonic cache changes. STATADIC already has separate capacity, logging and artifact changes, so reconcile manually.
- `ARCHITECTURE.md`: replaces the prototype warning and old URL with the Exordio URL.
- `README.md`: template update guidance and URL; STATADIC moved its corresponding guidance to `CONTRIBUTING.md`.
- `infra/nix/retry-tex-setup.sh`: new release script.
- `src/latex/justfile` and `src/lean/justfile`: route setup Tectonic calls through the retry script; Lean has STATADIC-specific changes that must be preserved.

Concrete migration: review and reconcile every delta above with STATADIC's current project-owned changes, retaining intentional local behavior; compare `/tmp/statadic-copier-old` and `/tmp/statadic-copier-alpha` and the actual working tree. Add the alpha retry script and integrate the two setup invocations if adopting them. Reconcile CI deliberately, as its local behavior diverges. Update project guidance links to `https://marcusinthesky.github.io/exordio/` and `https://github.com/marcusinthesky/exordio/releases/tag/v0.1.0-alpha`; use `copier update --vcs-ref v0.1.0-alpha` for the explicit prerelease, not the obsolete `feat-copier` reference. Only after the complete delta is accepted (including intentional local deviations), set `_src_path: https://github.com/marcusinthesky/exordio.git` and `_commit: v0.1.0-alpha` in the answers as the reviewed migration baseline. Copier can then compare subsequent releases to that actual alpha render. Preserve the saved selections.

Validate after migration with setup where needed, explicit sync, and `just check`. A future Copier update should be trialed in a clean temporary checkout before touching the project; this review did not run update or modify the shared repository. The alpha release's own README refers to nonexistent-by-this-evidence `v0.1.0`; the requested explicit alpha tag is verified.

## Coordinator follow-up

The reviewed migration was applied after the frozen complete `just check` passed.
STATADIC's README and CI were retained, the setup retry helper and recipe changes
were adopted, and contributor guidance now uses the canonical source, site and
explicit alpha tag. Post-migration source/workflow/recipe checks and rebuilt
documentation, Lean, LaTeX, AgentSync and browser gates passed. A direct canonical
Git query confirmed the tag hash and an allowed HTTP request returned 200 for
the requested Pages URL. These later checks supersede the earlier sandbox DNS
limitation; no whole-project template overwrite was performed.

## Prepared migration and verification

`/tmp/statadic-copier-migration.patch` contains a three-way reconciliation against a snapshot of the current dirty and untracked authored tree. The README and CI conflicts were resolved by retaining STATADIC's existing versions; the CI release changes are therefore intentional project deviations. Contributor guidance contains the alpha target and both requested links. The release retry script and its two setup integrations were applied, preserving STATADIC's Lean additions. Patch whitespace validation passed.

An isolated clone of the migrated scratch project was pointed at an isolated copy of the exact published alpha commit with the matching tag. Copier 9.16.0 `update --defaults --vcs-ref v0.1.0-alpha` exited successfully, reported `Keeping template version 0.1.0a0`, and left the scratch worktree clean. This confirms a same-release no-op with local source; network cloning the canonical URL was not tested. No full STATADIC checks were run in this scratch preview.
