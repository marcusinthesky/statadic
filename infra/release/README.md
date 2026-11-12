# Releases

The [release workflow](../../.github/workflows/release.yml) follows Insitu's
tag-driven release process. Only the public `statadic` package is published.
Internal applications and companion packages keep their separate versions.

The documentation site has a separate [GitHub Pages workflow](../../.github/workflows/pages.yml).
It builds on pull requests and publishes successful builds from `main`.
See the [docs application guide](../../src/python/apps/docs/README.md) for Pages setup.

Before publishing, configure these account settings:

1. Create the GitHub environment `release` in `marcusinthesky/statadic`.
   Restrict deployments to release tags and configure required reviewers if desired.
2. Register a [PyPI Trusted Publisher](https://docs.pypi.org/trusted-publishers/)
   for project `statadic`, owner `marcusinthesky`, repository `statadic`,
   workflow filename `release.yml`, environment `release`. Use a pending publisher
   if the project has not been created. No PyPI API token is needed.

For each release:

1. Update `src/python/packages/statadic/pyproject.toml`, the changelog and the
   matching `statadic` version in `src/python/uv.lock`. Use the locked uv tooling
   to refresh the lock. Review and refresh the Lean public API inventory with
   `just lean::inventory-refresh` so its package version agrees.
2. Open a pull request. Changes to release configuration or version inputs run
   the complete preflight as a dry run, without publishing. The normal CI gates
   still apply.
3. After merging, push an exact package-version tag, for example `v0.5.0a1`.
   Supported prerelease suffixes are `aN`, `bN` and `rcN`; they are marked as
   prereleases on GitHub. Versions are explicit; commits do not trigger bumps.

The version job rejects mismatched package, lock, registry and tag versions.
The reusable [preflight](../../.github/workflows/release-validation.yml) runs
on Linux x86-64 and ARM64 and verifies its retained archive. Both runners must
succeed before publication. PyPI receives the exact x86-64 preflight wheel and
sdist, already exercised by the installed-distribution checks. The wheel is
platform independent; dependencies such as JAX supply their own platform wheels.

After PyPI succeeds, the workflow creates a GitHub release with those same
distributions and `SHA256SUMS`. Detailed preflight evidence remains in each
workflow run's `release-evidence-*` artifacts. A pull request or a manual preflight
never publishes. Hosted preflight does not establish full scientific qualification;
retain the separate `just release-check` evidence before making that claim.

Run the lightweight version gates locally in the prepared environment:

```bash
uv run --project src/python --offline --frozen --package statadic-docs python infra/release/check_version.py --tag v0.5.0a1
uv run --project src/python --offline --frozen --package statadic-docs python -m unittest discover -s infra/release -p 'test_*.py' -v
```
