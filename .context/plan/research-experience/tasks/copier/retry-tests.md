# Exordio alpha retry and documentation checks

The release's `retry-tex-setup.sh` passed four isolated tests. The harness uses the pinned repository Bash and ripgrep, a fake Tectonic command, and fake sleep, so no network or delays are involved.

| Scenario | Result | Exit | Attempts | Sleeps |
| --- | --- | ---: | ---: | ---: |
| First attempt succeeds | Pass | 0 | 1 | 0 |
| Nontransient failure | Pass | 23 | 1 | 0 |
| HTTP 429, then success | Pass | 0 | 2 | 1 |
| Persistent HTTP 503 | Pass | 42 | 4 | 3 |

Harness and captured output are under `/tmp/statadic-retry-test/`. Each scenario asserted the exit, attempts and sleep count.

The pinned curl attempt to `https://marcusinthesky.github.io/exordio/` failed with DNS resolution error 6 in this sandbox; it did not establish whether the site serves content. GitHub's [Pages workflow](https://github.com/marcusinthesky/exordio/blob/main/.github/workflows/pages.yml) builds and deploys the site, and GitHub reports a [successful Pages workflow run](https://github.com/marcusinthesky/exordio/actions/runs/41472729022). The connector rejected its Pages metadata endpoint, so the exact live URL and HTTP response remain unverified by this check.
