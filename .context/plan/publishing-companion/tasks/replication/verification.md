# Independent reconstruction verification

The focused publishing check passed eight tests in both the default and
x64-enabled environments after independent review corrections. It includes the
fresh-wheel reconstruction and all required failure witnesses. Final integrated
checkpoint 2 passed and the coordinator marked this task done.

All commands ran through `devenv shell --` and the locked publishing project.
`just python::publishing::check` runs the test by default; no slow or manual lane
is needed. Its reconstruction subprocesses perform:

```text
uv build --offline --wheel --package statadic --out-dir <wheels>
uv build --offline --wheel --package statadic-evidence --out-dir <wheels>
uv build --offline --wheel --package statadic-publishing --out-dir <wheels>
uv venv --python <locked-interpreter> <venv-outside-checkout>
uv pip install --offline --python <venv>/bin/python <three-fresh-wheels>
<venv>/bin/python -I -m statadic_publishing build <extracted>/project <rebuilt>
<venv>/bin/python -I -m statadic_publishing verify <rebuilt> <extracted>/project
```

The initial archive is extracted with `statadic_evidence.extract_archive`. All three
installed import paths are asserted inside the new venv and outside the workspace.
Reconstruction runs outside the checkout and uses the staged TeX bundle,
bibliography, check script, lockfile and project from the archive. The rebuilt
PDF compiles and passes the unchanged log check; PDF byte identity is not claimed.

The test compares the complete generated-file set and SHA-256 of every member.
The corrected default-precision fixture produced these identical original and
rebuilt hashes:

| File | SHA-256 |
| --- | --- |
| `figure.tex` | `c7ffb03fcee279918ce3e749ea08f2e269ab13da822c15083af0be9b4d0198db` |
| `table.tex` | `fe5773db3038eed7d4a9db307d5c3095360600e6267636b272fd2e124366e2cc` |
| `methods.json` | `ef44ad4c7f71fc19c1920d58eff95f622736ddbec8daee45cfa0c942d36b2d5d` |
| `fit-snapshot/manifest.json` | `4acdbeda37559b87de85cdb5b57d29113e4ecce66dc980329f9500b3bb06f332` |
| `inference-snapshot/manifest.json` | `863a55a024985a16127ee9e9ae5bfcc6833dff58396582bc1d380d0f72672c90` |
| `fit-snapshot/arrays.bin` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `inference-snapshot/arrays.bin` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

Equality is asserted within each precision mode, not between precision modes.
The optional unwritten output remains `unavailable` and the build succeeds.
Rows marked unavailable retain their reason, invalid flag, status, and all five
missing inference fields through original, extracted and rebuilt snapshots.
Review caught the first fixture's missing unavailable flags; they were corrected
before the passing reruns.

Operations tests also exercise changed data, failed analysis with retained logs
and no manifest, missing required output, unresolved citation, tampering, schema
and path failures, and existing destinations. Repeated archives of one verified
bundle have identical bytes.

The final repository check and subsequent distribution gate passed; see the
[integrated results](../../../research-release/verification.md).
