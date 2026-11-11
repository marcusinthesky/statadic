"""Bounded evidence identity and CLI dispatch witnesses; no campaigns run."""

import contextlib
import importlib.metadata
import io
import json
from unittest.mock import patch

import jax
import numpy as np

from statadic_docs_labs.campaign_gate import check_execution_identity
from statadic_studies import campaign, cli, suite


def run():
    before = campaign.environment()
    version = importlib.metadata.version
    queried = []

    def changed_version(name):
        queried.append(name)
        return version(name) + "-review-change" if name == "optax" else version(name)

    with patch("importlib.metadata.version", side_effect=changed_version):
        after_dependency = campaign.environment()
    check_execution_identity(before, after_dependency)
    with jax.default_prng_impl("threefry2x32"):
        first_identity = campaign.environment()
        first = np.asarray(jax.random.normal(jax.random.key(123), (4,)))
    with jax.default_prng_impl("rbg"):
        second_identity = campaign.environment()
        second = np.asarray(jax.random.normal(jax.random.key(123), (4,)))
    check_execution_identity(first_identity, second_identity)
    arguments = [
        "studies",
        "suite",
        "/tmp/statadic-review-no-created-suite",
        "--repository",
        "/frozen/repository",
        "--wheel",
        "/frozen/statadic.whl",
        "--repo",
        "/overridden/repository",
        "--w",
        "/overridden/statadic.whl",
        "--study",
        "linear.ols-covariance",
    ]
    with (
        patch("sys.argv", arguments),
        patch.object(suite, "create") as create,
        patch.object(suite, "run", return_value={}),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        cli.main()
    return {
        "dependency_identity_unchanged": before == after_dependency,
        "optax_version_queried": "optax" in queried,
        "prng_identity_unchanged": first_identity == second_identity,
        "threefry_draws": first.tolist(),
        "rbg_draws": second.tolist(),
        "draws_equal": bool(np.array_equal(first, second)),
        "cli_dispatched_repository": str(create.call_args.args[1]),
        "cli_dispatched_wheel": str(create.call_args.kwargs["wheel"]),
        "limits": "Mocked metadata and orchestration; real effective PRNG contexts. No dependencies changed, preparation written, suite or release executed.",
    }


if __name__ == "__main__":
    with jax.enable_x64(new_val=True):
        print(json.dumps(run(), indent=2))
