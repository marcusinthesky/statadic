"""Run the registered finite-MMD study as a bounded teaching experiment."""

from pathlib import Path
from tempfile import TemporaryDirectory

from statadic_studies.analysis import summarize
from statadic_studies.campaign import create, execute, verify


def run_study(trials: int = 20, seed: int = 123) -> dict:
    """Return the frozen design and actual summary; do not retain trial records."""
    if type(trials) is not int or trials not in (20, 100, 1000):
        raise ValueError("Choose 20, 100 or 1000 trials per scenario")
    if type(seed) is not int or not 0 <= seed <= 10000:
        raise ValueError("Seed must be an integer from 0 to 10000")
    with TemporaryDirectory(prefix="statadic-mmd-teaching-") as directory:
        root = Path(directory) / "campaign"
        manifest = create(
            root, "mmd.finite-bias", trials=trials, seed=seed, profile="smoke"
        )
        execute(root)
        summary = summarize(root)
        # Read the population truths produced by the registered adapter, rather
        # than maintaining another version of its DGP or target in the lab.
        truths = {}
        for trial in verify(root):
            for result in trial["outcomes"].values():
                if result["valid"]:
                    scenario = trial["scenario"]
                    truth = result["truth"]
                    if scenario in truths and truths[scenario] != truth:
                        raise ValueError(
                            "Inconsistent population target in study records"
                        )
                    truths[scenario] = truth
        return {
            "purpose": "teaching only; no qualification; trials not retained",
            "manifest": manifest,
            "summary": summary,
            "population_truth": truths,
        }
