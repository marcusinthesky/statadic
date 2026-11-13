"""Repository and installed-wheel command line entry point."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from statadic_studies.contracts import STUDIES


def main() -> None:
    """Run explicitly requested study operations without modifying source records."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    suite = commands.add_parser("suite", allow_abbrev=False)
    suite.add_argument("directory", type=Path)
    suite.add_argument("--repository", type=Path)
    suite.add_argument("--study", action="append", choices=sorted(STUDIES))
    suite.add_argument("--available", action="store_true")
    suite.add_argument("--resume", action="store_true")
    suite.add_argument("--status", action="store_true")
    suite.add_argument("--profile", choices=("smoke", "evaluation"))
    suite.add_argument("--trials", type=int)
    suite.add_argument("--draws", type=int)
    suite.add_argument("--seed", type=int)
    suite.add_argument("--wheel", type=Path)
    suite.add_argument("--x64", choices=("0", "1"))
    check = commands.add_parser("check")
    check.add_argument("--repository", type=Path)
    for action in ("run", "resume", "summarize", "verify", "export"):
        sub = commands.add_parser(action)
        sub.add_argument("campaign", type=Path)
        if action in {"run", "resume"}:
            sub.add_argument("--start", type=int, default=0)
            sub.add_argument("--stop", type=int)
        if action == "run":
            sub.add_argument("--study", choices=sorted(STUDIES), required=True)
            sub.add_argument(
                "--profile", choices=("smoke", "evaluation"), default="evaluation"
            )
            sub.add_argument("--trials", type=int)
            sub.add_argument("--seed", type=int, default=20261104)
            sub.add_argument("--wheel", type=Path)
            sub.add_argument("--draws", type=int, help="Freeze inner resampling effort")
        if action == "export":
            sub.add_argument("destination", type=Path)
    sub = commands.add_parser("merge")
    sub.add_argument("destination", type=Path)
    sub.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "suite":
        if args.x64 is not None:
            os.environ["JAX_ENABLE_X64"] = args.x64
        os.environ["JAX_PLATFORMS"] = "cpu"
        from statadic_studies import suite as orchestration

        if args.status:
            print(json.dumps(orchestration.status(args.directory), indent=2))
            return
        if not args.resume:
            if args.repository is None or bool(args.study) == args.available:
                parser.error(
                    "Suite creation requires --repository and "
                    "either --study or --available"
                )
            orchestration.create(
                args.directory,
                args.repository,
                sorted(STUDIES) if args.available else args.study,
                profile=args.profile or "evaluation",
                trials=args.trials,
                draws=args.draws,
                seed=args.seed if args.seed is not None else 20261104,
                wheel=args.wheel,
            )
        elif (
            args.study
            or args.available
            or args.repository
            or args.wheel
            or args.profile
            or args.trials is not None
            or args.draws is not None
            or args.seed is not None
        ):
            parser.error("Resume uses only frozen suite selections and wheel")
        print(json.dumps(orchestration.run(args.directory), indent=2))
        return
    from statadic_studies.analysis import export, summarize, verify_export
    from statadic_studies.campaign import create, execute, merge, verify

    if args.command == "check":
        if args.repository:
            from statadic_studies.catalogue import check_catalogue

            print(json.dumps(check_catalogue(args.repository), indent=2))
        else:
            print(
                json.dumps(
                    {key: value.record() for key, value in STUDIES.items()}, indent=2
                )
            )
    elif args.command in {"run", "resume"}:
        if args.command == "run":
            create(
                args.campaign,
                args.study,
                trials=args.trials
                if args.trials is not None
                else (32 if args.profile == "smoke" else 10000),
                seed=args.seed,
                profile=args.profile,
                wheel=args.wheel,
                draws=args.draws,
            )
        execute(args.campaign, start=args.start, stop=args.stop)
        print(json.dumps(summarize(args.campaign), indent=2))
    elif args.command == "merge":
        merge(args.destination, args.sources)
    elif args.command == "verify":
        if (args.campaign / "evidence-manifest.json").exists():
            print(
                json.dumps(
                    {"verified_campaign": verify_export(args.campaign)["campaign"]}
                )
            )
        else:
            print(json.dumps({"verified_trials": len(verify(args.campaign))}))
    elif args.command == "summarize":
        print(json.dumps(summarize(args.campaign), indent=2))
    elif args.command == "export":
        export(args.campaign, args.destination)


if __name__ == "__main__":
    main()
