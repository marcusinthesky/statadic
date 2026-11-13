"""Build, verify, or archive a local publishing project."""

import argparse
from pathlib import Path

from .operations import archive, build, verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("build")
    command.add_argument("project", type=Path)
    command.add_argument("output", type=Path)
    command = commands.add_parser("verify")
    command.add_argument("bundle", type=Path)
    command.add_argument("project", type=Path)
    command = commands.add_parser("archive")
    command.add_argument("bundle", type=Path)
    command.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        print(build(args.project, args.output).path)
    elif args.command == "verify":
        result = verify(args.bundle, args.project)
        if not result.valid:
            parser.exit(1, "\n".join(result.failures) + "\n")
        print("verified")
    else:
        print(archive(args.bundle, args.destination))


if __name__ == "__main__":
    main()
