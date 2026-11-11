"""Open a native lab by its documented slug."""

import json
import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parents[1]


def main() -> None:
    labs = json.loads((APP / "labs.json").read_text())["labs"]
    slug = sys.argv[1]
    match = next((lab for lab in labs if lab["slug"] == slug), None)
    if match is None:
        raise SystemExit(
            f"Unknown lab {slug!r}; choose: " + ", ".join(lab["slug"] for lab in labs)
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "marimo",
            "edit",
            str(APP / "notebooks" / match["notebook"]),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
