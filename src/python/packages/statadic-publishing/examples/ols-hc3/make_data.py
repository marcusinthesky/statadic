"""Generate the synthetic, redistributable example data."""

import csv
import random
from pathlib import Path

rng = random.Random(17)
path = Path(__file__).parent / "data" / "synthetic.csv"
with path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["x", "y"])
    for index in range(40):
        x = index / 8
        writer.writerow(
            [f"{x:.3f}", f"{1 + 0.8 * x + rng.gauss(0, 0.25 + 0.04 * x):.8f}"]
        )
