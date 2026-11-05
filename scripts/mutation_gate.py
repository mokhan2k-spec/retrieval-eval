"""Run mutmut and fail if the mutation score is below a threshold.

score = (killed + timeout) / total mutants. Surviving mutants, mutants with
no covering test, unchecked and suspicious ones all count against the score.
If mutmut itself fails, no results are read and the gate fails.
"""

import argparse
import subprocess
import sys
from collections import Counter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=float, required=True, help="minimum score, percent")
    args = ap.parse_args()

    run = subprocess.run([sys.executable, "-m", "mutmut", "run"], check=False)
    if run.returncode != 0:
        print(f"mutmut run exited {run.returncode}", file=sys.stderr)
        return 1
    # mutmut 3.3 has no stats export; `results --all true` prints "    <mutant>: <status>".
    out = subprocess.run(
        [sys.executable, "-m", "mutmut", "results", "--all", "true"],
        check=True, capture_output=True, text=True,
    ).stdout
    stats = Counter(
        line.rsplit(": ", 1)[1].strip()
        for line in out.splitlines()
        if line.startswith("    ") and ": " in line
    )
    total = sum(stats.values())
    if total == 0:
        print("no mutants were generated", file=sys.stderr)
        return 1
    killed = stats["killed"] + stats["timeout"]
    score = 100.0 * killed / total
    print(f"mutation stats: {dict(stats)}")
    print(f"mutation score: {score:.1f}% ({killed}/{total}), required >= {args.min}%")
    return 0 if score >= args.min else 1


if __name__ == "__main__":
    sys.exit(main())
