"""Run mutmut and fail if the mutation score is below a threshold.

score = (killed + timeout) / total mutants. Surviving mutants, mutants with
no covering test, and suspicious ones all count against the score.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=float, required=True, help="minimum score, percent")
    args = ap.parse_args()

    subprocess.run([sys.executable, "-m", "mutmut", "run"], check=False)
    subprocess.run([sys.executable, "-m", "mutmut", "export-cicd-stats"], check=True)
    stats = json.loads(Path("mutants/mutmut-cicd-stats.json").read_text())
    total = stats["total"]
    if total == 0:
        print("no mutants were generated", file=sys.stderr)
        return 1
    killed = stats.get("killed", 0) + stats.get("timeout", 0)
    score = 100.0 * killed / total
    print(f"mutation stats: {stats}")
    print(f"mutation score: {score:.1f}% ({killed}/{total}), required >= {args.min}%")
    return 0 if score >= args.min else 1


if __name__ == "__main__":
    sys.exit(main())
