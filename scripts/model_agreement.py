"""Agreement between two independent model labellers and the SciFact qrels.

Usage: python scripts/model_agreement.py

Reads labels/model-claude.csv and labels/model-codex.csv (item_id,label,reason),
each produced blind (no key.csv, no tools) from labels/model-prompt.txt followed
by labels/labelling-sheet.csv. Prints Cohen's kappa for model vs model and each
model vs the qrels in labels/key.csv. This is model agreement, not human
calibration: the human labels are still pending.
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from retrieval_eval.kappa import cohen_kappa  # noqa: E402


def read(name, col):
    with open(ROOT / "labels" / name, newline="", encoding="utf-8") as f:
        return {r["item_id"].strip(): int(r[col]) for r in csv.DictReader(f)}


def main():
    key = read("key.csv", "qrels_label")
    claude = read("model-claude.csv", "label")
    codex = read("model-codex.csv", "label")
    ids = sorted(key)
    if not (set(ids) == set(claude) == set(codex)):
        sys.exit("label files do not cover the same items")
    pairs = {
        "claude vs codex": (claude, codex),
        "claude vs qrels": (claude, key),
        "codex vs qrels": (codex, key),
    }
    for name, (a, b) in pairs.items():
        k = cohen_kappa([a[i] for i in ids], [b[i] for i in ids])
        agree = sum(a[i] == b[i] for i in ids)
        print(f"{name}: kappa={k:.3f} agree={agree}/{len(ids)}")
    both = [i for i in ids if claude[i] == codex[i]]
    k = cohen_kappa([claude[i] for i in both], [key[i] for i in both])
    match = sum(claude[i] == key[i] for i in both)
    print(f"where both models agree (n={len(both)}) vs qrels: kappa={k:.3f} agree={match}/{len(both)}")


if __name__ == "__main__":
    main()
