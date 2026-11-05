"""Cohen's kappa and the human-vs-judge-vs-qrels agreement report."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

MIN_HUMAN_LABELS = 100


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """kappa = (po - pe) / (1 - pe).

    If both raters use a single identical label throughout, pe == 1 and the
    ratio is 0/0; we return 1.0 (perfect agreement) in that case.
    """
    if len(a) != len(b):
        raise ValueError("label sequences differ in length")
    n = len(a)
    if n == 0:
        raise ValueError("no labels")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def _read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def human_labels(sheet: Path) -> dict[str, int]:
    out = {}
    for r in _read(sheet):
        v = r["human_label"].strip()
        if v == "":
            continue
        if v not in ("0", "1"):
            raise ValueError(f"{r['item_id']}: human_label must be 0 or 1, got {v!r}")
        out[r["item_id"]] = int(v)
    return out


def report(labels_dir: Path) -> tuple[int, str]:
    """Return (exit_code, text). Refuses with < MIN_HUMAN_LABELS human labels."""
    labels_dir = Path(labels_dir)
    sheet = labels_dir / "labelling-sheet.csv"
    total = len(_read(sheet))
    human = human_labels(sheet)
    if len(human) < MIN_HUMAN_LABELS:
        missing = MIN_HUMAN_LABELS - len(human)
        return 1, (
            f"refusing: {len(human)} human labels filled in of {total} rows; "
            f"{missing} more needed (minimum {MIN_HUMAN_LABELS})"
        )
    lines = [f"human labels: {len(human)}"]
    key = {r["item_id"]: int(r["qrels_label"]) for r in _read(labels_dir / "key.csv")}
    pairs = {"human vs qrels": key}
    for mode in ("single", "adversarial"):
        p = labels_dir / f"judge-{mode}.csv"
        if p.exists():
            pairs[f"human vs judge-{mode}"] = {
                r["item_id"]: int(r["judge_label"]) for r in _read(p)
            }
        else:
            lines.append(f"human vs judge-{mode}: judge file missing ({p.name})")
    for name, other in pairs.items():
        ids = sorted(i for i in human if i in other)
        k = cohen_kappa([human[i] for i in ids], [other[i] for i in ids])
        lines.append(f"{name}: kappa={k:.3f} (n={len(ids)})")
    return 0, "\n".join(lines)
