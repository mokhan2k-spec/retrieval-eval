"""Build the human labelling sheet from a run's top-10 lists.

The sheet shows only the query and the document. The qrels label, the arm and
the ranks live in a separate key file, so a labeller cannot be primed by them.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from .data import Dataset

SHEET_COLUMNS = [
    "item_id", "query_id", "query", "doc_id", "doc_title", "doc_text",
    "human_label", "notes",
]
KEY_COLUMNS = [
    "item_id", "qrels_label", "arm", "rank_hybrid", "rank_hybrid+rerank", "category",
]

INSTRUCTIONS = """\
# Labelling instructions

You are judging whether a document is relevant to a query. There are no
right-or-wrong answers to hide: label what you honestly read.

The queries are scientific claims (BEIR SciFact). Each document is the title
and abstract of a paper.

Fill in the `human_label` column of `labelling-sheet.csv`:

- `1` (relevant): the abstract contains evidence that SUPPORTS or REFUTES the
  claim. It does not matter which way the evidence points, only that the
  abstract speaks directly to the claim.
- `0` (not relevant): the abstract is on a related topic but contains no
  evidence for or against the claim, or is unrelated.

Rules of thumb:

1. Judge only from the title and abstract shown. Do not search the web.
2. A shared keyword or topic is not enough. The abstract must bear on the
   specific claim.
3. If you genuinely cannot decide, pick the closer label and write why in
   `notes`.
4. Do not open `key.csv` until you have finished. It holds the dataset's own
   labels and which system returned each document, and would bias you.
5. Label at least 100 rows; all rows is better. Rows may be done in any order.

Save the file as CSV (UTF-8). Then run `make kappa`.
"""


def _rank(top: list[str], doc: str) -> int | None:
    return top.index(doc) + 1 if doc in top else None


def build_candidates(results: dict, data: Dataset) -> list[dict]:
    """Unique (query, doc) pairs from the hybrid and hybrid+rerank top-10s."""
    hy = results["arms"]["hybrid"]["top10"]
    rr = results["arms"]["hybrid+rerank"]["top10"]
    cands = []
    for qid in sorted(hy):
        docs = list(dict.fromkeys(hy[qid] + rr.get(qid, [])))
        for doc in docs:
            rh, rr_ = _rank(hy[qid], doc), _rank(rr.get(qid, []), doc)
            big = 10**6
            h, r = rh if rh is not None else big, rr_ if rr_ is not None else big
            if r < h:
                category = "promoted"
            elif r > h:
                category = "demoted"
            else:
                category = "unchanged"
            arm = "both" if rh and rr_ else ("hybrid" if rh else "hybrid+rerank")
            cands.append(
                {
                    "query_id": qid,
                    "doc_id": doc,
                    "qrels": 1 if data.qrels.get(qid, {}).get(doc, 0) > 0 else 0,
                    "arm": arm,
                    "rank_hybrid": rh or "",
                    "rank_hybrid+rerank": rr_ or "",
                    "category": category,
                }
            )
    return cands


# (qrels label, category) -> quota; the two halves each sum to 60.
QUOTAS = [
    (1, "promoted", 15), (1, "demoted", 15), (1, "unchanged", 30),
    (0, "promoted", 20), (0, "demoted", 20), (0, "unchanged", 20),
]


def sample(cands: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    half = n // 2
    chosen: list[dict] = []
    for label in (1, 0):
        groups = {c: [x for x in cands if x["qrels"] == label and x["category"] == c]
                  for c in ("promoted", "demoted", "unchanged")}
        for g in groups.values():
            g.sort(key=lambda x: (x["query_id"], x["doc_id"]))
            rng.shuffle(g)
        picked: list[dict] = []
        for lab, cat, quota in QUOTAS:
            if lab == label:
                scale = half / 60
                take = groups[cat][: round(quota * scale)]
                picked.extend(take)
                groups[cat] = groups[cat][len(take):]
        # backfill from whatever is left if a group was short
        rest = [x for g in groups.values() for x in g]
        rest.sort(key=lambda x: (x["query_id"], x["doc_id"]))
        rng.shuffle(rest)
        picked.extend(rest[: max(0, half - len(picked))])
        chosen.extend(picked)
    if len(chosen) < n:
        raise ValueError(f"only {len(chosen)} candidate pairs available, need {n}")
    chosen.sort(key=lambda x: (x["query_id"], x["doc_id"]))
    rng.shuffle(chosen)  # order must not reveal label, arm or query grouping
    return chosen


def write_sheet(
    results: dict, data: Dataset, out_dir: Path, n: int = 120, seed: int = 20260101
) -> tuple[Path, Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sample(build_candidates(results, data), n, seed)
    sheet, key, instr = (
        out_dir / "labelling-sheet.csv",
        out_dir / "key.csv",
        out_dir / "INSTRUCTIONS.md",
    )
    with open(sheet, "w", newline="", encoding="utf-8") as fs, open(
        key, "w", newline="", encoding="utf-8"
    ) as fk:
        ws = csv.DictWriter(fs, SHEET_COLUMNS)
        wk = csv.DictWriter(fk, KEY_COLUMNS)
        ws.writeheader()
        wk.writeheader()
        for i, r in enumerate(rows, start=1):
            item = f"item{i:03d}"
            d = data.corpus[r["doc_id"]]
            ws.writerow(
                {
                    "item_id": item,
                    "query_id": r["query_id"],
                    "query": data.queries[r["query_id"]],
                    "doc_id": r["doc_id"],
                    "doc_title": d["title"],
                    "doc_text": d["text"],
                    "human_label": "",
                    "notes": "",
                }
            )
            wk.writerow(
                {
                    "item_id": item,
                    "qrels_label": r["qrels"],
                    "arm": r["arm"],
                    "rank_hybrid": r["rank_hybrid"],
                    "rank_hybrid+rerank": r["rank_hybrid+rerank"],
                    "category": r["category"],
                }
            )
    instr.write_text(INSTRUCTIONS, encoding="utf-8")
    return sheet, key, instr
