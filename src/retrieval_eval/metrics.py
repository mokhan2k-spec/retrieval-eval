"""Ranking metrics, computed per query.

A document is relevant when its qrels grade is > 0. `recall@k` is the standard
IR quantity (fraction of the relevant documents found in the top k).
`success@k` is the weaker "at least one relevant document in the top k".
They are different numbers and are never mixed up in this package.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _relevant(qrel: Mapping[str, int]) -> set[str]:
    return {d for d, g in qrel.items() if g > 0}


def ndcg_at_k(ranking: Sequence[str], qrel: Mapping[str, int], k: int) -> float:
    """nDCG@k with linear gain (gain = grade), log2(rank + 1) discount."""
    dcg = sum(
        qrel.get(doc, 0) / math.log2(i + 2)
        for i, doc in enumerate(ranking[:k])
        if qrel.get(doc, 0) > 0
    )
    ideal = sorted((g for g in qrel.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    if idcg == 0:
        raise ValueError("query has no relevant documents")
    return dcg / idcg


def recall_at_k(ranking: Sequence[str], qrel: Mapping[str, int], k: int) -> float:
    rel = _relevant(qrel)
    if not rel:
        raise ValueError("query has no relevant documents")
    return len(rel.intersection(ranking[:k])) / len(rel)


def success_at_k(ranking: Sequence[str], qrel: Mapping[str, int], k: int) -> float:
    rel = _relevant(qrel)
    if not rel:
        raise ValueError("query has no relevant documents")
    return 1.0 if rel.intersection(ranking[:k]) else 0.0


def mrr_at_k(ranking: Sequence[str], qrel: Mapping[str, int], k: int) -> float:
    rel = _relevant(qrel)
    if not rel:
        raise ValueError("query has no relevant documents")
    for i, doc in enumerate(ranking[:k]):
        if doc in rel:
            return 1.0 / (i + 1)
    return 0.0


def query_metrics(
    ranking: Sequence[str],
    qrel: Mapping[str, int],
    *,
    ndcg_k: int,
    mrr_k: int,
    recall_ks: Sequence[int],
    success_ks: Sequence[int],
) -> dict[str, float]:
    out = {f"ndcg@{ndcg_k}": ndcg_at_k(ranking, qrel, ndcg_k)}
    for k in recall_ks:
        out[f"recall@{k}"] = recall_at_k(ranking, qrel, k)
    out[f"mrr@{mrr_k}"] = mrr_at_k(ranking, qrel, mrr_k)
    for k in success_ks:
        out[f"success@{k}"] = success_at_k(ranking, qrel, k)
    return out


def mean_metrics(per_query: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    if not per_query:
        raise ValueError("no queries")
    names = next(iter(per_query.values())).keys()
    n = len(per_query)
    return {m: sum(q[m] for q in per_query.values()) / n for m in names}
