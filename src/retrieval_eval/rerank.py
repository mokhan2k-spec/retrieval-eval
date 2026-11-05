"""Cross-encoder reranking that never fails open.

`rerank_candidates` either returns a valid reordering or raises `RerankError`.
It never hands back the input order as a fallback: a silent fallback would
make a degraded run look like a real reranker result.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

# A scorer maps (query, [passage, ...]) to one relevance score per passage.
Scorer = Callable[[str, Sequence[str]], Sequence[float]]


class RerankError(Exception):
    """The reranker errored or returned unusable scores."""


def rerank_candidates(
    scorer: Scorer,
    query: str,
    doc_ids: Sequence[str],
    texts: Sequence[str],
) -> list[str]:
    """Return doc_ids reordered by descending score (ties keep input order)."""
    if len(doc_ids) != len(texts):
        raise RerankError("doc_ids and texts differ in length")
    if not doc_ids:
        raise RerankError("no candidates to rerank")
    try:
        scores = list(scorer(query, list(texts)))
    except Exception as e:  # any scorer failure is a RerankError, never a fallback
        raise RerankError(f"{type(e).__name__}: {e}") from e
    if len(scores) != len(doc_ids):
        raise RerankError(
            f"scorer returned {len(scores)} scores for {len(doc_ids)} candidates"
        )
    for s in scores:
        try:
            ok = math.isfinite(float(s))
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise RerankError(f"scorer returned a non-finite score: {s!r}")
    order = sorted(range(len(doc_ids)), key=lambda i: (-float(scores[i]), i))
    return [doc_ids[i] for i in order]


class CrossEncoderScorer:
    """sentence-transformers CrossEncoder, loaded lazily on CPU."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device="cpu")

    def __call__(self, query: str, texts: Sequence[str]) -> list[float]:
        scores = self.model.predict(
            [(query, t) for t in texts], show_progress_bar=False
        )
        return [float(s) for s in scores]
