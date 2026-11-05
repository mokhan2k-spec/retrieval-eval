"""First-stage retrievers and rank fusion."""

from __future__ import annotations

from collections.abc import Sequence


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int, limit: int) -> list[str]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank).

    Ranks are 1-based. Ties are broken by first appearance, so the result is
    deterministic.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(doc, len(first_seen))
    ordered = sorted(scores, key=lambda d: (-scores[d], first_seen[d]))
    return ordered[:limit]


class BM25Retriever:
    def __init__(self, doc_ids: Sequence[str], texts: Sequence[str]):
        import bm25s
        import Stemmer

        self._bm25s = bm25s
        self._stemmer = Stemmer.Stemmer("english")
        self.doc_ids = list(doc_ids)
        tokens = bm25s.tokenize(
            list(texts), stopwords="en", stemmer=self._stemmer, show_progress=False
        )
        self.index = bm25s.BM25()
        self.index.index(tokens, show_progress=False)

    def search(self, query: str, k: int) -> list[str]:
        q = self._bm25s.tokenize(
            query,
            stopwords="en",
            stemmer=self._stemmer,
            return_ids=False,
            show_progress=False,
        )
        k = min(k, len(self.doc_ids))
        docs, _ = self.index.retrieve(q, k=k, show_progress=False)
        return [self.doc_ids[i] for i in docs[0]]


class DenseRetriever:
    def __init__(
        self,
        doc_ids: Sequence[str],
        texts: Sequence[str],
        model_name: str,
        query_prefix: str,
    ):
        from sentence_transformers import SentenceTransformer

        self.doc_ids = list(doc_ids)
        self.query_prefix = query_prefix
        self.model = SentenceTransformer(model_name, device="cpu")
        self.emb = self.model.encode(
            list(texts),
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def search(self, query: str, k: int) -> list[str]:
        import numpy as np

        q = self.model.encode(
            [self.query_prefix + query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        sims = self.emb @ q  # cosine, since both sides are unit-normalised
        k = min(k, len(self.doc_ids))
        # stable sort so equal scores resolve by corpus order
        order = np.argsort(-sims, kind="stable")[:k]
        return [self.doc_ids[i] for i in order]
