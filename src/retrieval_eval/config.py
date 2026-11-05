"""Run configuration and its hash.

The config hash is stored in every results file. `report` refuses to render a
results file whose hash differs from the current config, so a table can never
describe a run made with different models, depths or seeds.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    dataset: str = "scifact"
    split: str = "test"
    dense_model: str = "BAAI/bge-small-en-v1.5"
    # Recommended retrieval instruction for the BGE v1.5 family; queries only.
    query_prefix: str = "Represent this sentence for searching relevant passages: "
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    pool: int = 100  # per-retriever candidate depth, and final ranking length
    depth: int = 40  # how many hybrid candidates the cross-encoder rescores
    rrf_k: int = 60
    ndcg_k: int = 10
    mrr_k: int = 10
    recall_ks: tuple[int, ...] = (10, 100)
    success_ks: tuple[int, ...] = (1, 5)
    seed: int = 42
    permutations: int = 10_000
    bootstrap: int = 10_000

    def to_dict(self) -> dict:
        return asdict(self)

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


ARMS = ("bm25", "dense", "hybrid", "hybrid+rerank")
# (baseline arm, challenger arm) pairs that get paired significance tests.
COMPARISONS = (("hybrid", "hybrid+rerank"), ("bm25", "hybrid"))
