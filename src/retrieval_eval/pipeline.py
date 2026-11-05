"""Orchestration: arms -> fail-closed gate -> metrics -> stats -> results file."""

from __future__ import annotations

import subprocess
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .config import ARMS, COMPARISONS, Config
from .data import Dataset
from .metrics import mean_metrics, query_metrics
from .rerank import Scorer, rerank_candidates
from .retrievers import rrf_fuse
from .runner import (
    FailClosed,
    clear_stale,
    finalize,
    run_arm,
    write_failures,
    write_results_atomic,
)
from .stats import compare, wilson

SearchFn = Callable[[str, int], list[str]]


@dataclass
class Components:
    bm25: SearchFn
    dense: SearchFn
    scorer: Scorer
    model_revisions: dict[str, str | None]


def git_sha(cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4()}"


def build_components(cfg: Config, data: Dataset) -> Components:
    from .rerank import CrossEncoderScorer
    from .retrievers import BM25Retriever, DenseRetriever

    ids = list(data.corpus)
    texts = [data.doc_text(d) for d in ids]
    bm25 = BM25Retriever(ids, texts)
    dense = DenseRetriever(ids, texts, cfg.dense_model, cfg.query_prefix)
    scorer = CrossEncoderScorer(cfg.rerank_model)
    revisions: dict[str, str | None] = {}
    try:
        from huggingface_hub import model_info

        for key, name in (("dense", cfg.dense_model), ("rerank", cfg.rerank_model)):
            revisions[key] = model_info(name).sha
    except Exception:
        revisions = {"dense": None, "rerank": None}
    return Components(bm25.search, dense.search, scorer, revisions)


def execute(
    cfg: Config,
    data: Dataset,
    comps: Components,
    *,
    results_path: Path,
    runs_dir: Path,
    fault_query: str | None = None,
    repo_dir: Path | None = None,
) -> dict:
    """Run all arms. On success write results atomically and return them.

    On any arm/query failure write runs/<run_id>/failures.json, write NO
    results file, and re-raise FailClosed.
    """
    results_path = Path(results_path)
    clear_stale(results_path)  # a previous results file is never reused
    run_id = new_run_id()
    started = datetime.now(timezone.utc).isoformat()
    queries = data.queries
    qids = list(queries)

    o_bm25 = run_arm("bm25", queries, lambda q, t: comps.bm25(t, cfg.pool))
    o_dense = run_arm("dense", queries, lambda q, t: comps.dense(t, cfg.pool))

    def hybrid_fn(qid: str, _text: str) -> list[str]:
        if qid not in o_bm25.rankings or qid not in o_dense.rankings:
            raise RuntimeError("upstream retriever failed for this query")
        return rrf_fuse(
            [o_bm25.rankings[qid], o_dense.rankings[qid]], cfg.rrf_k, cfg.pool
        )

    o_hybrid = run_arm("hybrid", queries, hybrid_fn)

    def rerank_fn(qid: str, text: str) -> list[str]:
        if qid not in o_hybrid.rankings:
            raise RuntimeError("upstream hybrid failed for this query")
        if fault_query is not None and qid == fault_query:
            from .rerank import RerankError

            raise RerankError("injected test fault (RETRIEVAL_EVAL_FAULT_QUERY)")
        hybrid = o_hybrid.rankings[qid]
        head, tail = hybrid[: cfg.depth], hybrid[cfg.depth :]
        reranked = rerank_candidates(
            comps.scorer, text, head, [data.doc_text(d) for d in head]
        )
        return reranked + tail

    o_rerank = run_arm("hybrid+rerank", queries, rerank_fn)

    outcomes = [o_bm25, o_dense, o_hybrid, o_rerank]
    run_dir = Path(runs_dir) / run_id
    try:
        rankings = finalize(outcomes, qids)
    except FailClosed as e:
        write_failures(run_dir, e.failures)
        raise

    results = build_results(cfg, data, rankings, run_id, started, comps, repo_dir)
    write_results_atomic(results_path, results)
    return results


def build_results(
    cfg: Config,
    data: Dataset,
    rankings: dict[str, dict[str, list[str]]],
    run_id: str,
    started: str,
    comps: Components,
    repo_dir: Path | None = None,
) -> dict:
    arms: dict[str, dict] = {}
    for arm in ARMS:
        per_query = {
            qid: query_metrics(
                ranking,
                data.qrels[qid],
                ndcg_k=cfg.ndcg_k,
                mrr_k=cfg.mrr_k,
                recall_ks=cfg.recall_ks,
                success_ks=cfg.success_ks,
            )
            for qid, ranking in rankings[arm].items()
        }
        metrics = mean_metrics(per_query)
        n = len(per_query)
        k1 = sum(q["success@1"] for q in per_query.values())
        lo, hi = wilson(int(round(k1)), n)
        arms[arm] = {
            "metrics": metrics,
            "success@1_wilson95": [lo, hi],
            "per_query": per_query,
            "top10": {q: r[:10] for q, r in rankings[arm].items()},
        }

    nd = f"ndcg@{cfg.ndcg_k}"
    significance = {}
    for a, b in COMPARISONS:
        significance[f"{a}_vs_{b}"] = compare(
            {q: m[nd] for q, m in arms[a]["per_query"].items()},
            {q: m[nd] for q, m in arms[b]["per_query"].items()},
            {q: m["success@1"] for q, m in arms[a]["per_query"].items()},
            {q: m["success@1"] for q, m in arms[b]["per_query"].items()},
            permutations=cfg.permutations,
            bootstrap=cfg.bootstrap,
            seed=cfg.seed,
        )

    return {
        "schema": 1,
        "run_id": run_id,
        "started_utc": started,
        "dataset": {
            "name": data.name,
            "split": cfg.split,
            "zip_sha256": data.zip_sha256,
            "n_queries": len(data.queries),
            "n_docs": len(data.corpus),
        },
        "config": cfg.to_dict(),
        "config_hash": cfg.hash(),
        "package_version": __version__,
        "git_sha": git_sha(repo_dir),
        "model_revisions": comps.model_revisions,
        "arms": arms,
        "significance": significance,
    }
