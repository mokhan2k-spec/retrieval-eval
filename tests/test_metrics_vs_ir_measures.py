"""Cross-check our per-query metrics against ir_measures (pytrec_eval backend).

Our metrics are written out by hand so they can be mutation-tested; this test
pins them to the standard IR implementation on random graded qrels and rankings.
"""

import random

import ir_measures
import pytest
from ir_measures import RR, R, Success, nDCG

from retrieval_eval.metrics import query_metrics


def _case(rng: random.Random, qid: str):
    docs = [f"d{i}" for i in range(rng.randint(5, 150))]
    rel = rng.sample(docs, rng.randint(1, min(8, len(docs))))
    qrel = {d: rng.choice([1, 1, 2]) for d in rel}
    # A few judged non-relevant docs, as in real qrels.
    for d in rng.sample(docs, min(3, len(docs))):
        qrel.setdefault(d, 0)
    ranking = rng.sample(docs, len(docs))
    return qrel, ranking


@pytest.mark.parametrize("seed", range(5))
def test_matches_ir_measures(seed):
    rng = random.Random(seed)
    qrels, runs, ours = {}, {}, {}
    for n in range(40):
        qid = f"q{n}"
        qrel, ranking = _case(rng, qid)
        qrels[qid] = qrel
        # Strictly decreasing scores, so ir_measures sees exactly our order.
        runs[qid] = {d: float(len(ranking) - i) for i, d in enumerate(ranking)}
        ours[qid] = query_metrics(
            ranking, qrel, ndcg_k=10, mrr_k=10, recall_ks=(10, 100), success_ks=(1, 5)
        )
    measures = {
        nDCG @ 10: "ndcg@10",
        R @ 10: "recall@10",
        R @ 100: "recall@100",
        RR @ 10: "mrr@10",
        Success @ 1: "success@1",
        Success @ 5: "success@5",
    }
    seen = 0
    for m in ir_measures.iter_calc(list(measures), qrels, runs):
        assert ours[m.query_id][measures[m.measure]] == pytest.approx(m.value, abs=1e-9), (
            m.query_id,
            m.measure,
        )
        seen += 1
    assert seen == 40 * len(measures)
