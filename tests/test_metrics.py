import math

import pytest

from retrieval_eval import metrics as m

QREL = {"d1": 1, "d3": 1, "d5": 1, "dz": 0}
RANK = ["d2", "d1", "d4", "d3", "d6", "d7", "d8", "d9", "d10", "d11", "d5"]


def test_ndcg_hand_computed():
    # DCG = 1/log2(3) + 1/log2(5); IDCG = 1 + 1/log2(3) + 1/log2(4)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3) + 0.5
    assert m.ndcg_at_k(RANK, QREL, 10) == pytest.approx(dcg / idcg)
    assert m.ndcg_at_k(RANK, QREL, 10) == pytest.approx(0.4982, abs=1e-4)


def test_ndcg_cutoff_excludes_deeper_hits():
    # d5 sits at rank 11, so it must not count at k=10 but does at k=11
    assert m.ndcg_at_k(RANK, QREL, 11) > m.ndcg_at_k(RANK, QREL, 10)


def test_ndcg_perfect_and_zero():
    assert m.ndcg_at_k(["d1", "d3", "d5"], QREL, 10) == pytest.approx(1.0)
    assert m.ndcg_at_k(["x", "y"], QREL, 10) == 0.0


def test_ndcg_graded():
    qrel = {"a": 2, "b": 1}
    got = m.ndcg_at_k(["b", "a"], qrel, 10)
    dcg = 1 + 2 / math.log2(3)
    idcg = 2 + 1 / math.log2(3)
    assert got == pytest.approx(dcg / idcg)


def test_ndcg_ideal_truncated_to_k():
    qrel = {"a": 1, "b": 1, "c": 1}
    assert m.ndcg_at_k(["a"], qrel, 1) == pytest.approx(1.0)


def test_recall_multiple_relevant():
    assert m.recall_at_k(RANK, QREL, 10) == pytest.approx(2 / 3)
    assert m.recall_at_k(RANK, QREL, 11) == pytest.approx(1.0)
    assert m.recall_at_k(RANK, QREL, 2) == pytest.approx(1 / 3)
    assert m.recall_at_k(RANK, QREL, 1) == 0.0


def test_grade_zero_is_not_relevant():
    qrel = {"a": 0, "b": 1}
    assert m.recall_at_k(["a"], qrel, 5) == 0.0
    assert m.success_at_k(["a"], qrel, 5) == 0.0
    assert m.mrr_at_k(["a", "b"], qrel, 5) == 0.5
    assert m.ndcg_at_k(["a", "b"], qrel, 5) == pytest.approx(1 / math.log2(3))


def test_success_differs_from_recall():
    assert m.success_at_k(RANK, QREL, 2) == 1.0  # one hit is enough
    assert m.recall_at_k(RANK, QREL, 2) < 1.0
    assert m.success_at_k(RANK, QREL, 1) == 0.0


def test_mrr():
    assert m.mrr_at_k(RANK, QREL, 10) == 0.5
    assert m.mrr_at_k(RANK, QREL, 1) == 0.0
    assert m.mrr_at_k(["d3"], QREL, 10) == 1.0


@pytest.mark.parametrize(
    "fn", [m.ndcg_at_k, m.recall_at_k, m.success_at_k, m.mrr_at_k]
)
def test_no_relevant_raises(fn):
    with pytest.raises(ValueError):
        fn(["a"], {"a": 0}, 5)


def test_query_metrics_names_and_values():
    out = m.query_metrics(
        RANK, QREL, ndcg_k=10, mrr_k=10, recall_ks=(10, 100), success_ks=(1, 5)
    )
    assert list(out) == [
        "ndcg@10", "recall@10", "recall@100", "mrr@10", "success@1", "success@5",
    ]
    assert out["recall@100"] == 1.0
    assert out["recall@10"] == pytest.approx(2 / 3)
    assert out["mrr@10"] == 0.5
    assert out["success@1"] == 0.0 and out["success@5"] == 1.0


def test_mean_metrics():
    pq = {"a": {"x": 1.0, "y": 0.0}, "b": {"x": 0.0, "y": 0.5}}
    assert m.mean_metrics(pq) == {"x": 0.5, "y": 0.25}
    with pytest.raises(ValueError):
        m.mean_metrics({})
