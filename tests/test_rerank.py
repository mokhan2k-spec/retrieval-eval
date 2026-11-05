import pytest

from retrieval_eval.rerank import RerankError, rerank_candidates

IDS = ["a", "b", "c"]
TXT = ["ta", "tb", "tc"]


def test_orders_by_score_desc_with_stable_ties():
    out = rerank_candidates(lambda q, t: [0.1, 0.9, 0.1], "q", IDS, TXT)
    assert out == ["b", "a", "c"]


def test_exception_becomes_rerank_error():
    def boom(q, t):
        raise TimeoutError("slow")

    with pytest.raises(RerankError, match="TimeoutError"):
        rerank_candidates(boom, "q", IDS, TXT)


@pytest.mark.parametrize(
    "scores",
    [
        [1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, float("nan"), 2.0],
        [1.0, float("inf"), 2.0],
        [1.0, None, 2.0],
        ["x", 1.0, 2.0],
    ],
)
def test_bad_scores_raise(scores):
    with pytest.raises(RerankError):
        rerank_candidates(lambda q, t: scores, "q", IDS, TXT)


def test_mismatched_inputs_and_empty():
    with pytest.raises(RerankError):
        rerank_candidates(lambda q, t: [], "q", ["a"], [])
    with pytest.raises(RerankError):
        rerank_candidates(lambda q, t: [], "q", [], [])


def test_scorer_receives_query_and_texts():
    seen = {}

    def sc(q, t):
        seen["q"], seen["t"] = q, t
        return [1, 2, 3]

    rerank_candidates(sc, "hello", IDS, TXT)
    assert seen == {"q": "hello", "t": TXT}
