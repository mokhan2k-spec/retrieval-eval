from retrieval_eval.retrievers import rrf_fuse


def test_rrf_hand_computed():
    out = rrf_fuse([["a", "b", "c"], ["b", "c", "d"]], k=60, limit=10)
    # b: 1/62+1/61, c: 1/63+1/62, a: 1/61, d: 1/63
    assert out == ["b", "c", "a", "d"]


def test_rrf_limit_and_ties():
    assert rrf_fuse([["x", "y"], ["y", "x"]], k=60, limit=1) == ["x"]
    assert rrf_fuse([["a"]], k=60, limit=0) == []


def test_rrf_identical_lists_keep_order():
    assert rrf_fuse([["a", "b"], ["a", "b"]], k=1, limit=2) == ["a", "b"]
