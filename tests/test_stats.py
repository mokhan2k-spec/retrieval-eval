import math

import numpy as np
import pytest

from retrieval_eval import stats as s


def test_wilson_values():
    lo, hi = s.wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)
    assert s.wilson(0, 0) == (0.0, 0.0)
    lo, hi = s.wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-12) and 0 < hi < 0.35
    lo, hi = s.wilson(10, 10)
    assert hi == pytest.approx(1.0) and lo > 0.65


def test_mcnemar_values():
    assert s.mcnemar(0, 0) == 1.0
    assert s.mcnemar(5, 0) == pytest.approx(2 / 32)
    assert s.mcnemar(0, 5) == pytest.approx(2 / 32)
    assert s.mcnemar(3, 7) == pytest.approx(2 * 176 / 1024)
    assert s.mcnemar(5, 5) == 1.0
    assert s.mcnemar(20, 0) == pytest.approx(2 / 2**20)


def test_complete_pairs_drops_missing_and_nan():
    a = {"q1": 1.0, "q2": 2.0, "q3": None, "q4": 4.0}
    b = {"q1": 1.5, "q2": float("nan"), "q3": 3.0, "q5": 9.0}
    qids, xa, xb = s.complete_pairs(a, b)
    assert qids == ["q1"]
    assert list(xa) == [1.0] and list(xb) == [1.5]


def test_complete_pairs_sorted_and_aligned():
    qids, xa, xb = s.complete_pairs({"b": 2.0, "a": 1.0}, {"a": 10.0, "b": 20.0})
    assert qids == ["a", "b"]
    assert list(xa) == [1.0, 2.0] and list(xb) == [10.0, 20.0]


def test_permutation_test_clear_effect_and_null():
    rng = np.random.default_rng(0)
    d = rng.normal(0.3, 0.1, 50)
    assert s.permutation_test(d, 2000, 1) < 0.01
    null = rng.normal(0.0, 1.0, 50)
    assert s.permutation_test(null, 2000, 1) > 0.05


def test_permutation_exact_small_case():
    # diffs (+1, +1): 4 sign patterns, 2 give |mean| = 1. p ~ 0.5 (add-one)
    p = s.permutation_test(np.array([1.0, 1.0]), 20000, 3)
    assert p == pytest.approx(0.5, abs=0.02)
    assert s.permutation_test(np.zeros(5), 100, 0) == 1.0


def test_permutation_deterministic_for_seed():
    d = np.array([0.1, -0.2, 0.3, 0.05, 0.4])
    assert s.permutation_test(d, 500, 7) == s.permutation_test(d, 500, 7)


def test_permutation_is_two_sided():
    d = np.array([0.5, 0.4, 0.6, 0.45, 0.55, 0.5])
    assert s.permutation_test(d, 2000, 1) == pytest.approx(
        s.permutation_test(-d, 2000, 1)
    )


def test_wilcoxon():
    assert s.wilcoxon_p(np.zeros(4)) == 1.0
    d = np.arange(1, 11, dtype=float)
    assert s.wilcoxon_p(d) == pytest.approx(2 / 1024)


def test_bootstrap_ci_contains_mean_and_is_ordered():
    d = np.random.default_rng(0).normal(0.2, 0.3, 100)
    lo, hi = s.bootstrap_ci(d, 1000, 5)
    assert lo < d.mean() < hi
    assert hi - lo < 0.3
    assert s.bootstrap_ci(d, 1000, 5) == (lo, hi)
    assert s.bootstrap_ci(np.full(10, 0.25), 100, 0) == (0.25, 0.25)


def test_bootstrap_alpha_widens_interval():
    d = np.random.default_rng(1).normal(0.0, 1.0, 60)
    lo95, hi95 = s.bootstrap_ci(d, 2000, 2, alpha=0.05)
    lo50, hi50 = s.bootstrap_ci(d, 2000, 2, alpha=0.5)
    assert lo95 < lo50 < hi50 < hi95


@pytest.mark.parametrize(
    "fn",
    [
        lambda: s.permutation_test(np.array([]), 10, 0),
        lambda: s.wilcoxon_p(np.array([])),
        lambda: s.bootstrap_ci(np.array([]), 10, 0),
    ],
)
def test_empty_raises(fn):
    with pytest.raises(ValueError):
        fn()


def test_compare_uses_only_complete_pairs():
    qs = [f"q{i}" for i in range(40)]
    a = {q: 0.2 for q in qs}
    b = {q: 0.6 for q in qs}
    a1 = {q: 0.0 for q in qs}
    b1 = {q: 1.0 for q in qs}
    # drop 5 from arm B, null out 3 more in arm A's success vector
    for q in qs[:5]:
        del b[q]
    for q in qs[5:8]:
        a1[q] = None
    out = s.compare(a, b, a1, b1, permutations=200, bootstrap=200, seed=1)
    assert out["n_pairs"] == 35
    assert out["mean_diff_ndcg"] == pytest.approx(0.4)
    mc = out["mcnemar_success1"]
    assert mc["n_pairs"] == 37  # 40 minus the 3 None entries
    assert (mc["b_wins"], mc["c_losses"]) == (37, 0)
    assert mc["p"] == pytest.approx(2 / 2**37)
    assert math.isfinite(out["wilcoxon_p_ndcg"])


def test_compare_no_pairs_raises():
    with pytest.raises(ValueError):
        s.compare({"a": 1.0}, {"b": 1.0}, {}, {}, permutations=10, bootstrap=10, seed=0)


def test_compare_mcnemar_direction_and_sign():
    a = {"q1": 0.0, "q2": 1.0, "q3": 0.0}
    b = {"q1": 1.0, "q2": 1.0, "q3": 1.0}
    out = s.compare(a, b, a, b, permutations=50, bootstrap=50, seed=0)
    assert out["mcnemar_success1"]["b_wins"] == 2
    assert out["mcnemar_success1"]["c_losses"] == 0
    assert out["mean_diff_ndcg"] > 0  # B - A
    lo, hi = out["ci95_ndcg"]
    assert lo <= out["mean_diff_ndcg"] <= hi
