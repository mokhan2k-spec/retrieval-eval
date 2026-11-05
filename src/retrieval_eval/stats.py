"""Paired significance tests.

Every comparison is paired by query and uses only COMPLETE pairs: queries for
which both arms have a finite value. `n_pairs` is always reported.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from scipy import stats as sps


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval; never leaves [0, 1]."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided binomial McNemar test on the discordant pairs only.

    b = A wrong, B right (B's wins); c = A right, B wrong (B's losses).
    Concordant pairs carry no information about which arm is better.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def _finite(x) -> bool:
    return x is not None and math.isfinite(x)


def complete_pairs(
    a: Mapping[str, float | None], b: Mapping[str, float | None]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Query ids present and finite in BOTH arms (sorted), with their values."""
    qids = sorted(q for q in a if q in b and _finite(a[q]) and _finite(b[q]))
    return (
        qids,
        np.array([a[q] for q in qids], dtype=float),
        np.array([b[q] for q in qids], dtype=float),
    )


def permutation_test(diffs: np.ndarray, n_perm: int, seed: int) -> float:
    """Two-sided paired randomization test on the mean difference.

    Under H0 the sign of each paired difference is exchangeable, so we flip
    signs at random and count how often the permuted |mean| reaches the
    observed |mean|. Add-one smoothing keeps p > 0.
    """
    if len(diffs) == 0:
        raise ValueError("no complete pairs")
    rng = np.random.default_rng(seed)
    observed = abs(diffs.mean())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(diffs)))
    perm_means = np.abs((signs * diffs).mean(axis=1))
    hits = int((perm_means >= observed - 1e-12).sum())
    return (hits + 1) / (n_perm + 1)


def wilcoxon_p(diffs: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p; 1.0 when every difference is zero."""
    if len(diffs) == 0:
        raise ValueError("no complete pairs")
    if not np.any(diffs != 0):
        return 1.0
    return float(sps.wilcoxon(diffs, alternative="two-sided").pvalue)


def bootstrap_ci(
    diffs: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean paired difference."""
    if len(diffs) == 0:
        raise ValueError("no complete pairs")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def compare(
    a_ndcg: Mapping[str, float | None],
    b_ndcg: Mapping[str, float | None],
    a_success1: Mapping[str, float | None],
    b_success1: Mapping[str, float | None],
    *,
    permutations: int,
    bootstrap: int,
    seed: int,
) -> dict:
    """Arm B vs arm A. Positive mean_diff means B is better."""
    qids, xa, xb = complete_pairs(a_ndcg, b_ndcg)
    diffs = xb - xa
    lo, hi = bootstrap_ci(diffs, bootstrap, seed)

    s_qids, sa, sb = complete_pairs(a_success1, b_success1)
    wins = int(((sa == 0) & (sb == 1)).sum())  # A wrong, B right
    losses = int(((sa == 1) & (sb == 0)).sum())  # A right, B wrong
    return {
        "n_pairs": len(qids),
        "mean_diff_ndcg": float(diffs.mean()),
        "ci95_ndcg": [lo, hi],
        "permutation_p_ndcg": permutation_test(diffs, permutations, seed),
        "wilcoxon_p_ndcg": wilcoxon_p(diffs),
        "mcnemar_success1": {
            "n_pairs": len(s_qids),
            "b_wins": wins,
            "c_losses": losses,
            "p": mcnemar(wins, losses),
        },
    }
