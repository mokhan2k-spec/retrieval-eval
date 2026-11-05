import csv

import pytest

from retrieval_eval import cli
from retrieval_eval.kappa import cohen_kappa, human_labels, report


def test_perfect_agreement():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)


def test_perfect_disagreement_is_minus_one():
    assert cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == pytest.approx(-1.0)


def test_chance_agreement_is_zero():
    # po = 0.5, pe = 0.5
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)


def test_textbook_example():
    # Wikipedia: 50 proposals; yes/yes 20, yes/no 5, no/yes 10, no/no 15
    a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_constant_identical_raters():
    assert cohen_kappa([1, 1, 1], [1, 1, 1]) == 1.0


def test_one_rater_constant_other_not():
    assert cohen_kappa([1, 1, 1, 1], [1, 0, 1, 1]) == pytest.approx(0.0)


def test_errors():
    with pytest.raises(ValueError):
        cohen_kappa([1], [1, 0])
    with pytest.raises(ValueError):
        cohen_kappa([], [])


def test_matches_sklearn_if_installed():
    sk = pytest.importorskip("sklearn.metrics")
    import random

    rng = random.Random(0)
    a = [rng.randint(0, 1) for _ in range(200)]
    b = [x if rng.random() < 0.8 else 1 - x for x in a]
    assert cohen_kappa(a, b) == pytest.approx(sk.cohen_kappa_score(a, b))


def _write_labels(d, n_items, n_human, judge=True):
    d.mkdir(exist_ok=True)
    with open(d / "labelling-sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "human_label"])
        for i in range(n_items):
            w.writerow([f"i{i}", (i % 2) if i < n_human else ""])
    with open(d / "key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "qrels_label"])
        for i in range(n_items):
            w.writerow([f"i{i}", i % 2])
    if judge:
        with open(d / "judge-single.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "judge_label"])
            for i in range(n_items):
                w.writerow([f"i{i}", (i + (i % 4 == 0)) % 2])


def test_report_refuses_below_100(tmp_path):
    _write_labels(tmp_path / "l", 120, 99)
    code, text = report(tmp_path / "l")
    assert code == 1
    assert "99" in text and "1 more" in text


def test_report_ok_at_100(tmp_path):
    _write_labels(tmp_path / "l", 120, 100)
    code, text = report(tmp_path / "l")
    assert code == 0
    assert "human vs qrels: kappa=1.000 (n=100)" in text
    assert "human vs judge-single: kappa=" in text
    assert "judge-adversarial: judge file missing" in text


def test_human_labels_rejects_garbage(tmp_path):
    _write_labels(tmp_path / "l", 5, 5)
    p = tmp_path / "l" / "labelling-sheet.csv"
    p.write_text("item_id,human_label\ni0,yes\n", encoding="utf-8")
    with pytest.raises(ValueError):
        human_labels(p)


def test_cli_kappa_exit_codes(tmp_path):
    _write_labels(tmp_path / "labels", 120, 10)
    assert cli.main(["kappa", "--root", str(tmp_path)]) == 1
    _write_labels(tmp_path / "labels", 120, 110)
    assert cli.main(["kappa", "--root", str(tmp_path)]) == 0
