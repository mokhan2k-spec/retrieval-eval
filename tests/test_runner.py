import json

import pytest

from retrieval_eval import runner as r


def ok(qid, text):
    return [f"{qid}-a", f"{qid}-b"]


Q = {"q1": "one", "q2": "two", "q3": "three"}


def test_run_arm_success():
    o = r.run_arm("x", Q, ok)
    assert o.arm == "x"
    assert o.failures == []
    assert o.rankings["q2"] == ["q2-a", "q2-b"]


def test_run_arm_passes_text_to_fn():
    seen = []
    r.run_arm("x", Q, lambda q, t: seen.append((q, t)) or ["d"])
    assert seen == [("q1", "one"), ("q2", "two"), ("q3", "three")]


def test_run_arm_records_failure_metadata():
    def fn(qid, text):
        if qid == "q2":
            raise ValueError("boom")
        return ["a"]

    o = r.run_arm("rerank", Q, fn)
    assert set(o.rankings) == {"q1", "q3"}
    assert o.failures == [r.Failure("rerank", "q2", "ValueError", "boom")]


def test_finalize_ok():
    o1, o2 = r.run_arm("a", Q, ok), r.run_arm("b", Q, ok)
    out = r.finalize([o1, o2], list(Q))
    assert set(out) == {"a", "b"} and set(out["a"]) == set(Q)
    assert out["b"]["q1"] == ["q1-a", "q1-b"]


def test_finalize_raises_on_recorded_failure():
    def bad(qid, text):
        raise RuntimeError("x")

    with pytest.raises(r.FailClosed) as ei:
        r.finalize([r.run_arm("a", Q, ok), r.run_arm("b", Q, bad)], list(Q))
    assert len(ei.value.failures) == 3
    assert {f.arm for f in ei.value.failures} == {"b"}


def test_finalize_one_bad_query_is_enough():
    def fn(qid, text):
        if qid == "q3":
            raise KeyError("k")
        return ["d"]

    with pytest.raises(r.FailClosed) as ei:
        r.finalize([r.run_arm("a", Q, fn)], list(Q))
    assert [f.query_id for f in ei.value.failures] == ["q3"]


def test_finalize_flags_missing_and_empty():
    o = r.ArmOutcome("a", rankings={"q1": ["x"], "q3": []})
    with pytest.raises(r.FailClosed) as ei:
        r.finalize([o], list(Q))
    kinds = {(f.query_id, f.error_type) for f in ei.value.failures}
    assert kinds == {("q2", "MissingRanking"), ("q3", "EmptyRanking")}


def test_finalize_does_not_double_report_recorded_failure():
    o = r.ArmOutcome("a", rankings={"q1": ["x"], "q3": ["y"]})
    o.failures.append(r.Failure("a", "q2", "E", "m"))
    with pytest.raises(r.FailClosed) as ei:
        r.finalize([o], list(Q))
    assert len(ei.value.failures) == 1


def test_fail_closed_message_counts_failures():
    e = r.FailClosed([r.Failure("a", "q", "E", "m")])
    assert "1" in str(e)


def test_clear_stale(tmp_path):
    p = tmp_path / "res.json"
    p.write_text("old")
    r.clear_stale(p)
    assert not p.exists()
    r.clear_stale(p)  # missing file is fine


def test_write_failures(tmp_path):
    f = r.Failure("a", "q1", "E", "m")
    path = r.write_failures(tmp_path / "run1", [f])
    assert path == tmp_path / "run1" / "failures.json"
    assert json.loads(path.read_text()) == {
        "failures": [{"arm": "a", "query_id": "q1", "error_type": "E", "message": "m"}]
    }


def test_atomic_write(tmp_path):
    p = tmp_path / "sub" / "res.json"
    r.write_results_atomic(p, {"a": 1})
    assert json.loads(p.read_text()) == {"a": 1}
    assert not list(p.parent.glob("*.tmp"))
    r.write_results_atomic(p, {"a": 2})
    assert json.loads(p.read_text()) == {"a": 2}
