"""End-to-end fail-closed behaviour with fake components."""

import json
import math

import pytest

from retrieval_eval import cli
from retrieval_eval.pipeline import execute
from retrieval_eval.runner import FailClosed


def _run(cfg, dataset, comps, tmp_path, **kw):
    return execute(
        cfg, dataset, comps,
        results_path=tmp_path / "results" / "fake.json",
        runs_dir=tmp_path / "runs", **kw,
    )


def _failures(tmp_path):
    files = list((tmp_path / "runs").glob("*/failures.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text())["failures"]


def test_success_writes_results_with_provenance(cfg, dataset, make_comps, tmp_path):
    res = _run(cfg, dataset, make_comps(), tmp_path)
    path = tmp_path / "results" / "fake.json"
    on_disk = json.loads(path.read_text())
    assert on_disk["run_id"] == res["run_id"]
    assert on_disk["dataset"]["zip_sha256"] == "0" * 64
    assert on_disk["dataset"]["n_queries"] == 160
    assert on_disk["config_hash"] == cfg.hash()
    assert on_disk["package_version"]
    assert set(on_disk["arms"]) == {"bm25", "dense", "hybrid", "hybrid+rerank"}
    sig = on_disk["significance"]["hybrid_vs_hybrid+rerank"]
    assert sig["n_pairs"] == 160
    assert not (tmp_path / "runs").exists()  # no failures.json on success


def test_rerank_error_on_one_query_fails_closed(cfg, dataset, make_comps, tmp_path):
    calls = {"n": 0}

    def flaky(query, texts):
        if query == "claim7":
            raise TimeoutError("reranker down")
        calls["n"] += 1
        return [float(i) for i in range(len(texts))]

    with pytest.raises(FailClosed):
        _run(cfg, dataset, make_comps(flaky), tmp_path)
    assert not (tmp_path / "results" / "fake.json").exists()
    fails = _failures(tmp_path)
    assert fails == [
        {
            "arm": "hybrid+rerank",
            "query_id": "q7",
            "error_type": "RerankError",
            "message": "TimeoutError: reranker down",
        }
    ]
    assert calls["n"] == 159  # the run kept going to report every failure


@pytest.mark.parametrize(
    "bad",
    [
        lambda texts: [math.nan] * len(texts),
        lambda texts: [1.0] * (len(texts) - 1),
        lambda texts: [1.0] * (len(texts) + 1),
    ],
    ids=["nan", "short", "long"],
)
def test_bad_reranker_output_fails_closed(cfg, dataset, make_comps, tmp_path, bad):
    def scorer(query, texts):
        return bad(texts) if query == "claim3" else [1.0] * len(texts)

    with pytest.raises(FailClosed):
        _run(cfg, dataset, make_comps(scorer), tmp_path)
    assert not (tmp_path / "results" / "fake.json").exists()
    fails = _failures(tmp_path)
    assert [(f["arm"], f["query_id"], f["error_type"]) for f in fails] == [
        ("hybrid+rerank", "q3", "RerankError")
    ]


def test_failing_first_stage_arm_is_recorded_and_cascades(
    cfg, dataset, make_comps, tmp_path
):
    comps = make_comps()
    real = comps.bm25

    def bm25(text, k):
        if text == "claim5":
            raise ConnectionError("index gone")
        return real(text, k)

    comps.bm25 = bm25
    with pytest.raises(FailClosed):
        _run(cfg, dataset, comps, tmp_path)
    arms = {(f["arm"], f["query_id"]) for f in _failures(tmp_path)}
    assert arms == {("bm25", "q5"), ("hybrid", "q5"), ("hybrid+rerank", "q5")}


def test_stale_results_deleted_at_start_even_when_run_fails(
    cfg, dataset, make_comps, tmp_path
):
    stale = tmp_path / "results" / "fake.json"
    stale.parent.mkdir()
    stale.write_text('{"stale": true}')

    def boom(query, texts):
        raise RuntimeError("x")

    with pytest.raises(FailClosed):
        _run(cfg, dataset, make_comps(boom), tmp_path)
    assert not stale.exists()


def test_stale_results_replaced_on_success(cfg, dataset, make_comps, tmp_path):
    stale = tmp_path / "results" / "fake.json"
    stale.parent.mkdir()
    stale.write_text('{"stale": true}')
    res = _run(cfg, dataset, make_comps(), tmp_path)
    assert json.loads(stale.read_text())["run_id"] == res["run_id"]


def test_fault_injection_hook(cfg, dataset, make_comps, tmp_path):
    with pytest.raises(FailClosed):
        _run(cfg, dataset, make_comps(), tmp_path, fault_query="q11")
    f = _failures(tmp_path)
    assert [(x["arm"], x["query_id"]) for x in f] == [("hybrid+rerank", "q11")]


def test_reranker_only_reorders_top_depth(cfg, dataset, make_comps, tmp_path):
    res = _run(cfg, dataset, make_comps(lambda q, t: list(range(len(t)))), tmp_path)
    # scorer reverses the head; the tail beyond depth is unchanged, so
    # recall@100 is identical for hybrid and hybrid+rerank
    a = res["arms"]["hybrid"]["metrics"]["recall@100"]
    b = res["arms"]["hybrid+rerank"]["metrics"]["recall@100"]
    assert a == pytest.approx(b)


def test_cli_run_fault_exits_nonzero_without_results(
    cfg, dataset, make_comps, tmp_path, monkeypatch, capsys
):
    import retrieval_eval.data as data_mod
    import retrieval_eval.pipeline as pipe

    monkeypatch.setattr(data_mod, "load_dataset", lambda *a, **k: dataset)
    monkeypatch.setattr(pipe, "build_components", lambda c, d: make_comps())
    monkeypatch.setenv("RETRIEVAL_EVAL_FAULT_QUERY", "q4")
    stale = tmp_path / "results" / "scifact.json"
    stale.parent.mkdir()
    stale.write_text("{}")
    code = cli.main(["run", "--root", str(tmp_path)])
    assert code != 0
    assert not stale.exists()
    assert len(list((tmp_path / "runs").glob("*/failures.json"))) == 1
    assert "FAIL-CLOSED" in capsys.readouterr().err
