import json
from dataclasses import replace

import pytest

from retrieval_eval import cli
from retrieval_eval.pipeline import execute
from retrieval_eval.report import (
    END, START, ReportError, load_results, render_block, update_readme,
)


@pytest.fixture
def results_file(cfg, dataset, make_comps, tmp_path):
    path = tmp_path / "res.json"
    execute(cfg, dataset, make_comps(), results_path=path, runs_dir=tmp_path / "runs")
    return path


def test_load_ok(cfg, results_file):
    assert load_results(results_file, cfg)["config_hash"] == cfg.hash()


def test_refuses_on_config_hash_mismatch(cfg, results_file):
    other = replace(cfg, depth=20)
    with pytest.raises(ReportError, match="hash mismatch"):
        load_results(results_file, other)


def test_refuses_missing_file(cfg, tmp_path):
    with pytest.raises(ReportError, match="not found"):
        load_results(tmp_path / "nope.json", cfg)


def test_config_hash_covers_all_fields(cfg):
    for change in (
        {"seed": 1}, {"dense_model": "x"}, {"rerank_model": "x"}, {"depth": 1},
        {"recall_ks": (5,)}, {"rrf_k": 1}, {"dataset": "fiqa"},
    ):
        assert replace(cfg, **change).hash() != cfg.hash()
    assert replace(cfg).hash() == cfg.hash()


def test_render_block_contents(cfg, results_file):
    block = render_block(load_results(results_file, cfg))
    assert "| arm | ndcg@10 | recall@10 | recall@100 | mrr@10 | success@1 | success@5 |" in block
    assert "| hybrid+rerank |" in block
    assert "n_pairs=160" in block
    assert "McNemar success@1" in block


def test_update_readme_replaces_between_markers(tmp_path):
    p = tmp_path / "README.md"
    p.write_text(f"top\n{START}\nold\n{END}\nbottom\n")
    update_readme(p, "NEW")
    assert p.read_text() == f"top\n{START}\nNEW\n{END}\nbottom\n"
    update_readme(p, "NEWER")
    assert "old" not in p.read_text() and "NEW\n" not in p.read_text()


def test_update_readme_needs_markers(tmp_path):
    p = tmp_path / "README.md"
    p.write_text("nothing here")
    with pytest.raises(ReportError):
        update_readme(p, "x")


def test_cli_report_refuses_on_mismatch(tmp_path, results_file, cfg):
    (tmp_path / "results").mkdir()
    doc = json.loads(results_file.read_text())
    doc["config_hash"] = "deadbeef"
    (tmp_path / "results" / "scifact.json").write_text(json.dumps(doc))
    (tmp_path / "README.md").write_text(f"{START}\n{END}\n")
    assert cli.main(["report", "--root", str(tmp_path)]) != 0
    assert (tmp_path / "README.md").read_text() == f"{START}\n{END}\n"
