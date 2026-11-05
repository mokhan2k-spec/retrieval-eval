import csv

import pytest

from retrieval_eval.labels import write_sheet
from retrieval_eval.pipeline import execute


@pytest.fixture
def results(cfg, dataset, make_comps, tmp_path):
    return execute(
        cfg, dataset, make_comps(),
        results_path=tmp_path / "r.json", runs_dir=tmp_path / "runs",
    )


def _read(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_sheet_shape_and_hidden_fields(results, dataset, tmp_path):
    sheet, key, instr = write_sheet(results, dataset, tmp_path / "labels")
    rows = _read(sheet)
    assert len(rows) >= 120
    assert list(rows[0]) == [
        "item_id", "query_id", "query", "doc_id", "doc_title", "doc_text",
        "human_label", "notes",
    ]
    assert all(r["human_label"] == "" for r in rows)
    assert len({r["item_id"] for r in rows}) == len(rows)
    assert len({(r["query_id"], r["doc_id"]) for r in rows}) == len(rows)
    header = open(sheet, encoding="utf-8").readline()
    for hidden in ("qrels", "arm", "rank", "category"):
        assert hidden not in header
    assert "qrels" not in sheet.read_text(encoding="utf-8").splitlines()[0]
    assert instr.read_text(encoding="utf-8").startswith("# Labelling instructions")


def test_key_maps_items_and_is_balanced(results, dataset, tmp_path):
    sheet, key, _ = write_sheet(results, dataset, tmp_path / "labels")
    krows = _read(key)
    srows = _read(sheet)
    assert [k["item_id"] for k in krows] == [s["item_id"] for s in srows]
    rel = sum(int(k["qrels_label"]) for k in krows)
    assert rel == len(krows) // 2
    by_item = {s["item_id"]: s for s in srows}
    for k in krows:
        s = by_item[k["item_id"]]
        truth = dataset.qrels[s["query_id"]].get(s["doc_id"], 0)
        assert int(k["qrels_label"]) == (1 if truth > 0 else 0)
    cats = {k["category"] for k in krows}
    assert {"promoted", "demoted"} <= cats


def test_sheet_deterministic_for_seed(results, dataset, tmp_path):
    a, _, _ = write_sheet(results, dataset, tmp_path / "a", seed=5)
    b, _, _ = write_sheet(results, dataset, tmp_path / "b", seed=5)
    c, _, _ = write_sheet(results, dataset, tmp_path / "c", seed=6)
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes() != c.read_bytes()


def test_too_few_candidates_raises(results, dataset, tmp_path):
    with pytest.raises(ValueError, match="candidate pairs"):
        write_sheet(results, dataset, tmp_path / "x", n=100000)
