import hashlib
import json
import zipfile

import pytest

from retrieval_eval.data import DatasetError, load_zip, sha256_file, verify_zip


@pytest.fixture
def tiny_zip(tmp_path):
    p = tmp_path / "tiny.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("tiny/corpus.jsonl", json.dumps({"_id": "d1", "title": "T", "text": "x"}) + "\n"
                   + json.dumps({"_id": "d2", "title": "U", "text": "y"}) + "\n")
        z.writestr("tiny/queries.jsonl", json.dumps({"_id": "1", "text": "claim a"}) + "\n"
                   + json.dumps({"_id": "2", "text": "claim b"}) + "\n")
        z.writestr("tiny/qrels/test.tsv", "query-id\tcorpus-id\tscore\n1\td1\t1\n")
        z.writestr("tiny/qrels/train.tsv", "query-id\tcorpus-id\tscore\n2\td2\t1\n")
    return p


def test_load_zip_uses_only_test_qrels_queries(tiny_zip):
    ds = load_zip(tiny_zip, "tiny", "test", "abc")
    assert ds.queries == {"1": "claim a"}
    assert ds.qrels == {"1": {"d1": 1}}
    assert ds.corpus["d2"]["title"] == "U"
    assert ds.doc_text("d1") == "T x"
    assert ds.zip_sha256 == "abc"


def test_verify_zip(tiny_zip):
    sha = hashlib.sha256(tiny_zip.read_bytes()).hexdigest()
    assert sha256_file(tiny_zip) == sha
    assert verify_zip(tiny_zip, "tiny", {"tiny": sha}) == sha
    with pytest.raises(DatasetError, match="mismatch"):
        verify_zip(tiny_zip, "tiny", {"tiny": "0" * 64})
    with pytest.raises(DatasetError, match="no pinned"):
        verify_zip(tiny_zip, "other", {"tiny": sha})
