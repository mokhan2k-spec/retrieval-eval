import csv
import json

import pytest

from retrieval_eval.judge import Judge, JudgeError, parse_label, run_sheet


class FakeResp:
    def __init__(self, content, status=200):
        self._c, self.status = content, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return {"choices": [{"message": {"content": self._c}}]}


class FakeHttp:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def post(self, url, headers=None, json=None):
        self.calls.append((url, headers, json))
        return self.replies.pop(0)


def mk(replies):
    http = FakeHttp(replies)
    return Judge("http://judge.example/v1/", "m", "k", http), http


ROW = {"item_id": "item001", "query": "claim", "doc_title": "T", "doc_text": "abs"}


def test_parse_label():
    assert parse_label('{"label": 1, "reason": "x"}') == 1
    assert parse_label('noise {"label": 0, "reason": "x"} noise') == 0
    with pytest.raises(JudgeError):
        parse_label("maybe")
    with pytest.raises(JudgeError):
        parse_label('{"label": 2}')


def test_single_mode_request_shape():
    j, http = mk([FakeResp('{"label": 1, "reason": "supports"}')])
    out = j.judge_single(ROW)
    assert out == {"label": 1, "reason": "supports"}
    url, headers, body = http.calls[0]
    assert url == "http://judge.example/v1/chat/completions"
    assert headers["Authorization"] == "Bearer k"
    assert body["model"] == "m" and body["temperature"] == 0
    assert "Claim: claim" in body["messages"][1]["content"]


def test_adversarial_makes_second_pass_with_first_label():
    j, http = mk([
        FakeResp('{"label": 1, "reason": "first"}'),
        FakeResp('{"label": 0, "reason": "final"}'),
    ])
    out = j.judge_adversarial(ROW)
    assert out == {"label": 0, "reason": "final", "first_label": 1}
    assert len(http.calls) == 2
    second = http.calls[1][2]["messages"][1]["content"]
    assert "Argue AGAINST" in second and "first" in second


def test_http_error_aborts():
    j, _ = mk([FakeResp("", status=500)])
    with pytest.raises(RuntimeError):
        j.judge_single(ROW)


def test_unparseable_reply_aborts():
    j, _ = mk([FakeResp("I think so")])
    with pytest.raises(JudgeError):
        j.judge_single(ROW)


def test_missing_env_is_error(monkeypatch):
    for v in ("JUDGE_BASE_URL", "JUDGE_MODEL", "JUDGE_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(JudgeError):
        Judge.from_env()


def test_run_sheet_writes_csv(tmp_path):
    sheet = tmp_path / "s.csv"
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["item_id", "query", "doc_title", "doc_text"])
        w.writeheader()
        for i in (1, 2):
            w.writerow({**ROW, "item_id": f"item00{i}"})
    j, _ = mk([FakeResp('{"label": 1, "reason": "a"}'), FakeResp('{"label": 0, "reason": "b"}')])
    out = run_sheet(j, sheet, "single", tmp_path / "out" / "judge-single.csv")
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert [(r["item_id"], r["judge_label"]) for r in rows] == [("item001", "1"), ("item002", "0")]
    with pytest.raises(JudgeError):
        run_sheet(j, sheet, "weird", tmp_path / "x.csv")


def test_chat_bad_shape():
    class Bad:
        def raise_for_status(self):
            pass

        def json(self):
            return {"nope": 1}

    j, _ = mk([Bad()])
    with pytest.raises(JudgeError):
        j.chat("x")
    assert json.dumps({})  # keep json import used
