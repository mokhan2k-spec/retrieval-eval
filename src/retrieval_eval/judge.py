"""LLM judge over the labelling sheet (any OpenAI-compatible endpoint).

Configured only through environment variables: JUDGE_BASE_URL, JUDGE_MODEL,
JUDGE_API_KEY. Run it by hand; it is never run in CI. Any HTTP or parse error
aborts the run rather than guessing a label.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

SYSTEM = (
    "You judge relevance for scientific claim verification. A document is "
    "relevant (1) if its abstract contains evidence that supports or refutes "
    "the claim; otherwise not relevant (0). A shared topic is not enough."
)


class JudgeError(Exception):
    pass


def parse_label(text: str) -> int:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and obj.get("label") in (0, 1):
            return int(obj["label"])
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r'"label"\s*:\s*"?([01])\b', text)
    if not m:
        raise JudgeError(f"could not parse a 0/1 label from: {text[:200]!r}")
    return int(m.group(1))


def parse_reason(text: str) -> str:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return str(obj.get("reason", ""))
    except (json.JSONDecodeError, TypeError):
        pass
    return ""


class Judge:
    def __init__(self, base_url: str, model: str, api_key: str, http=None):
        if not (base_url and model and api_key):
            raise JudgeError("JUDGE_BASE_URL, JUDGE_MODEL and JUDGE_API_KEY must be set")
        if http is None:
            import httpx

            http = httpx.Client(timeout=120)
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.http = http

    @classmethod
    def from_env(cls, http=None) -> "Judge":
        return cls(
            os.environ.get("JUDGE_BASE_URL", ""),
            os.environ.get("JUDGE_MODEL", ""),
            os.environ.get("JUDGE_API_KEY", ""),
            http,
        )

    def chat(self, user: str) -> str:
        resp = self.http.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                ],
            },
        )
        resp.raise_for_status()
        try:
            return resp.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise JudgeError(f"unexpected response shape: {e}") from e

    @staticmethod
    def _item(row: dict) -> str:
        return (
            f"Claim: {row['query']}\n\nTitle: {row['doc_title']}\n"
            f"Abstract: {row['doc_text']}\n"
        )

    def judge_single(self, row: dict) -> dict:
        out = self.chat(
            self._item(row)
            + '\nReply with JSON only: {"label": 0 or 1, "reason": "<one sentence>"}'
        )
        return {"label": parse_label(out), "reason": parse_reason(out)}

    def judge_adversarial(self, row: dict) -> dict:
        first = self.judge_single(row)
        out = self.chat(
            self._item(row)
            + f"\nA first reviewer labelled this {first['label']} because: "
            f"{first['reason']}\nArgue AGAINST that label as strongly as you can, "
            "then decide the final label on the merits. Reply with JSON only: "
            '{"label": 0 or 1, "reason": "<one sentence>"}'
        )
        return {
            "label": parse_label(out),
            "reason": parse_reason(out),
            "first_label": first["label"],
        }


def run_sheet(judge: Judge, sheet: Path, mode: str, out: Path) -> Path:
    if mode not in ("single", "adversarial"):
        raise JudgeError(f"unknown mode {mode!r}")
    with open(sheet, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    results = []
    for row in rows:
        r = (
            judge.judge_single(row)
            if mode == "single"
            else judge.judge_adversarial(row)
        )
        results.append(
            {
                "item_id": row["item_id"],
                "judge_label": r["label"],
                "first_label": r.get("first_label", ""),
                "reason": r["reason"],
            }
        )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["item_id", "judge_label", "first_label", "reason"])
        w.writeheader()
        w.writerows(results)
    return Path(out)
