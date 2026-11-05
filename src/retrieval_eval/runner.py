"""Fail-closed run logic (pure, no models, no network).

Contract: a results file exists only if EVERY arm answered EVERY query with a
non-empty ranking. Anything else raises `FailClosed`; the caller writes
`failures.json` and exits non-zero without producing a results file.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Failure:
    arm: str
    query_id: str
    error_type: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


class FailClosed(Exception):
    def __init__(self, failures: Sequence[Failure]):
        self.failures = list(failures)
        super().__init__(f"{len(self.failures)} arm/query failure(s)")


@dataclass
class ArmOutcome:
    arm: str
    rankings: dict[str, list[str]] = field(default_factory=dict)
    failures: list[Failure] = field(default_factory=list)


def run_arm(
    arm: str,
    queries: Mapping[str, str],
    fn: Callable[[str, str], Sequence[str]],
) -> ArmOutcome:
    """Run `fn(query_id, query_text) -> ranked doc ids` for every query.

    Errors are recorded per query, never swallowed and never replaced by a
    substitute ranking.
    """
    out = ArmOutcome(arm)
    for qid, text in queries.items():
        try:
            ranking = list(fn(qid, text))
        except Exception as e:
            out.failures.append(Failure(arm, qid, type(e).__name__, str(e)))
            continue
        out.rankings[qid] = ranking
    return out


def finalize(
    outcomes: Sequence[ArmOutcome], query_ids: Sequence[str]
) -> dict[str, dict[str, list[str]]]:
    """Return {arm: {qid: ranking}} or raise FailClosed.

    Besides recorded errors, a missing or empty ranking for any (arm, query)
    is itself a failure.
    """
    failures: list[Failure] = []
    for o in outcomes:
        failures.extend(o.failures)
        recorded = {f.query_id for f in o.failures}
        for qid in query_ids:
            if qid in recorded:
                continue
            ranking = o.rankings.get(qid)
            if ranking is None:
                failures.append(
                    Failure(o.arm, qid, "MissingRanking", "arm produced no ranking")
                )
            elif len(ranking) == 0:
                failures.append(
                    Failure(o.arm, qid, "EmptyRanking", "arm produced an empty ranking")
                )
    if failures:
        raise FailClosed(failures)
    return {o.arm: {q: o.rankings[q] for q in query_ids} for o in outcomes}


def clear_stale(results_path: Path) -> None:
    """Delete any results file left by an earlier run; results are never reused."""
    Path(results_path).unlink(missing_ok=True)


def write_failures(run_dir: Path, failures: Sequence[Failure]) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "failures.json"
    path.write_text(
        json.dumps({"failures": [f.to_dict() for f in failures]}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def write_results_atomic(path: Path, results: Mapping) -> None:
    """Write via a temp file in the same directory, then os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
