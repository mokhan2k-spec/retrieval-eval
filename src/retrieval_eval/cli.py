"""Command line: python -m retrieval_eval <command>."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

from .config import Config


def _paths(args, cfg: Config) -> dict[str, Path]:
    root = Path(args.root).resolve()
    return {
        "root": root,
        "data": root / "data",
        "runs": root / "runs",
        "results": root / "results" / f"{cfg.dataset}.json",
        "readme": root / "README.md",
        "labels": root / "labels",
    }


def cmd_download(args, cfg) -> int:
    from .data import download, sha256_file

    p = _paths(args, cfg)
    z = download(cfg.dataset, p["data"])
    print(f"{z}\nsha256 {sha256_file(z)}")
    return 0


def cmd_run(args, cfg) -> int:
    from .data import load_dataset
    from .pipeline import build_components, execute
    from .runner import FailClosed

    p = _paths(args, cfg)
    # Delete a stale results file before anything that could fail.
    from .runner import clear_stale

    clear_stale(p["results"])
    data = load_dataset(cfg.dataset, cfg.split, p["data"])
    comps = build_components(cfg, data)
    fault = os.environ.get("RETRIEVAL_EVAL_FAULT_QUERY") or None  # TEST ONLY
    try:
        execute(
            cfg, data, comps,
            results_path=p["results"], runs_dir=p["runs"],
            fault_query=fault, repo_dir=p["root"],
        )
    except FailClosed as e:
        print(f"FAIL-CLOSED: {len(e.failures)} failure(s); no results file written.",
              file=sys.stderr)
        for f in e.failures[:10]:
            print(f"  {f.arm} query={f.query_id} {f.error_type}: {f.message}",
                  file=sys.stderr)
        print(f"details: {p['runs']}/<run_id>/failures.json", file=sys.stderr)
        return 2
    print(f"wrote {p['results']}")
    return 0


def cmd_report(args, cfg) -> int:
    from .report import ReportError, load_results, render_block, update_readme

    p = _paths(args, cfg)
    try:
        results = load_results(p["results"], cfg)
        block = render_block(results)
        update_readme(p["readme"], block)
    except ReportError as e:
        print(f"report refused: {e}", file=sys.stderr)
        return 3
    print(block)
    return 0


def cmd_reproduce(args, cfg) -> int:
    rc = cmd_run(args, cfg)
    return rc if rc != 0 else cmd_report(args, cfg)


def cmd_labels(args, cfg) -> int:
    from .data import load_dataset
    from .labels import write_sheet
    from .report import ReportError, load_results

    p = _paths(args, cfg)
    try:
        results = load_results(p["results"], cfg)
    except ReportError as e:
        print(f"labels refused: {e}", file=sys.stderr)
        return 3
    data = load_dataset(cfg.dataset, cfg.split, p["data"])
    for path in write_sheet(results, data, p["labels"]):
        print(path)
    return 0


def cmd_judge(args, cfg) -> int:
    from .judge import Judge, JudgeError, run_sheet

    p = _paths(args, cfg)
    try:
        judge = Judge.from_env()
        out = run_sheet(
            judge, p["labels"] / "labelling-sheet.csv", args.mode,
            p["labels"] / f"judge-{args.mode}.csv",
        )
    except JudgeError as e:
        print(f"judge failed: {e}", file=sys.stderr)
        return 4
    print(out)
    return 0


def cmd_kappa(args, cfg) -> int:
    from .kappa import report

    code, text = report(_paths(args, cfg)["labels"])
    print(text, file=sys.stderr if code else sys.stdout)
    return code


COMMANDS = {
    "download": cmd_download,
    "run": cmd_run,
    "report": cmd_report,
    "reproduce": cmd_reproduce,
    "labels": cmd_labels,
    "judge": cmd_judge,
    "kappa": cmd_kappa,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="retrieval_eval")
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--root", default=".", help="repo root (default: cwd)")
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--mode", choices=["single", "adversarial"], default="single")
    args = ap.parse_args(argv)
    cfg = replace(Config(), dataset=args.dataset)
    return COMMANDS[args.command](args, cfg)


if __name__ == "__main__":
    sys.exit(main())
