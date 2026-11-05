"""Render results into the README. Refuses stale or mismatched results."""

from __future__ import annotations

import json
from pathlib import Path

from .config import ARMS, COMPARISONS, Config

START = "<!-- results:start -->"
END = "<!-- results:end -->"


class ReportError(Exception):
    pass


def load_results(path: Path, cfg: Config) -> dict:
    path = Path(path)
    if not path.exists():
        raise ReportError(f"results file not found: {path}")
    results = json.loads(path.read_text(encoding="utf-8"))
    if results.get("config_hash") != cfg.hash():
        raise ReportError(
            "config hash mismatch: results were produced with a different "
            f"config (file {results.get('config_hash')}, current {cfg.hash()}); "
            "re-run `reproduce`"
        )
    return results


def render_table(results: dict) -> str:
    cfg = results["config"]
    cols = [f"ndcg@{cfg['ndcg_k']}"]
    cols += [f"recall@{k}" for k in cfg["recall_ks"]]
    cols += [f"mrr@{cfg['mrr_k']}"]
    cols += [f"success@{k}" for k in cfg["success_ks"]]
    lines = [
        "| arm | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for arm in ARMS:
        m = results["arms"][arm]["metrics"]
        lines.append(f"| {arm} | " + " | ".join(f"{m[c]:.4f}" for c in cols) + " |")
    ds = results["dataset"]
    lines.append("")
    lines.append(
        f"{ds['name']} {ds['split']}: {ds['n_queries']} queries, {ds['n_docs']} docs. "
        f"run `{results['run_id']}`, config `{results['config_hash'][:12]}`, "
        f"dataset sha256 `{ds['zip_sha256'][:12]}`."
    )
    return "\n".join(lines)


def render_significance(results: dict) -> str:
    lines = []
    for a, b in COMPARISONS:
        s = results["significance"][f"{a}_vs_{b}"]
        lo, hi = s["ci95_ndcg"]
        mc = s["mcnemar_success1"]
        lines.append(
            f"- {a} -> {b} (n_pairs={s['n_pairs']}): "
            f"mean nDCG@10 diff {s['mean_diff_ndcg']:+.4f} "
            f"(95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]); "
            f"permutation p={s['permutation_p_ndcg']:.4g}, "
            f"Wilcoxon p={s['wilcoxon_p_ndcg']:.4g}; "
            f"McNemar success@1 p={mc['p']:.4g} "
            f"(wins {mc['b_wins']}, losses {mc['c_losses']})"
        )
    return "\n".join(lines)


def render_block(results: dict) -> str:
    return (
        render_table(results)
        + "\n\nPaired significance (B vs A, positive = B better):\n\n"
        + render_significance(results)
    )


def update_readme(readme: Path, block: str) -> None:
    text = Path(readme).read_text(encoding="utf-8")
    i, j = text.find(START), text.find(END)
    if i == -1 or j == -1 or j < i:
        raise ReportError("README is missing the results markers")
    new = text[: i + len(START)] + "\n" + block + "\n" + text[j:]
    Path(readme).write_text(new, encoding="utf-8")
