# retrieval-eval

![CI](https://github.com/mokhan2k-spec/retrieval-eval/actions/workflows/ci.yml/badge.svg)

A small, reproducible evaluation harness for retrieval pipelines on BEIR
datasets (SciFact first). It compares four arms on the same queries:

| arm | what it is |
|---|---|
| `bm25` | lexical search with `bm25s` (English stopwords, Snowball stemming) |
| `dense` | `BAAI/bge-small-en-v1.5`, normalised embeddings, cosine similarity |
| `hybrid` | reciprocal rank fusion (k=60) of `bm25` and `dense` |
| `hybrid+rerank` | `cross-encoder/ms-marco-MiniLM-L-6-v2` rescoring the top 40 hybrid candidates |

Everything runs on CPU. The dataset zip is checksum-pinned.

## Why it fails closed

Many eval scripts fail open: when the reranker times out or returns garbage on
a few queries, they print a warning, fall back to the input order for those
queries and still write a results row, so the table quietly mixes reranked and
un-reranked rankings and the headline number is wrong. This harness does the
opposite. If any arm errors or degrades on any query, the process exits
non-zero, writes **no** results file, and writes
`runs/<run_id>/failures.json` listing every failed (arm, query id, error type,
message). The reranker wrapper never returns the input order on error; it
raises `RerankError`, and wrong-length or NaN scores count as errors. Any
existing results file is deleted at the start of every run and results are
written atomically only after all arms finish. `report` refuses to render a
results file whose config hash differs from the current config.

## Metrics

- `nDCG@10`: linear gain, log2 discount, ideal ranking from the qrels.
- `recall@k`: the standard IR quantity, the **fraction of the relevant
  documents** that appear in the top k.
- `success@k`: whether **at least one** relevant document appears in the top k.
  This is a weaker, per-query yes/no number and is always labelled
  `success@k`, never `recall`.
- `MRR@10`: reciprocal rank of the first relevant document.

Significance (hybrid vs hybrid+rerank, and bm25 vs hybrid) is computed only
over complete pairs (queries where both arms have a value) and reports
`n_pairs`: a two-sided paired randomization test on per-query nDCG@10 (10,000
sign flips, fixed seed), a Wilcoxon signed-rank test, the mean difference with
a 95% bootstrap CI, and an exact McNemar test on `success@1`.

## Reproduce

With `make` (Linux/macOS, needs [uv](https://docs.astral.sh/uv/)):

```
make setup        # venv + all dependencies (downloads torch)
make test         # pytest, no network, no models
make mutation     # mutmut on metrics.py, stats.py, runner.py (Linux only)
make reproduce    # download + verify data, run all arms, write results/scifact.json, update this README
make labels       # build labels/labelling-sheet.csv from the run
make kappa        # agreement report (needs >= 100 human labels)
```

Without `make` (Windows, or anywhere):

```
uv venv
uv pip install -e ".[dev,models]"
uv run --no-sync python -m pytest -q
uv run --no-sync python -m retrieval_eval reproduce
uv run --no-sync python -m retrieval_eval labels
uv run --no-sync python -m retrieval_eval judge --mode single        # optional, see below
uv run --no-sync python -m retrieval_eval kappa
```

Other commands: `download`, `run` (arms only), `report` (render table).
`--dataset NAME` selects a BEIR dataset once its sha256 is pinned in
`src/retrieval_eval/data.py`.

Test-only fault hook: setting `RETRIEVAL_EVAL_FAULT_QUERY=<query id>` makes the
reranker raise on that query, to demonstrate the fail-closed exit. Never set it
for a real run.

## Results

SciFact test split. Produced by `make reproduce`; do not edit by hand.

<!-- results:start -->
<!-- results:end -->

## Judge calibration

Status: **pending, needs >= 100 human labels.**

`labels/labelling-sheet.csv` holds 120 (query, document) pairs sampled with a
fixed seed from the hybrid and hybrid+rerank top-10s, about half qrels-relevant
and half not, including documents the reranker promoted and demoted. The sheet
hides the qrels label and the arm; `labels/key.csv` holds them. Follow
`labels/INSTRUCTIONS.md`, fill `human_label`, then run `kappa`.

`judge` runs an LLM judge over the same sheet through any OpenAI-compatible
endpoint (`JUDGE_BASE_URL`, `JUDGE_MODEL`, `JUDGE_API_KEY`), in `single` mode
or `adversarial` mode (a second pass argues against the first label). It is
never run in CI. `kappa` reports Cohen's kappa for human vs qrels, human vs
judge-single and human vs judge-adversarial, and refuses to report with fewer
than 100 human labels.

## Layout

```
src/retrieval_eval/   data, retrievers, rerank, metrics, stats, runner (fail-closed gate),
                      pipeline, report, labels, judge, kappa, cli
tests/                pytest, fakes only
scripts/              mutation_gate.py
```

MIT licensed.
