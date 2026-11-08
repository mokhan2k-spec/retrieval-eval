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

The metrics are written out by hand so they can be mutation-tested.
`tests/test_metrics_vs_ir_measures.py` checks every one of them, per query,
against [`ir_measures`](https://github.com/terrierteam/ir_measures) (the
`pytrec_eval` backend) on random graded qrels.

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

Run on 2026-11-05 on a Windows 11 laptop, CPU only, while the machine was otherwise loaded. `reproduce` took 1917 s (about 32 min) wall time, about 17 min of it embedding the corpus. Dense embeddings on CPU can differ in the last float digits across machines.

<!-- results:start -->
| arm | ndcg@10 | recall@10 | recall@100 | mrr@10 | success@1 | success@5 |
|---|---|---|---|---|---|---|
| bm25 | 0.6863 | 0.8187 | 0.9127 | 0.6492 | 0.5500 | 0.7567 |
| dense | 0.7127 | 0.8362 | 0.9417 | 0.6822 | 0.6067 | 0.7800 |
| hybrid | 0.7234 | 0.8554 | 0.9650 | 0.6879 | 0.5967 | 0.7867 |
| hybrid+rerank | 0.6976 | 0.8322 | 0.9650 | 0.6646 | 0.5833 | 0.7667 |

scifact test: 300 queries, 5183 docs. run `20261105T211238Z-63155615-f156-47c2-a829-e59e9dfd3ba2`, config `cb43cf34fb92`, dataset sha256 `536e14446a0b`.

Paired significance (B vs A, positive = B better):

- hybrid -> hybrid+rerank (n_pairs=300): mean nDCG@10 diff -0.0258 (95% bootstrap CI [-0.0526, +0.0015]); permutation p=0.06719, Wilcoxon p=0.08208; McNemar success@1 p=0.6655 (wins 22, losses 26)
- bm25 -> hybrid (n_pairs=300): mean nDCG@10 diff +0.0371 (95% bootstrap CI [+0.0169, +0.0568]); permutation p=0.0004, Wilcoxon p=2.54e-05; McNemar success@1 p=0.01254 (wins 21, losses 7)
<!-- results:end -->

## Judge calibration

Status: **two-model agreement done (8 Nov 2026); human calibration pending, needs >= 100 human labels.**

### Two independent model labellers

Two different models from two vendors each labelled all 120 rows of the sheet
on their own. Each run was blind: no `key.csv`, no tools and no web, only
`labels/model-prompt.txt` with the sheet pasted after it, in a fresh session.

- Claude: Claude Code CLI 2.1.319, `--model sonnet`. Labels are in `labels/model-claude.csv`.
- Codex: Codex CLI 0.160.1, model `gpt-6.1-sol`, empty `CODEX_HOME`. Labels are in `labels/model-codex.csv`.

`python scripts/model_agreement.py`:

| pair | Cohen's kappa | raw agreement |
|---|---|---|
| Claude vs Codex | **0.878** | 113/120 |
| Claude vs qrels | 0.717 | 103/120 |
| Codex vs qrels | 0.667 | 100/120 |
| both models agree (n=113) vs qrels | 0.733 | 98/113 |

What this shows: the two models agree with each other almost perfectly. Each
model agrees substantially with the dataset's own labels (Landis and Koch
bands). Both models mark fewer pairs relevant than the qrels do (47 and 48
against 60), so most disagreements are qrels-relevant documents that the
models read as off-claim.

What it does not show: that the models agree with a person. Two models can
share a bias, so this is a model-consistency check, not a human calibration.
The human column below is still empty.

### Human labels (pending)

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
