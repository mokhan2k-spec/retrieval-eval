# Labelling instructions

You are judging whether a document is relevant to a query. There are no
right-or-wrong answers to hide: label what you honestly read.

The queries are scientific claims (BEIR SciFact). Each document is the title
and abstract of a paper.

Fill in the `human_label` column of `labelling-sheet.csv`:

- `1` (relevant): the abstract contains evidence that SUPPORTS or REFUTES the
  claim. It does not matter which way the evidence points, only that the
  abstract speaks directly to the claim.
- `0` (not relevant): the abstract is on a related topic but contains no
  evidence for or against the claim, or is unrelated.

Rules of thumb:

1. Judge only from the title and abstract shown. Do not search the web.
2. A shared keyword or topic is not enough. The abstract must bear on the
   specific claim.
3. If you genuinely cannot decide, pick the closer label and write why in
   `notes`.
4. Do not open `key.csv` until you have finished. It holds the dataset's own
   labels and which system returned each document, and would bias you.
5. Label at least 100 rows; all rows is better. Rows may be done in any order.

Save the file as CSV (UTF-8). Then run `make kappa`.
