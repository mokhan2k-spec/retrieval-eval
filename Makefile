.PHONY: setup setup-lite test mutation reproduce labels kappa

# Run inside the project venv without re-syncing it.
RUN = uv run --no-sync

# Full install, including the model-running dependencies (torch etc.).
setup:
	uv venv
	uv pip install -e ".[dev,models]"

# Tests and mutation testing need no models; CI uses this.
setup-lite:
	uv venv
	uv pip install -e ".[dev]"

test:
	$(RUN) pytest -q

# Mutation score on metrics.py, stats.py and runner.py must reach MIN_SCORE.
MIN_SCORE ?= 70
mutation:
	$(RUN) python scripts/mutation_gate.py --min $(MIN_SCORE)

reproduce:
	$(RUN) python -m retrieval_eval reproduce

labels:
	$(RUN) python -m retrieval_eval labels

kappa:
	$(RUN) python -m retrieval_eval kappa
