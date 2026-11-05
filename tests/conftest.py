"""Synthetic dataset and fake retrievers/rerankers: no network, no models."""

import random
from dataclasses import replace

import pytest

from retrieval_eval.config import Config
from retrieval_eval.data import Dataset
from retrieval_eval.pipeline import Components

N_Q, N_D = 160, 400


@pytest.fixture
def cfg():
    return replace(Config(), permutations=300, bootstrap=300)


@pytest.fixture
def dataset():
    corpus = {
        f"d{i}": {"title": f"title{i}", "text": f"body{i} claim{i % N_Q}"}
        for i in range(N_D)
    }
    queries = {f"q{i}": f"claim{i}" for i in range(N_Q)}
    qrels = {f"q{i}": {f"d{i}": 1} for i in range(N_Q)}
    return Dataset("fake", "0" * 64, corpus, queries, qrels)


def _search(prefix, dataset, hit_rate):
    ids = list(dataset.corpus)
    rel = {t: next(iter(dataset.qrels[q])) for q, t in dataset.queries.items()}

    def fn(text, k):
        rng = random.Random(prefix + text)
        docs = ids[:]
        rng.shuffle(docs)
        if rng.random() < hit_rate:
            docs.remove(rel[text])
            docs.insert(rng.randrange(0, 12), rel[text])
        return docs[:k]

    return fn


@pytest.fixture
def make_comps(dataset):
    def make(scorer=None):
        def default_scorer(query, texts):
            rng = random.Random(query)
            return [(5.0 if query in t else 0.0) + rng.random() for t in texts]

        return Components(
            _search("b", dataset, 0.7),
            _search("d", dataset, 0.8),
            scorer or default_scorer,
            {"dense": None, "rerank": None},
        )

    return make
