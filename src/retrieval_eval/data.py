"""BEIR dataset download, checksum verification and loading."""

from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"

# sha256 of each pinned zip. A dataset without a pin cannot be used: an
# unverifiable download is a hard error, not a warning.
DATASET_SHA256: dict[str, str] = {
    "scifact": "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165",
}


class DatasetError(Exception):
    pass


@dataclass
class Dataset:
    name: str
    zip_sha256: str
    corpus: dict[str, dict[str, str]]  # doc_id -> {"title", "text"}
    queries: dict[str, str]  # only queries that have test qrels
    qrels: dict[str, dict[str, int]]

    def doc_text(self, doc_id: str) -> str:
        d = self.corpus[doc_id]
        return (d.get("title", "") + " " + d.get("text", "")).strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_zip(path: Path, name: str, pins: dict[str, str] | None = None) -> str:
    pins = DATASET_SHA256 if pins is None else pins
    expected = pins.get(name)
    if not expected:
        raise DatasetError(f"no pinned sha256 for dataset {name!r}")
    actual = sha256_file(path)
    if actual != expected:
        raise DatasetError(
            f"sha256 mismatch for {path.name}: expected {expected}, got {actual}"
        )
    return actual


def download(name: str, data_dir: Path) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / f"{name}.zip"
    if not zip_path.exists():
        try:  # use the OS trust store when available (some hosts omit intermediates)
            import truststore

            truststore.inject_into_ssl()
        except ImportError:
            pass
        tmp = zip_path.with_name(zip_path.name + ".part")
        urllib.request.urlretrieve(BASE_URL.format(name=name), tmp)
        tmp.replace(zip_path)
    return zip_path


def load_zip(zip_path: Path, name: str, split: str, zip_sha256: str) -> Dataset:
    corpus: dict[str, dict[str, str]] = {}
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()

        def find(suffix: str) -> str:
            hits = [n for n in names if n.endswith(suffix)]
            if not hits:
                raise DatasetError(f"{suffix} not found in {zip_path.name}")
            return hits[0]

        with z.open(find("corpus.jsonl")) as f:
            for line in f:
                row = json.loads(line)
                corpus[str(row["_id"])] = {
                    "title": row.get("title", ""),
                    "text": row.get("text", ""),
                }
        all_queries: dict[str, str] = {}
        with z.open(find("queries.jsonl")) as f:
            for line in f:
                row = json.loads(line)
                all_queries[str(row["_id"])] = row["text"]
        with z.open(find(f"qrels/{split}.tsv")) as f:
            rows = csv.reader((ln.decode("utf-8") for ln in f), delimiter="\t")
            next(rows)  # header
            for r in rows:
                if len(r) < 3:
                    continue
                qrels.setdefault(r[0], {})[r[1]] = int(r[2])
    for qid in sorted(qrels):
        queries[qid] = all_queries[qid]
    return Dataset(name, zip_sha256, corpus, queries, qrels)


def load_dataset(name: str, split: str, data_dir: Path) -> Dataset:
    zip_path = download(name, data_dir)
    sha = verify_zip(zip_path, name)
    return load_zip(zip_path, name, split, sha)
