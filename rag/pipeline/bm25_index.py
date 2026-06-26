#!/usr/bin/env python3
"""Indexed BM25 builder, loader, and scorer."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from rag.pipeline.bm25_search import tokenize


INDEX_FORMAT_VERSION = 1
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
_INDEX_CACHE: dict[tuple[Path, Path | None, bool, bool], "NumpyBM25Index"] = {}


class IndexedBM25Error(RuntimeError):
    """Raised when an indexed BM25 asset is missing, stale, or malformed."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_metadata(path: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        "file": path.name,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _safe_field_dir(field_name: str) -> str:
    return hashlib.sha256(field_name.encode("utf-8")).hexdigest()[:16]


def _write_field_from_tokenized(
    root: Path,
    field_name: str,
    corpus_tokens: list[list[str]],
    *,
    k1: float,
    b: float,
) -> dict[str, Any]:
    field_dir = root / _safe_field_dir(field_name)
    field_dir.mkdir(parents=True, exist_ok=False)
    doc_count = len(corpus_tokens)
    doc_lens = np.asarray([len(tokens) for tokens in corpus_tokens], dtype=np.uint32)
    document_frequencies: Counter[str] = Counter()
    term_frequencies: list[Counter[str]] = []
    for tokens in corpus_tokens:
        counts = Counter(tokens)
        term_frequencies.append(counts)
        document_frequencies.update(counts.keys())

    terms = sorted(document_frequencies)
    term_to_index = {term: index for index, term in enumerate(terms)}
    dfs = np.asarray([document_frequencies[term] for term in terms], dtype=np.uint64)
    term_offsets = np.empty(len(terms) + 1, dtype=np.uint64)
    term_offsets[0] = 0
    np.cumsum(dfs, out=term_offsets[1:])
    posting_count = int(term_offsets[-1])
    posting_doc_indices = np.empty(posting_count, dtype=np.uint32)
    posting_term_frequencies = np.empty(posting_count, dtype=np.uint32)
    cursors = term_offsets[:-1].copy()
    for doc_index, counts in enumerate(term_frequencies):
        for term, frequency in counts.items():
            term_index = term_to_index[term]
            position = int(cursors[term_index])
            posting_doc_indices[position] = doc_index
            posting_term_frequencies[position] = frequency
            cursors[term_index] += 1
    if not np.array_equal(cursors, term_offsets[1:]):
        raise IndexedBM25Error(f"{field_name}: posting cursor mismatch")

    if doc_count:
        idf = np.log(
            1.0 + (doc_count - dfs.astype(np.float64) + 0.5)
            / (dfs.astype(np.float64) + 0.5)
        )
    else:
        idf = np.zeros(len(terms), dtype=np.float64)
    arrays = {
        "doc_lens": doc_lens,
        "posting_doc_indices": posting_doc_indices,
        "posting_term_frequencies": posting_term_frequencies,
        "term_offsets": term_offsets,
        "term_idf": idf.astype(np.float64, copy=False),
    }
    array_metadata: dict[str, Any] = {}
    for name, array in arrays.items():
        path = field_dir / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        array_metadata[name] = _array_metadata(path, array)
    terms_path = field_dir / "terms.pkl"
    with terms_path.open("wb") as handle:
        pickle.dump(terms, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return {
        "field_name": field_name,
        "directory": field_dir.name,
        "doc_count": doc_count,
        "average_doc_length": (
            float(doc_lens.astype(np.float64).mean()) if doc_count else 0.0
        ),
        "vocabulary_size": len(terms),
        "posting_count": posting_count,
        "k1": k1,
        "b": b,
        "arrays": array_metadata,
        "terms": {
            "file": terms_path.name,
            "count": len(terms),
            "bytes": terms_path.stat().st_size,
            "sha256": sha256_file(terms_path),
        },
    }


def build_index_from_dataframe(
    df: pd.DataFrame,
    output_dir: Path,
    fields: list[str],
    *,
    corpus_metadata: dict[str, Any] | None = None,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> dict[str, Any]:
    """Build a small/test index atomically from an in-memory DataFrame."""
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Index output already exists: {output_dir}")
    missing = [field for field in fields if field not in df.columns]
    if missing:
        raise KeyError(f"Missing indexed fields: {missing}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    started = time.perf_counter()
    try:
        field_records = []
        for field_name in fields:
            values = df[field_name].fillna("").astype(str)
            field_records.append(
                _write_field_from_tokenized(
                    temporary,
                    field_name,
                    [tokenize(value) for value in values],
                    k1=k1,
                    b=b,
                )
            )
        manifest = {
            "schema_version": 1,
            "status": "COMPLETE",
            "index_format_version": INDEX_FORMAT_VERSION,
            "doc_count": len(df),
            "fields": field_records,
            "bm25": {"k1": k1, "b": b},
            "corpus": corpus_metadata or {"row_count": len(df)},
            "build": {
                "duration_seconds": time.perf_counter() - started,
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__,
            },
        }
        manifest["semantic_sha256"] = canonical_json_sha256(
            {
                "index_format_version": manifest["index_format_version"],
                "doc_count": manifest["doc_count"],
                "fields": manifest["fields"],
                "bm25": manifest["bm25"],
                "corpus": manifest["corpus"],
            }
        )
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_dir)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


class IndexedBM25Field:
    def __init__(self, root: Path, metadata: dict[str, Any], mmap: bool) -> None:
        self.metadata = metadata
        self.field_name = str(metadata["field_name"])
        field_dir = root / metadata["directory"]
        mode = "r" if mmap else None
        self.doc_lens = np.load(field_dir / "doc_lens.npy", mmap_mode=mode)
        self.posting_doc_indices = np.load(
            field_dir / "posting_doc_indices.npy", mmap_mode=mode
        )
        self.posting_term_frequencies = np.load(
            field_dir / "posting_term_frequencies.npy", mmap_mode=mode
        )
        self.term_offsets = np.load(field_dir / "term_offsets.npy", mmap_mode=mode)
        self.term_idf = np.load(field_dir / "term_idf.npy", mmap_mode=mode)
        with (field_dir / "terms.pkl").open("rb") as handle:
            terms = pickle.load(handle)
        if not isinstance(terms, list) or not all(
            isinstance(term, str) for term in terms
        ):
            raise IndexedBM25Error(f"{self.field_name}: invalid term dictionary")
        self.term_to_index = {term: index for index, term in enumerate(terms)}
        self.doc_count = int(metadata["doc_count"])
        self.avg_doc_len = float(metadata["average_doc_length"])
        self.k1 = float(metadata["k1"])
        self.b = float(metadata["b"])
        if len(self.doc_lens) != self.doc_count:
            raise IndexedBM25Error(f"{self.field_name}: doc length shape mismatch")
        if len(self.term_offsets) != len(terms) + 1:
            raise IndexedBM25Error(f"{self.field_name}: offset shape mismatch")

    def close(self) -> None:
        for array in (
            self.doc_lens,
            self.posting_doc_indices,
            self.posting_term_frequencies,
            self.term_offsets,
            self.term_idf,
        ):
            mapping = getattr(array, "_mmap", None)
            if mapping is not None:
                mapping.close()

    def add_scores(
        self,
        scores: np.ndarray,
        query_tokens: Iterable[str],
        weight: float,
    ) -> None:
        if weight <= 0:
            return
        for term in query_tokens:
            term_index = self.term_to_index.get(term)
            if term_index is None:
                continue
            start = int(self.term_offsets[term_index])
            end = int(self.term_offsets[term_index + 1])
            docs = np.asarray(self.posting_doc_indices[start:end], dtype=np.intp)
            frequencies = np.asarray(
                self.posting_term_frequencies[start:end],
                dtype=np.float64,
            )
            lengths = np.asarray(self.doc_lens[docs], dtype=np.float64)
            norm = self.k1 * (
                1.0 - self.b + self.b * (lengths / self.avg_doc_len)
            ) if self.avg_doc_len else self.k1
            contribution = float(self.term_idf[term_index]) * (
                frequencies * (self.k1 + 1.0)
                / (frequencies + norm)
            )
            scores[docs] += weight * contribution


class NumpyBM25Index:
    def __init__(
        self,
        root: Path,
        manifest: dict[str, Any],
        *,
        mmap: bool,
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.doc_count = int(manifest["doc_count"])
        self.fields = {
            str(metadata["field_name"]): IndexedBM25Field(root, metadata, mmap)
            for metadata in manifest["fields"]
        }

    def close(self) -> None:
        for field in self.fields.values():
            field.close()

    @classmethod
    def load(
        cls,
        root: Path,
        *,
        corpus_path: Path | None = None,
        validate_corpus_hash: bool = True,
        mmap: bool = True,
    ) -> "NumpyBM25Index":
        root = root.resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise IndexedBM25Error(f"Indexed BM25 manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "COMPLETE":
            raise IndexedBM25Error("Indexed BM25 manifest is incomplete")
        if manifest.get("index_format_version") != INDEX_FORMAT_VERSION:
            raise IndexedBM25Error("Indexed BM25 format version mismatch")
        corpus = manifest.get("corpus") or {}
        if corpus_path is not None:
            resolved_corpus = corpus_path.resolve()
            if not resolved_corpus.exists():
                raise IndexedBM25Error(f"Corpus does not exist: {resolved_corpus}")
            if int(corpus.get("bytes", -1)) != resolved_corpus.stat().st_size:
                raise IndexedBM25Error("Indexed BM25 corpus byte-size mismatch")
            expected_hash = corpus.get("sha256")
            if validate_corpus_hash and expected_hash:
                actual_hash = sha256_file(resolved_corpus)
                if actual_hash != expected_hash:
                    raise IndexedBM25Error("Indexed BM25 corpus SHA-256 mismatch")
        return cls(root, manifest, mmap=mmap)

    def weighted_scores(
        self,
        field_queries: list[tuple[str, float, list[str]]],
    ) -> np.ndarray:
        scores = np.zeros(self.doc_count, dtype=np.float64)
        for field_name, weight, query_tokens in field_queries:
            if not query_tokens or weight <= 0:
                continue
            field = self.fields.get(field_name)
            if field is None:
                raise IndexedBM25Error(f"Indexed BM25 field missing: {field_name}")
            field.add_scores(scores, query_tokens, weight)
        return scores

    @staticmethod
    def top_k_indices(
        scores: np.ndarray,
        top_k: int,
        *,
        include_zero: bool = True,
    ) -> np.ndarray:
        if top_k <= 0:
            return np.asarray([], dtype=np.intp)
        positive = np.flatnonzero(scores > 0)
        if len(positive):
            order = np.lexsort((positive, -scores[positive]))
            selected = positive[order[:top_k]]
        else:
            selected = np.asarray([], dtype=np.intp)
        if include_zero and len(selected) < min(top_k, len(scores)):
            selected_set = set(int(index) for index in selected)
            zeros: list[int] = []
            for index in range(len(scores)):
                if scores[index] == 0 and index not in selected_set:
                    zeros.append(index)
                    if len(selected) + len(zeros) >= top_k:
                        break
            if zeros:
                selected = np.concatenate(
                    [selected, np.asarray(zeros, dtype=np.intp)]
                )
        return selected[:top_k]


def load_indexed_bm25(
    root: Path,
    *,
    corpus_path: Path | None = None,
    validate_corpus_hash: bool = True,
    mmap: bool = True,
) -> NumpyBM25Index:
    key = (
        root.resolve(),
        corpus_path.resolve() if corpus_path is not None else None,
        validate_corpus_hash,
        mmap,
    )
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    index = NumpyBM25Index.load(
        root,
        corpus_path=corpus_path,
        validate_corpus_hash=validate_corpus_hash,
        mmap=mmap,
    )
    _INDEX_CACHE[key] = index
    return index


def clear_indexed_bm25_cache() -> None:
    for index in _INDEX_CACHE.values():
        index.close()
    _INDEX_CACHE.clear()
