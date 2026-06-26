#!/usr/bin/env python3
"""BM25 tokenization, corpus loading, and field-index helpers."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

TOKEN_RE = re.compile(r"[a-z0-9]+(?:/[a-z0-9]+)?")
_CORPUS_CACHE: dict[Path, pd.DataFrame] = {}
_BM25_CACHE: dict[tuple[int, str], "SimpleBM25"] = {}


def strip_accents(text: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or "").lower())
    stripped = "".join(char for char in normalized if not unicodedata.combining(char))
    return stripped.replace("\u0111", "d").replace("\u0110", "D")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(strip_accents(text))


def normalize_query(query: str) -> str:
    return " ".join(tokenize(query))


class SimpleBM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.doc_count = len(corpus_tokens)
        self.doc_lens = [len(doc) for doc in corpus_tokens]
        self.avg_doc_len = sum(self.doc_lens) / self.doc_count if self.doc_count else 0.0
        self.term_frequencies = [Counter(doc) for doc in corpus_tokens]
        self.document_frequencies = self._build_document_frequencies()
        self.postings = self._build_postings()
        self.idf = self._build_idf()

    def _build_document_frequencies(self) -> Counter[str]:
        df: Counter[str] = Counter()
        for doc in self.corpus_tokens:
            for term in set(doc):
                df[term] += 1
        return df

    def _build_idf(self) -> dict[str, float]:
        idf: dict[str, float] = {}
        for term, freq in self.document_frequencies.items():
            idf[term] = math.log(1 + (self.doc_count - freq + 0.5) / (freq + 0.5))
        return idf

    def _build_postings(self) -> dict[str, list[tuple[int, int]]]:
        postings: dict[str, list[tuple[int, int]]] = {}
        for index, term_frequency in enumerate(self.term_frequencies):
            for term, count in term_frequency.items():
                postings.setdefault(term, []).append((index, count))
        return postings

    def score(self, query_tokens: list[str], index: int) -> float:
        score = 0.0
        if not query_tokens or self.doc_count == 0:
            return score

        tf = self.term_frequencies[index]
        doc_len = self.doc_lens[index]
        norm = self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len)) if self.avg_doc_len else self.k1

        for term in query_tokens:
            if term not in tf:
                continue
            term_freq = tf[term]
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + norm
            score += self.idf.get(term, 0.0) * (numerator / denominator)
        return score

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.doc_count
        if not query_tokens or self.doc_count == 0:
            return scores

        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, term_freq in self.postings.get(term, []):
                doc_len = self.doc_lens[index]
                norm = self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len)) if self.avg_doc_len else self.k1
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + norm
                scores[index] += idf * (numerator / denominator)
        return scores


def load_corpus(path: Path) -> pd.DataFrame:
    resolved = path.resolve()
    if resolved in _CORPUS_CACHE:
        return _CORPUS_CACHE[resolved]
    if not resolved.exists():
        raise FileNotFoundError(f"Retrieval corpus not found: {resolved}")
    df = pd.read_csv(resolved)
    _CORPUS_CACHE[resolved] = df
    return df


def build_field_bm25(df: pd.DataFrame, field_name: str) -> SimpleBM25 | None:
    if field_name not in df.columns:
        return None
    cache_key = (id(df), field_name)
    if cache_key in _BM25_CACHE:
        return _BM25_CACHE[cache_key]
    field_values = df[field_name].fillna("").astype(str).tolist()
    bm25 = SimpleBM25([tokenize(text) for text in field_values])
    _BM25_CACHE[cache_key] = bm25
    return bm25
