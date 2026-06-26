#!/usr/bin/env python3
"""Embedding query encoder for vector retrieval."""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

LOGGER = logging.getLogger("embedding_encoder")
_MODEL_CACHE: dict[tuple[str, str | None, bool], Any] = {}


def load_sentence_transformer() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is not installed") from exc
    return SentenceTransformer


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def encode_query(query: str, model_name: str, device: str | None, local_files_only: bool) -> np.ndarray:
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    cache_key = (model_name, device, local_files_only)
    model = _MODEL_CACHE.get(cache_key)
    if model is None:
        SentenceTransformer = load_sentence_transformer()
        LOGGER.info("Loading embedding model: %s", model_name)
        model = SentenceTransformer(model_name, device=device, local_files_only=local_files_only)
        _MODEL_CACHE[cache_key] = model
    vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=False)[0].astype("float32")
    return normalize_vector(vector).astype("float32", copy=False)
