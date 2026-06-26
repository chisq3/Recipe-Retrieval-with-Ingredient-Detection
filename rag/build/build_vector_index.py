#!/usr/bin/env python3
"""Build a dense vector index from retrieval_corpus.csv."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("build_vector_index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dense embeddings for recipe vector retrieval")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/retrieval_corpus_runtime.csv"),
        help="Path to retrieval corpus CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/vector_index"),
        help="Directory for vector index files",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="vector_text",
        help="Text field to embed",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="SentenceTransformer model name or local path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Embedding batch size",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Optional row limit for debugging",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device for SentenceTransformer, e.g. cpu or cuda",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load embedding model from local HuggingFace cache only; do not access network",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_sentence_transformer() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Install it first, for example:\n"
            "  pip install sentence-transformers\n"
            "If network access is unavailable, install from a local wheel/cache or use a local environment "
            "that already has the package and model."
        ) from exc
    return SentenceTransformer


def load_corpus(path: Path, text_field: str, sample: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Retrieval corpus not found: {path}")

    LOGGER.info("Loading corpus: %s", path)
    nrows = sample if sample > 0 else None
    df = pd.read_csv(path, nrows=nrows)
    if text_field not in df.columns:
        raise KeyError(f"Text field '{text_field}' not found in corpus")
    if "doc_id" not in df.columns:
        raise KeyError("Corpus must contain doc_id")

    df[text_field] = df[text_field].fillna("").astype(str)
    empty_count = int((df[text_field].str.strip() == "").sum())
    if empty_count:
        LOGGER.warning("Text field '%s' has %d empty rows", text_field, empty_count)
    return df


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def build_embeddings(
    texts: list[str],
    model_name: str,
    batch_size: int,
    device: str | None,
    local_files_only: bool,
) -> np.ndarray:
    SentenceTransformer = load_sentence_transformer()
    LOGGER.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(
        model_name,
        device=device,
        local_files_only=local_files_only,
    )

    LOGGER.info("Encoding %d texts", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embeddings = embeddings.astype("float32", copy=False)
    return normalize_embeddings(embeddings).astype("float32", copy=False)


def save_index(df: pd.DataFrame, embeddings: np.ndarray, output_dir: Path, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / "embeddings.npy"
    metadata_path = output_dir / "metadata.csv"
    config_path = output_dir / "config.json"

    metadata_columns = [
        "doc_id",
        "RecipeId",
        "title",
        "title_clean",
        "link",
        "source_primary_category",
        "broad_meal_category",
        "specific_dish_type",
        "main_ingredient_category",
        "normalized_cuisine_tags",
    ]
    metadata_columns = [col for col in metadata_columns if col in df.columns]

    LOGGER.info("Writing embeddings: %s", embeddings_path)
    np.save(embeddings_path, embeddings)

    LOGGER.info("Writing metadata: %s", metadata_path)
    df[metadata_columns].to_csv(metadata_path, index=False, encoding="utf-8")

    config = {
        "input": str(args.input),
        "text_field": args.text_field,
        "model": args.model,
        "rows": int(len(df)),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
        "normalized": True,
        "local_files_only": bool(args.local_files_only),
        "files": {
            "embeddings": embeddings_path.name,
            "metadata": metadata_path.name,
        },
    }
    LOGGER.info("Writing config: %s", config_path)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    try:
        df = load_corpus(args.input.resolve(), args.text_field, args.sample)
        texts = df[args.text_field].tolist()
        embeddings = build_embeddings(texts, args.model, args.batch_size, args.device, args.local_files_only)
        save_index(df, embeddings, args.output_dir.resolve(), args)
        LOGGER.info("Vector index shape: %s", embeddings.shape)
    except Exception as exc:
        LOGGER.error("Vector index build failed: %s", exc, exc_info=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
