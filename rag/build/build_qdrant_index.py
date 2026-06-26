#!/usr/bin/env python3
"""Build a local Qdrant vector index for recipe retrieval."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from rag.build.build_vector_index import load_sentence_transformer, normalize_embeddings

LOGGER = logging.getLogger("build_qdrant_index")

PAYLOAD_COLUMNS = [
    "doc_id",
    "RecipeId",
    "title",
    "link",
    "primary_image_url",
    "normalized_meal_type",
    "normalized_dish_type",
    "normalized_method_tags",
    "normalized_diet_tags",
    "normalized_cost_tags",
    "normalized_cuisine_tags",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Qdrant index from retrieval corpus")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/retrieval_corpus_runtime.csv"),
        help="Path to retrieval corpus CSV",
    )
    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=Path("outputs/qdrant_db"),
        help="Local Qdrant storage path, or config output directory when --qdrant-url is set",
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default="",
        help="Qdrant server URL, e.g. http://localhost:6333. If set, server mode is used.",
    )
    parser.add_argument(
        "--qdrant-timeout",
        type=float,
        default=120.0,
        help="Qdrant HTTP timeout in seconds when --qdrant-url is set.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="recipes",
        help="Qdrant collection name",
    )
    parser.add_argument("--text-field", type=str, default="vector_text", help="Text field to embed")
    parser.add_argument("--model", type=str, default="BAAI/bge-m3", help="SentenceTransformer model name/path")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--upsert-batch-size", type=int, default=256, help="Qdrant upsert batch size")
    parser.add_argument(
        "--encode-chunk-size",
        type=int,
        default=0,
        help="Number of rows to encode before each Qdrant upsert. Defaults to --upsert-batch-size.",
    )
    parser.add_argument("--sample", type=int, default=0, help="Optional row limit for debugging")
    parser.add_argument("--device", type=str, default=None, help="Optional device, e.g. cpu or cuda")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load embedding model from local HuggingFace cache only",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate collection if it already exists",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted build from the current Qdrant point count. Do not use with --recreate.",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=-1,
        help="Manual global row offset for --resume. Skips exact Qdrant count when set.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s | %(levelname)s | %(message)s")


def load_qdrant() -> tuple[Any, Any, Any]:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
    except ImportError as exc:
        raise RuntimeError(
            "qdrant-client is not installed. Install it first:\n"
            "  pip install qdrant-client"
        ) from exc
    return QdrantClient, models.Distance, models.PointStruct


def clean_payload_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    return value


def build_payloads(df: pd.DataFrame) -> list[dict[str, Any]]:
    available = [column for column in PAYLOAD_COLUMNS if column in df.columns]
    payloads: list[dict[str, Any]] = []
    for _, row in df[available].iterrows():
        payloads.append({column: clean_payload_value(row[column]) for column in available})
    return payloads


def validate_corpus_header(path: Path, text_field: str) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Retrieval corpus not found: {path}")
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    if text_field not in columns:
        raise KeyError(f"Text field '{text_field}' not found in corpus")
    if "doc_id" not in columns:
        raise KeyError("Corpus must contain doc_id")
    return columns


def iter_corpus_chunks(path: Path, chunk_size: int, sample: int) -> Any:
    remaining = sample if sample > 0 else None
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        if remaining is not None:
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
            remaining -= len(chunk)
        yield chunk


def collection_exists(client: Any, collection: str) -> bool:
    return any(item.name == collection for item in client.get_collections().collections)


def create_collection(
    client: Any,
    collection: str,
    vector_size: int,
    distance: Any,
) -> None:
    LOGGER.info("Creating collection '%s' with vector size %d", collection, vector_size)
    client.create_collection(
        collection_name=collection,
        vectors_config={"size": vector_size, "distance": distance.COSINE},
    )


def get_collection_count(client: Any, collection: str) -> int:
    result = client.count(collection_name=collection, exact=True)
    return int(result.count)


def load_embedding_model(model_name: str, device: str | None, local_files_only: bool) -> Any:
    SentenceTransformer = load_sentence_transformer()
    LOGGER.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(
        model_name,
        device=device,
        local_files_only=local_files_only,
    )


def encode_texts(model: Any, texts: list[str], batch_size: int) -> Any:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embeddings = embeddings.astype("float32", copy=False)
    return normalize_embeddings(embeddings).astype("float32", copy=False)


def upsert_points(
    client: Any,
    collection: str,
    point_cls: Any,
    embeddings: Any,
    payloads: list[dict[str, Any]],
    batch_size: int,
    id_offset: int = 0,
) -> None:
    total = len(payloads)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        points = [
            point_cls(
                id=id_offset + start + offset,
                vector=embeddings[start + offset].tolist(),
                payload=payloads[start + offset],
            )
            for offset in range(end - start)
        ]
        client.upsert(collection_name=collection, points=points)
        LOGGER.info("Upserted rows %d-%d", id_offset + start, id_offset + end - 1)


def save_config(args: argparse.Namespace, rows: int, vector_size: int) -> None:
    args.qdrant_path.mkdir(parents=True, exist_ok=True)
    config = {
        "input": str(args.input),
        "qdrant_path": str(args.qdrant_path),
        "qdrant_url": args.qdrant_url,
        "qdrant_mode": "server" if args.qdrant_url else "local_path",
        "collection": args.collection,
        "text_field": args.text_field,
        "model": args.model,
        "rows": rows,
        "vector_size": vector_size,
        "distance": "cosine",
    }
    config_path = args.qdrant_path / f"{args.collection}_config.json"
    LOGGER.info("Writing config: %s", config_path)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    try:
        if args.resume and args.recreate:
            raise ValueError("Use either --resume or --recreate, not both.")

        QdrantClient, distance, point_cls = load_qdrant()
        input_path = args.input.resolve()
        LOGGER.info("Validating corpus header: %s", input_path)
        validate_corpus_header(input_path, args.text_field)
        encode_chunk_size = args.encode_chunk_size or args.upsert_batch_size
        if encode_chunk_size <= 0:
            raise ValueError("--encode-chunk-size must be positive")

        LOGGER.info("Opening Qdrant client")
        if args.qdrant_url:
            client = QdrantClient(url=args.qdrant_url, timeout=args.qdrant_timeout)
        else:
            client = QdrantClient(path=str(args.qdrant_path.resolve()))
        LOGGER.info("Qdrant client opened")

        LOGGER.info("Checking collection status: %s", args.collection)
        exists = collection_exists(client, args.collection)
        start_row = 0
        if exists and args.recreate:
            LOGGER.info("Deleting existing collection: %s", args.collection)
            client.delete_collection(args.collection)
            exists = False
        elif exists and args.resume:
            if args.resume_from >= 0:
                start_row = args.resume_from
                LOGGER.info("Using manual resume offset: %d", start_row)
            else:
                LOGGER.info("Counting existing Qdrant points for resume")
                start_row = get_collection_count(client, args.collection)
            LOGGER.info("Resuming collection '%s' from row %d", args.collection, start_row)
        elif exists:
            raise ValueError(
                f"Collection '{args.collection}' already exists. Use --resume to continue "
                "or --recreate to rebuild it."
            )

        model = load_embedding_model(args.model, args.device, args.local_files_only)
        vector_size: int | None = None
        processed_rows = 0

        for chunk in iter_corpus_chunks(input_path, encode_chunk_size, args.sample):
            chunk_start = processed_rows
            chunk_end = processed_rows + len(chunk)
            processed_rows = chunk_end

            if chunk_end <= start_row:
                LOGGER.info("Skipping already indexed rows %d-%d", chunk_start, chunk_end - 1)
                continue

            if chunk_start < start_row:
                chunk = chunk.iloc[start_row - chunk_start :]
                chunk_start = start_row

            chunk[args.text_field] = chunk[args.text_field].fillna("").astype(str)
            texts = chunk[args.text_field].tolist()
            embeddings = encode_texts(model, texts, args.batch_size)
            vector_size = int(embeddings.shape[1])

            if not exists:
                create_collection(
                    client=client,
                    collection=args.collection,
                    vector_size=vector_size,
                    distance=distance,
                )
                exists = True

            payloads = build_payloads(chunk)
            upsert_points(
                client,
                args.collection,
                point_cls,
                embeddings,
                payloads,
                args.upsert_batch_size,
                id_offset=chunk_start,
            )
            LOGGER.info("Completed through row %d", chunk_end)

        if processed_rows <= start_row:
            LOGGER.info("Collection already contains all requested rows; nothing to do.")

        save_config(args, rows=processed_rows, vector_size=vector_size or 0)
        target = args.qdrant_url or str(args.qdrant_path)
        LOGGER.info("Qdrant index build complete: %s/%s", target, args.collection)
    except Exception as exc:
        LOGGER.error("Qdrant index build failed: %s", exc, exc_info=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
