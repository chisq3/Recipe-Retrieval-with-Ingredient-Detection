#!/usr/bin/env python3
"""Qdrant client and search helpers for vector retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import urllib.parse
import urllib.request

_QDRANT_CLIENT_CACHE: dict[tuple[str, str], Any] = {}
_CONFIG_CACHE: dict[tuple[Path, str], dict[str, Any]] = {}


class QdrantHandle:
    def __init__(self, client: Any, url: str = "") -> None:
        self.client = client
        self.url = url.rstrip("/")


def load_qdrant_client() -> Any:
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise RuntimeError("qdrant-client is not installed. Install it first: pip install qdrant-client") from exc
    return QdrantClient


def create_qdrant_client(qdrant_path: Path, qdrant_url: str = "") -> Any:
    cache_key = (str(qdrant_path.resolve()), qdrant_url.rstrip("/"))
    if cache_key in _QDRANT_CLIENT_CACHE:
        return _QDRANT_CLIENT_CACHE[cache_key]
    QdrantClient = load_qdrant_client()
    if qdrant_url:
        client = QdrantHandle(QdrantClient(url=qdrant_url, check_compatibility=False), qdrant_url)
    else:
        client = QdrantHandle(QdrantClient(path=str(qdrant_path.resolve()), check_compatibility=False))
    _QDRANT_CLIENT_CACHE[cache_key] = client
    return client


def load_config(qdrant_path: Path, collection: str) -> dict[str, Any]:
    cache_key = (qdrant_path.resolve(), collection)
    if cache_key in _CONFIG_CACHE:
        return _CONFIG_CACHE[cache_key]
    config_path = qdrant_path / f"{collection}_config.json"
    if not config_path.exists():
        _CONFIG_CACHE[cache_key] = {}
        return _CONFIG_CACHE[cache_key]
    _CONFIG_CACHE[cache_key] = json.loads(config_path.read_text(encoding="utf-8"))
    return _CONFIG_CACHE[cache_key]


def search_qdrant_server(url: str, collection: str, query_vector: list[float], top_k: int) -> list[Any]:
    endpoint = f"{url}/collections/{urllib.parse.quote(collection)}/points/search"
    body = json.dumps(
        {
            "vector": query_vector,
            "limit": top_k,
            "with_payload": True,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [
        SimpleNamespace(
            id=item.get("id"),
            score=item.get("score", 0.0),
            payload=item.get("payload") or {},
        )
        for item in payload.get("result", [])
    ]


def search(client: Any, collection: str, query_vector: list[float], top_k: int) -> list[Any]:
    if isinstance(client, QdrantHandle) and client.url:
        return search_qdrant_server(client.url, collection, query_vector, top_k)
    qdrant_client = client.client if isinstance(client, QdrantHandle) else client
    return list(qdrant_client.search(collection_name=collection, query_vector=query_vector, limit=top_k, with_payload=True))
