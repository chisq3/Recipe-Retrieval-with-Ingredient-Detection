#!/usr/bin/env python3
"""FastAPI backend for the Recipe Assistant demo (PA1).

Two endpoints, models loaded once at startup, single local provider:
  POST /detect    multipart image  -> YOLO ingredient detections (for HITL chips)
  POST /recommend {query, ingredients[]} -> grounded recipe answer (shaped per the
                  demo_api_contract.md §3 shape)

Reuses the existing pipeline verbatim (no retrieval/gate logic re-implemented):
  - preload_runtime  warms corpus / BM25 / vector model / Qdrant caches (defined below)
  - recommendation_pipeline.run_case  runs extraction -> retrieval -> gate -> answer
  - yolo_bridge.detect_ingredients runs the detector
  - recipe_facts / ingredients_full are enriched by doc_id lookup into the corpus df
    (display-only; never touches retrieval/gate).

Prereqs to actually serve requests: Qdrant (:6333, collection recipes_bge_m3_full)
and Ollama (:11434, qwen3:1.7b) running. Run with:
  cd RAG && conda run --no-capture-output -n recipe-rag uvicorn rag.demo_api:app --port 8000
(from the RAG directory).
"""
from __future__ import annotations

import argparse
import math
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag.pipeline.bm25_index import load_indexed_bm25
from rag.pipeline.recommendation_pipeline import run_case
from rag.pipeline.bm25_search import build_field_bm25, load_corpus
from rag.pipeline.qdrant_search import create_qdrant_client, load_config
from rag.pipeline.embedding_encoder import encode_query
from rag.pipeline.yolo_bridge import detect_ingredients, load_yolo

from rag.paths import RAG_ROOT, FRONTEND_DIR

BASE = RAG_ROOT
YOLO_MODEL = RAG_ROOT / "recipe_yolo" / "runs" / "yolo11" / "weights" / "best.pt"
WEB_DIR = FRONTEND_DIR  # built React frontend served at "/" (RAG/frontend/dist)

# dataset column -> nutrition response key (the 9 real columns; no Iron/Calcium)
NUTRITION_MAP = [
    ("Calories", "calories"),
    ("ProteinContent", "protein_g"),
    ("FatContent", "fat_g"),
    ("SaturatedFatContent", "saturated_fat_g"),
    ("CholesterolContent", "cholesterol_mg"),
    ("SodiumContent", "sodium_mg"),
    ("CarbohydrateContent", "carbohydrate_g"),
    ("FiberContent", "fiber_g"),
    ("SugarContent", "sugar_g"),
]

STATE: dict[str, Any] = {}


def build_args() -> argparse.Namespace:
    """Runtime namespace consumed by recommendation_pipeline.run_case."""
    return argparse.Namespace(
        input=BASE / "outputs" / "retrieval_corpus_runtime.csv",
        model="qwen3:1.7b",
        extract_model="qwen3:1.7b",
        answer_model="qwen3:1.7b",
        endpoint="http://localhost:11434/api/chat",
        timeout=900,
        extract_num_predict=2048,
        answer_timeout=60,
        answer_num_predict=512,  # short answer JSON; smoke-tested to avoid truncation while reducing latency
        temperature=0.0,
        seed=None,
        skip_answer_llm=False,
        show_intermediate=False,
        show_timing=True,
        constrained_extraction=True,
        mode="weighted_rrf",  # eval quality-best (BM25+vector RRF); vector = faster conservative default
        candidate_k=200,
        answer_candidate_k=200,
        bm25_index_dir=BASE / "outputs" / "bm25_structured_clean_metadata_index",
        bm25_metadata_field="clean_metadata_text",  # clean index field (weighted_rrf/hybrid use BM25)
        no_auto_title_terms=True,
        qdrant_path=BASE / "outputs" / "qdrant_bge_m3_full_config",
        qdrant_url="http://localhost:6333",
        collection="recipes_bge_m3_full",
        vector_model="BAAI/bge-m3",
        local_files_only=False,
    )


def preload_runtime(args: argparse.Namespace) -> None:
    """Warm corpus / BM25 / vector model / Qdrant module caches before serving."""
    start = time.perf_counter()
    print("Preloading corpus and BM25 indexes...")
    df = load_corpus(args.input.resolve())
    if args.mode in {"bm25", "hybrid", "weighted_rrf"}:
        if args.bm25_index_dir is not None:
            load_indexed_bm25(args.bm25_index_dir, corpus_path=args.input)
        else:
            title_field = "title_text" if "title_text" in df.columns else "title"
            ingredient_field = "ingredient_text" if "ingredient_text" in df.columns else "canonical_ingredients_text"
            for field_name in [title_field, ingredient_field, "metadata_text"]:
                build_field_bm25(df, field_name)
    if args.mode in {"vector", "hybrid", "weighted_rrf"}:
        print("Preloading vector model and Qdrant client...")
        config = load_config(args.qdrant_path.resolve(), args.collection)
        model_name = args.vector_model or config.get("model", "")
        if not model_name:
            raise ValueError("No vector model specified and Qdrant config has no model field")
        encode_query("warmup recipe search", model_name, device=None, local_files_only=args.local_files_only)
        create_qdrant_client(args.qdrant_path, args.qdrant_url)
    print(f"Preload finished in {time.perf_counter() - start:.1f}s")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    args = build_args()
    preload_runtime(args)  # warms corpus/BM25/vector/Qdrant module caches
    df = load_corpus(args.input.resolve())  # cached -> instant; used only for enrich
    enrich = df.set_index("doc_id", drop=False)  # doc_id is unique -> O(1) .loc
    yolo = load_yolo(YOLO_MODEL)
    STATE.update(args=args, enrich=enrich, yolo=yolo)
    print(f"Recipe Assistant API ready: {len(df):,} recipes, YOLO={YOLO_MODEL.name} (cpu)")
    yield
    STATE.clear()


app = FastAPI(title="Recipe Assistant API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local desktop demo
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "ready": bool(STATE), "recipes": int(len(STATE["enrich"])) if STATE else 0}


@app.post("/detect")
def detect(image: UploadFile = File(...)) -> dict[str, Any]:
    suffix = Path(image.filename or "").suffix or ".jpg"
    data = image.file.read()
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        detections = detect_ingredients(
            STATE["yolo"], tmp.name, conf=0.25, iou=0.45, imgsz=768, device="cpu"
        )
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    return {"detections": detections}


class RecommendRequest(BaseModel):
    query: str = ""
    ingredients: list[str] = []


@app.post("/recommend")
def recommend(req: RecommendRequest) -> dict[str, Any]:
    case = {
        "id": f"demo_{int(time.time() * 1000)}",
        "query": req.query,
        "ingredients": req.ingredients,
        "expected": {},
    }
    payload = run_case(case, STATE["args"])
    resp = shape_response(payload, STATE["enrich"])
    return resp


# --------------------------------------------------------------------------- #
# Response shaping (payload -> demo_api_contract §3 shape)
# --------------------------------------------------------------------------- #
def _num(value: Any) -> float | int | None:
    """Dataset number -> JSON number, or None for missing/NaN ('N/A' on the UI)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return int(f) if f.is_integer() else round(f, 1)


def _split_pipe(row, col) -> list[str]:
    v = row.get(col)
    return [s for s in str(v).split(" | ") if s] if isinstance(v, str) and v else []


def _clean_steps(items: list[str]) -> list[str]:
    """RecipeNLG's sentence splitter sometimes leaves orphan punctuation as its own
    step (e.g. a lone ").") because it split "...bowl. )." badly. Merge any fragment
    with no letters into the previous step so it reads as one sentence, and drop a
    leading letterless fragment."""
    out: list[str] = []
    for it in items:
        s = it.strip()
        if not s:
            continue
        if any(ch.isalpha() for ch in s):
            out.append(s)
        elif out:
            out[-1] = f"{out[-1]} {s}".strip()
        # else: leading punctuation-only fragment -> drop
    return out


def enrich_facts(doc_id: str, enrich) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    if not doc_id or doc_id not in enrich.index:
        return None, [], []
    row = enrich.loc[doc_id]
    if hasattr(row, "iloc") and getattr(row, "ndim", 1) > 1:  # duplicate index guard
        row = row.iloc[0]
    facts = {
        "time_minutes": {
            "prep": _num(row.get("PrepTime_minutes")),
            "cook": _num(row.get("CookTime_minutes")),
            "total": _num(row.get("TotalTime_minutes")),
        },
        "servings": _num(row.get("RecipeServings")),
        "rating": {
            "average": _num(row.get("AggregatedRating")),
            "review_count": _num(row.get("ReviewCount")),
        },
        "nutrition": {dst: _num(row.get(src)) for src, dst in NUTRITION_MAP},
    }
    ingredients_full = _split_pipe(row, "ingredients_display")
    instructions = _clean_steps(_split_pipe(row, "directions_display"))
    return facts, ingredients_full, instructions


def shape_response(payload: dict[str, Any], enrich) -> dict[str, Any]:
    if payload.get("status") == "out_of_scope":
        timings = payload.get("timings_seconds", {}) or {}
        return {
            "status": "out_of_scope",
            "message": payload.get("message"),
            "extraction_source": payload.get("extraction_source", "llm"),
            "evidence": {
                "timings_seconds": {k: round(float(v), 2) for k, v in timings.items()},
            },
        }

    if payload.get("status") == "no_safe_candidate":
        return {
            "status": "no_safe_candidate",
            "message": payload.get("message"),
            "considered_count": payload.get("considered_count"),
            "rejected_count": payload.get("rejected_count"),
            "violated_constraint_types": payload.get("violated_constraint_types", []),
        }

    retrieval = payload.get("retrieval", {}) or {}
    final = payload.get("final_answer", {}) or {}
    ctx = payload.get("selected_recipe_context", {}) or {}
    checks = payload.get("code_checks", {}) or {}
    computed = checks.get("computed_feasibility", {}) or {}
    gate = computed.get("gate_summary", {}) or {}
    extracted = payload.get("extracted_request", {}) or {}
    constraints = extracted.get("constraints", {}) or {}
    intent = extracted.get("intent", {}) or {}
    timings = payload.get("timings_seconds", {}) or {}

    doc_id = retrieval.get("selected_doc_id")
    facts, ingredients_full, instructions = enrich_facts(doc_id, enrich)

    return {
        "status": "ok",
        "extraction_source": payload.get("extraction_source", "llm"),
        "recipe_title": retrieval.get("selected_title") or final.get("recipe_title"),
        "feasibility": final.get("feasibility"),
        "candidate_source": retrieval.get("candidate_source"),
        "selected_rank": retrieval.get("selected_rank"),
        "cuisine_tags": retrieval.get("selected_normalized_cuisine_tags"),
        "image_url": ctx.get("primary_image_url", ""),
        "ingredients_text": ctx.get("ingredients_text", ""),
        "ingredients_full": ingredients_full,
        "instructions": instructions,
        "normalized_ingredient_terms": ctx.get("normalized_ingredient_terms", ""),
        "missing_core_ingredients": final.get("missing_core_ingredients", []) or [],
        "shopping_list": final.get("shopping_list", []) or [],
        "why_recommended": final.get("why_recommended", "") or "",
        "warning": final.get("warning", "") or "",
        "adapted_steps": final.get("adapted_steps", []) or [],
        "recipe_facts": facts,
        "evidence": {
            "understood_request": {
                "diet": constraints.get("diet"),
                "method_exclude": constraints.get("method_exclude", []) or [],
                "exclude": constraints.get("ingredient_exclude", []) or [],
                "cuisine": constraints.get("cuisine"),
                "meal_type": intent.get("meal_type"),
                "max_time": constraints.get("max_time"),
            },
            "rejected_by_gate": gate.get("rejected_count"),
            "validation_issues_repaired": len(checks.get("validation_issues") or []),
            "timings_seconds": {k: round(float(v), 2) for k, v in timings.items()},
        },
    }


# Serve the built React frontend at "/" (mounted LAST so the API routes above win).
# Guarded: if the frontend hasn't been built yet, the API still starts (UI just 404s).
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
