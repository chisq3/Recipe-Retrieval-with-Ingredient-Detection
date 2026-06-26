"""Runtime orchestration for one recipe recommendation request.

This module owns the production flow used by the web API:
extract request -> normalize -> retrieve -> gate/select -> answer -> repair.
Evaluation scripts should call this module instead of hosting runtime logic.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any

from rag.pipeline.answer_builder import (
    build_final_payload,
    build_no_safe_candidate_payload,
    retrieve_candidates,
    select_answer_candidate,
)
from rag.pipeline.ingredient_normalization import RUNTIME_RULES
from rag.pipeline.request_normalization import build_retrieval_inputs, normalize_extracted_request
from rag.pipeline.llm_service import (
    answer_output_schema,
    build_answer_messages,
    build_extraction_messages,
    call_ollama,
    extract_json_object,
    extraction_output_schema,
    repair_answer_output,
    validate_answer_output,
)


def default_generation_args() -> argparse.Namespace:
    """Defaults shared by API/eval wrappers before they apply environment-specific overrides."""
    pantry = [
        str(item).strip()
        for item in RUNTIME_RULES.get("default_pantry_ingredients", [])
        if str(item).strip()
    ]
    return argparse.Namespace(
        input=Path("outputs/retrieval_corpus_runtime.csv"),
        query="",
        ingredients="",
        pantry_ingredients=", ".join(pantry),
        model="qwen3:4b",
        extract_model=None,
        answer_model=None,
        endpoint="http://localhost:11434/api/chat",
        timeout=900,
        extract_num_predict=2048,
        answer_timeout=60,
        answer_num_predict=2048,
        temperature=0.0,
        seed=None,
        skip_answer_llm=False,
        show_intermediate=False,
        show_timing=False,
        constrained_extraction=True,
        mode="vector",
        candidate_k=200,
        answer_candidate_k=200,
        bm25_index_dir=None,
        bm25_metadata_field="metadata_text",
        no_auto_title_terms=True,
        qdrant_path=Path("outputs/qdrant_bge_m3_50k"),
        qdrant_url="",
        collection="recipes_bge_m3",
        vector_model="BAAI/bge-m3",
        local_files_only=False,
        vector_weight=0.35,
        title_weight=0.25,
        ingredient_weight=0.50,
        metadata_weight=0.25,
        ingredient_coverage_weight=2.0,
        metadata_match_weight=1.0,
        title_match_weight=3.0,
        missing_ingredient_penalty=0.75,
        missing_metadata_penalty=0.75,
    )


def generation_args(eval_args: argparse.Namespace) -> argparse.Namespace:
    """Convert an eval/API namespace into the generation namespace used downstream."""
    args = default_generation_args()
    args.input = eval_args.input
    args.model = eval_args.model
    args.extract_model = eval_args.extract_model
    args.answer_model = eval_args.answer_model
    args.endpoint = eval_args.endpoint
    args.timeout = eval_args.timeout
    args.extract_num_predict = eval_args.extract_num_predict
    args.answer_timeout = eval_args.answer_timeout
    args.answer_num_predict = eval_args.answer_num_predict
    args.temperature = eval_args.temperature
    args.pantry_ingredients = getattr(eval_args, "pantry_ingredients", args.pantry_ingredients)
    args.mode = eval_args.mode
    args.candidate_k = eval_args.candidate_k
    args.answer_candidate_k = eval_args.answer_candidate_k
    args.bm25_index_dir = eval_args.bm25_index_dir
    args.bm25_metadata_field = getattr(eval_args, "bm25_metadata_field", args.bm25_metadata_field)
    args.no_auto_title_terms = eval_args.no_auto_title_terms
    args.qdrant_path = eval_args.qdrant_path
    args.qdrant_url = eval_args.qdrant_url
    args.collection = eval_args.collection
    args.vector_model = eval_args.vector_model
    args.local_files_only = eval_args.local_files_only
    args.vector_weight = getattr(eval_args, "vector_weight", args.vector_weight)
    args.title_weight = getattr(eval_args, "title_weight", args.title_weight)
    args.ingredient_weight = getattr(eval_args, "ingredient_weight", args.ingredient_weight)
    args.metadata_weight = getattr(eval_args, "metadata_weight", args.metadata_weight)
    args.ingredient_coverage_weight = getattr(
        eval_args, "ingredient_coverage_weight", args.ingredient_coverage_weight
    )
    args.metadata_match_weight = getattr(eval_args, "metadata_match_weight", args.metadata_match_weight)
    args.title_match_weight = getattr(eval_args, "title_match_weight", args.title_match_weight)
    args.missing_ingredient_penalty = getattr(
        eval_args, "missing_ingredient_penalty", args.missing_ingredient_penalty
    )
    args.missing_metadata_penalty = getattr(eval_args, "missing_metadata_penalty", args.missing_metadata_penalty)
    args.show_intermediate = eval_args.show_intermediate
    return args


def extract_case_request(
    case: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run only the production extraction and normalization stages."""
    query = str(case.get("query") or "")
    ingredients = ", ".join(case.get("ingredients", []))
    start = time.perf_counter()
    raw_extraction = call_ollama(
        endpoint=args.endpoint,
        model=args.extract_model or args.model,
        messages=build_extraction_messages(query, ingredients),
        timeout=args.timeout,
        temperature=args.temperature,
        num_predict=args.extract_num_predict,
        output_schema=extraction_output_schema()
        if getattr(args, "constrained_extraction", True)
        else None,
        seed=getattr(args, "seed", None),
    )
    llm_seconds = time.perf_counter() - start

    start = time.perf_counter()
    extracted = normalize_extracted_request(
        extract_json_object(raw_extraction),
        query,
        ingredients,
    )
    normalization_seconds = time.perf_counter() - start
    return {
        "raw_extraction": raw_extraction,
        "extracted_request": extracted,
        "timings_seconds": {
            "llm_extraction": llm_seconds,
            "request_normalization": normalization_seconds,
        },
    }


def normalize_confirmed_extracted_request(
    confirmed: dict[str, Any],
    query: str,
    ingredients: str,
) -> dict[str, Any]:
    """Sanitize the editable review payload before it reaches retrieval."""
    payload = copy.deepcopy(confirmed)
    constraints = payload.get("constraints")
    if not isinstance(constraints, dict):
        constraints = {}
        payload["constraints"] = constraints
    ingredient_exclude = constraints.get("ingredient_exclude")
    if not isinstance(ingredient_exclude, list):
        ingredient_exclude = []
        constraints["ingredient_exclude"] = ingredient_exclude
    # Older confirmed payloads may still carry a top-level exclude field.
    # The nested constraint field is canonical; dropping the legacy key prevents
    # a removed exclusion from reappearing during fallback normalization.
    payload.pop("exclude_ingredients", None)
    must_use = payload.get("must_use_ingredients")
    must_use = must_use if isinstance(must_use, list) else []
    available = payload.get("available_ingredients")
    available = available if isinstance(available, list) else []
    payload["available_ingredients"] = list(dict.fromkeys([*available, *must_use]))
    return normalize_extracted_request(payload, query, ingredients)


def has_recipe_request_signal(extracted: dict[str, Any]) -> bool:
    """Whether the normalized request contains enough recipe-domain signal to retrieve.

    The extractor returns empty structured fields for out-of-domain questions. In that
    case we stop before retrieval so the app does not force an unrelated recipe.
    """
    if any(extracted.get(key) for key in (
        "dish_name",
        "available_ingredients",
        "must_use_ingredients",
    )):
        return True

    intent = extracted.get("intent") if isinstance(extracted.get("intent"), dict) else {}
    if any(intent.get(key) for key in ("meal_type", "dish_type", "main_ingredient_focus", "difficulty")):
        return True
    if intent.get("goal"):
        return True

    constraints = extracted.get("constraints") if isinstance(extracted.get("constraints"), dict) else {}
    return any(constraints.get(key) for key in (
        "diet",
        "cuisine",
        "method_include",
        "method_exclude",
        "ingredient_exclude",
        "max_time",
        "cost",
    ))


def build_out_of_scope_payload(
    extracted: dict[str, Any],
    timings: dict[str, float],
    extraction_source: str,
) -> dict[str, Any]:
    return {
        "status": "out_of_scope",
        "message": (
            "I'm focused on recipe recommendations, so I can't help with that request. "
            "Ask me about ingredients, cooking constraints, meal ideas, or dish suggestions. "
            'For example: "tomato pasta dinner."'
        ),
        "extracted_request": extracted,
        "extraction_source": extraction_source,
        "timings_seconds": timings,
    }


def run_case(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Run one recommendation case through the production pipeline."""
    gen_args = generation_args(args)
    gen_args.query = case["query"]
    gen_args.ingredients = ", ".join(case.get("ingredients", []))
    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    confirmed = case.get("confirmed_extracted_request")
    if isinstance(confirmed, dict):
        raw_extraction = ""
        timings["llm_extraction"] = 0.0
        start = time.perf_counter()
        extracted = normalize_confirmed_extracted_request(
            confirmed,
            gen_args.query,
            gen_args.ingredients,
        )
        timings["request_normalization"] = time.perf_counter() - start
        extraction_source = "human_confirmed"
    elif not str(gen_args.query).strip() and case.get("ingredients"):
        # Pure ingredient-chip requests cannot carry hard constraints; skipping
        # extraction removes image-demo latency without changing text-query behavior.
        raw_extraction = ""
        timings["llm_extraction"] = 0.0
        start = time.perf_counter()
        extracted = normalize_confirmed_extracted_request(
            {"available_ingredients": list(case.get("ingredients", []))},
            gen_args.query,
            gen_args.ingredients,
        )
        timings["request_normalization"] = time.perf_counter() - start
        extraction_source = "structural_skip"
    else:
        extraction = extract_case_request(case, args)
        raw_extraction = extraction["raw_extraction"]
        extracted = extraction["extracted_request"]
        timings.update(extraction["timings_seconds"])
        extraction_source = "llm"

    if not has_recipe_request_signal(extracted):
        timings["retrieval_input_build"] = 0.0
        timings["retrieval"] = 0.0
        timings["candidate_selection"] = 0.0
        timings["llm_answer"] = 0.0
        timings["validate_repair"] = 0.0
        timings["total"] = time.perf_counter() - total_start
        return build_out_of_scope_payload(extracted, timings, extraction_source)

    start = time.perf_counter()
    retrieval_inputs = build_retrieval_inputs(extracted, gen_args.query, gen_args.ingredients)
    timings["retrieval_input_build"] = time.perf_counter() - start

    start = time.perf_counter()
    reranked, structured_query = retrieve_candidates(gen_args, retrieval_inputs)
    timings["retrieval"] = time.perf_counter() - start

    start = time.perf_counter()
    recipe, comparison, computed = select_answer_candidate(
        reranked,
        gen_args.query,
        retrieval_inputs["ingredients"],
        gen_args.pantry_ingredients,
        gen_args.answer_candidate_k,
        extracted.get("intent"),
        extracted.get("constraints"),
        structured_query,
    )
    timings["candidate_selection"] = time.perf_counter() - start

    if computed.get("status") == "no_safe_candidate":
        timings["llm_answer"] = 0.0
        timings["total"] = time.perf_counter() - total_start
        payload = build_no_safe_candidate_payload(
            extracted,
            structured_query,
            computed,
            timings,
        )
        payload["extraction_source"] = extraction_source
        return payload

    raw_answer = ""
    parsed_answer = None
    if not args.skip_answer_llm:
        start = time.perf_counter()
        raw_answer = call_ollama(
            endpoint=args.endpoint,
            model=args.answer_model or args.model,
            messages=build_answer_messages(
                gen_args.query,
                retrieval_inputs["ingredients"],
                gen_args.pantry_ingredients,
                recipe,
                extracted.get("constraints"),
            ),
            timeout=args.answer_timeout,
            temperature=args.temperature,
            num_predict=args.answer_num_predict,
            seed=getattr(args, "seed", None),
            output_schema=answer_output_schema(),
        )
        timings["llm_answer"] = time.perf_counter() - start
        parsed_answer = extract_json_object(raw_answer)
    else:
        timings["llm_answer"] = 0.0

    start = time.perf_counter()
    if args.skip_answer_llm:
        validation_issues = []
    else:
        validation_issues = validate_answer_output(
            parsed_answer,
            gen_args.query,
            retrieval_inputs["ingredients"],
            gen_args.pantry_ingredients,
            recipe,
            extracted.get("constraints"),
        )
    final_answer = repair_answer_output(
        parsed_answer or {},
        gen_args.query,
        retrieval_inputs["ingredients"],
        gen_args.pantry_ingredients,
        recipe,
        extracted.get("constraints"),
        validation_issues=validation_issues,
    )
    if final_answer is None:
        final_answer = {
            "recipe_title": recipe.get("title", ""),
            "feasibility": computed["computed_feasibility"],
            "available_core_ingredients": comparison["available_core_ingredients"],
            "assumed_available_pantry": comparison["assumed_available_pantry"],
            "missing_core_ingredients": computed["computed_missing_core_ingredients"],
            "missing_pantry_ingredients": computed["computed_missing_pantry_ingredients"],
            "critical_missing_ingredients": computed["computed_critical_missing_ingredients"],
            "shopping_list": computed["computed_missing_core_ingredients"],
            "substitutions": [],
            "why_recommended": "Deterministic fallback answer was used.",
            "adapted_steps": [],
            "warning": " ".join([*computed.get("blocking_reasons", []), *computed.get("warnings", [])]),
            "evidence_used": {
                "recipe_ingredients_used": recipe.get("canonical_ingredients_text", ""),
                "instruction_source": "instruction_text",
            },
        }
    timings["validate_repair"] = time.perf_counter() - start
    timings["total"] = time.perf_counter() - total_start

    payload = build_final_payload(
        extracted=extracted,
        structured_query=structured_query,
        recipe=recipe,
        comparison=comparison,
        computed=computed,
        raw_answer=raw_answer if args.show_intermediate else "",
        parsed_answer=parsed_answer if args.show_intermediate else None,
        validation_issues=validation_issues,
        final_answer=final_answer,
        timings=timings if args.show_timing else None,
    )
    payload["case_id"] = case["id"]
    payload["case_query"] = case["query"]
    payload["case_ingredients"] = case.get("ingredients", [])
    payload["extraction_source"] = extraction_source
    return payload
