#!/usr/bin/env python3
"""Answer retrieval helpers, candidate selection, and payload builders.

The web demo's production orchestration lives in ``recommendation_pipeline``.
This module keeps the shared retrieval wrapper, answer-candidate selection, and
JSON payload construction used by the API and eval scripts.
"""

from __future__ import annotations

import argparse
import re
from typing import Any

import pandas as pd

from rag.pipeline.retrieve_recipes import (
    build_candidates,
    build_structured_query,
    rerank_candidates,
)
from rag.pipeline.bm25_search import load_corpus
from rag.pipeline.llm_service import (
    build_ingredient_comparison,
    compute_feasibility,
)
from rag.pipeline.constraint_checker import (
    CONSTRAINT_RULES,
    apply_hard_constraint_gate,
    contains_term,
    has_hard_constraint,
    normalize_constraints,
    to_checker_recipe,
)
from rag.pipeline.request_normalization import (
    normalize_term,
    normalized_text,
)

FIELD_ALIGNMENT_COLUMNS = [
    "title",
    "title_text",
    "metadata_text",
    "normalized_meal_type",
    "normalized_dish_type",
    "normalized_method_tags",
    "normalized_diet_tags",
    "normalized_cost_tags",
    "normalized_cuisine_tags",
    "normalized_context_tags",
]
METHOD_INCLUDE_CONFLICTS = CONSTRAINT_RULES.get("method_include_conflicts", {})
# Final recipe selection only considers the top retrieved candidates. Picking a very
# deep candidate (e.g. rank 80) is not defensible and was a source of bad picks when a
# soft score on a hallucinated dish_type out-weighed a strong top-ranked match.
ANSWER_SELECTION_DEPTH = 30

MEAL_TYPE_STRUCTURED_MATCH_BONUS = 1.5
MEAL_TYPE_TEXT_MATCH_BONUS = 0.2
DISH_INTENT_MATCH_BONUS = 2.0
DISH_INTENT_MISSING_PENALTY = 2.5
DISH_NAME_TITLE_MATCH_BONUS = 1.0
DISH_NAME_MISSING_PENALTY = 0.25
DIFFICULTY_MATCH_BONUS = 0.35
COST_STRUCTURED_MATCH_BONUS = 0.6
COST_TEXT_MATCH_BONUS = 0.4
MAX_TIME_MATCH_BONUS = 0.9
MAX_TIME_MISSING_PENALTY = 1.0
DIET_STRUCTURED_MATCH_BONUS = 0.8
DIET_TEXT_MATCH_BONUS = 0.5
DIET_MISSING_PENALTY = 0.4
CUISINE_STRUCTURED_MATCH_BONUS = 0.7
CUISINE_TEXT_MATCH_BONUS = 0.6
CUISINE_MISSING_PENALTY = 0.2
METHOD_CONFLICT_ALIGNMENT_PENALTY = 2.5
METHOD_STRUCTURED_MATCH_BONUS = 1.0
METHOD_TEXT_MATCH_BONUS = 0.7
METHOD_MISSING_PENALTY = 1.0
HARD_CONSTRAINT_ALIGNMENT_PENALTY = 3.0

FEASIBILITY_RANKS = {
    "cookable_as_is": 3.0,
    "cookable_with_minor_adjustment": 2.0,
    "not_recommended": 0.0,
}
RANK_PRIOR_TOP_10 = 0.8
RANK_PRIOR_TOP_30 = 0.4
RANK_PRIOR_TOP_80 = 0.0
RANK_PRIOR_TOP_120 = -0.5
RANK_PRIOR_DEEP = -1.0
NO_USER_INGREDIENT_MATCH_PENALTY = 4.0
MISSING_REQUIRED_METHOD_SELECTION_PENALTY = 5.0
METHOD_CONFLICT_SELECTION_PENALTY = 7.0
MISSING_CUISINE_SELECTION_PENALTY = 2.0
MISSING_MEAL_TYPE_SELECTION_PENALTY = 1.5
MISSING_DISH_INTENT_SELECTION_PENALTY = 3.0
ALIGNMENT_SELECTION_WEIGHT = 3.0
AVAILABLE_MATCH_SELECTION_WEIGHT = 2.0
INGREDIENT_COVERAGE_SELECTION_WEIGHT = 4.0
METADATA_MATCH_SELECTION_WEIGHT = 0.8
FEASIBILITY_RANK_SELECTION_WEIGHT = 0.5
CRITICAL_MISSING_SELECTION_PENALTY = 0.7
MISSING_INGREDIENT_SELECTION_PENALTY = 0.5
STRUCTURED_FOOD_ROLE_SELECTION_BONUS = 1.2
RETRIEVAL_SCORE_SELECTION_WEIGHT = 0.5


# Text helpers used by candidate alignment.
def candidate_field_text(recipe: dict[str, Any], columns: list[str] | None = None) -> str:
    selected_columns = columns or FIELD_ALIGNMENT_COLUMNS
    return " ".join(normalized_text(recipe.get(column, "")) for column in selected_columns)


def text_has_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = normalized_text(phrase)
    return bool(normalized_phrase) and normalized_phrase in text


def any_phrase_matches(text: str, phrases: list[str]) -> bool:
    return any(text_has_phrase(text, phrase) for phrase in phrases)


def split_normalized_values(value: Any) -> set[str]:
    text = normalized_text(value)
    if not text:
        return set()
    return {part.strip() for part in re.split(r"\s*\|\s*|\s*,\s*|\s*;\s*", text) if part.strip()}


def normalized_column_values(recipe: dict[str, Any], column: str) -> set[str]:
    return split_normalized_values(recipe.get(column, ""))


def normalized_values_match(values: set[str], targets: list[str]) -> bool:
    normalized_targets = {normalize_term(target) for target in targets if normalize_term(target)}
    return bool(values & normalized_targets)


def recipe_has_structured_food_role(recipe: dict[str, Any]) -> bool:
    return bool(
        normalized_column_values(recipe, "normalized_meal_type")
        or normalized_column_values(recipe, "normalized_dish_type")
    )


def should_prefer_structured_food_role(
    intent: dict[str, Any],
    constraints: dict[str, Any] | None,
    structured_query: dict[str, Any],
) -> bool:
    constraints = constraints if isinstance(constraints, dict) else {}
    explicit_title_terms = structured_query.get("title_terms") if isinstance(structured_query.get("title_terms"), list) else []
    auto_title_terms = (
        structured_query.get("auto_title_terms") if isinstance(structured_query.get("auto_title_terms"), list) else []
    )
    if normalize_term(intent.get("dish_name") or "") or explicit_title_terms or auto_title_terms:
        return False
    return bool(
        normalize_term(intent.get("meal_type") or "")
        or normalize_term(intent.get("dish_type") or "")
        or normalize_term(constraints.get("cuisine") or "")
        or normalize_term(constraints.get("diet") or "")
    )


def method_include_conflicts_with_recipe(recipe: dict[str, Any], method: str) -> bool:
    rule = METHOD_INCLUDE_CONFLICTS.get(normalize_term(method))
    if not isinstance(rule, dict):
        return False
    conflict_terms = [str(term).strip() for term in rule.get("conflicting_recipe_terms", []) if str(term).strip()]
    if not conflict_terms:
        return False
    conflict_text = candidate_field_text(recipe, ["instruction_text", "method_tags_text"])
    has_conflict = any(contains_term(conflict_text, term) for term in conflict_terms)
    return has_conflict


def detect_dish_intents(query: str, intent: dict[str, Any] | None = None) -> list[str]:
    intent = intent if isinstance(intent, dict) else {}
    detected: list[str] = []
    dish_type = normalize_term(intent.get("dish_type") or "")
    meal_type = normalize_term(intent.get("meal_type") or "")
    if dish_type and dish_type != meal_type:
        detected.append(dish_type)
    focus_terms = intent.get("main_ingredient_focus", [])
    if isinstance(focus_terms, list):
        for term in focus_terms:
            normalized_focus = normalize_term(term)
            if normalized_focus:
                detected.append(normalized_focus)
    dish_name = normalize_term(intent.get("dish_name") or "")
    if dish_name:
        detected.append(dish_name)
    return list(dict.fromkeys(detected))

def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value

# Retrieval wrapper shared by runtime and eval scripts.
def retrieval_args_from_generation_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        input=args.input,
        mode=args.mode,
        candidate_k=args.candidate_k,
        bm25_index_dir=getattr(args, "bm25_index_dir", None),
        qdrant_path=args.qdrant_path,
        qdrant_url=args.qdrant_url,
        collection=args.collection,
        vector_model=args.vector_model,
        local_files_only=args.local_files_only,
        title_weight=args.title_weight,
        ingredient_weight=args.ingredient_weight,
        metadata_weight=args.metadata_weight,
        bm25_metadata_field=getattr(args, "bm25_metadata_field", "metadata_text"),
        no_auto_title_terms=getattr(args, "no_auto_title_terms", False),
        structured_bm25=getattr(args, "structured_bm25", False),
    )


# Weighted-RRF runtime mode constants (mirror the eval harness: baseline/run_hybrid_v3.py).
_RRF_K = 60
_RRF_VECTOR_WEIGHT = 2.0
_RRF_BM25_WEIGHT = 1.0


def _weighted_rrf_reorder(reranked: pd.DataFrame) -> pd.DataFrame:
    """Reorder hybrid candidates by weighted Reciprocal Rank Fusion of the vector and BM25
    score ranks: 2/(60+rank_vec) + 1/(60+rank_bm25). A doc contributes from a backend only if
    that backend scored it (>0), matching the eval-harness rrf_rank (run_hybrid_v3). This is the
    runtime weighted_rrf mode; downstream hard-constraint gate + selection run on this order."""
    if reranked.empty or "vector_score" not in reranked.columns or "bm25_score" not in reranked.columns:
        return reranked
    df = reranked.copy()
    df["doc_id"] = df["doc_id"].astype(str)
    vec = df[df["vector_score"].fillna(0) > 0].sort_values("vector_score", ascending=False)["doc_id"].tolist()
    bm = df[df["bm25_score"].fillna(0) > 0].sort_values("bm25_score", ascending=False)["doc_id"].tolist()
    vrank = {d: i + 1 for i, d in enumerate(vec)}
    brank = {d: i + 1 for i, d in enumerate(bm)}

    def rrf(doc_id: str) -> float:
        s = 0.0
        if doc_id in vrank:
            s += _RRF_VECTOR_WEIGHT / (_RRF_K + vrank[doc_id])
        if doc_id in brank:
            s += _RRF_BM25_WEIGHT / (_RRF_K + brank[doc_id])
        return s

    df["rrf_score"] = df["doc_id"].map(rrf)
    # Deterministic tie-break matching the eval harness rrf_rank (run_hybrid_v3):
    # rrf_score desc, then vector_rank asc, bm25_rank asc, doc_id asc.
    df["_vr"] = df["doc_id"].map(lambda d: vrank.get(d, 10**9))
    df["_br"] = df["doc_id"].map(lambda d: brank.get(d, 10**9))
    return (
        df.sort_values(["rrf_score", "_vr", "_br", "doc_id"], ascending=[False, True, True, True])
        .drop(columns=["_vr", "_br"])
        .reset_index(drop=True)
    )


def retrieve_candidates(
    args: argparse.Namespace,
    retrieval_inputs: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    structured_query = build_structured_query(
        retrieval_inputs["query"],
        retrieval_inputs["ingredients"],
        retrieval_inputs["intent"],
        retrieval_inputs["constraints"],
        retrieval_inputs.get("must_use_ingredients", ""),
        retrieval_inputs.get("title_intent", ""),
        getattr(args, "pantry_ingredients", ""),
        retrieval_inputs.get("excluded_metadata_terms", ""),
    )
    search_query = str(structured_query["search_query"])
    if not search_query:
        raise ValueError("Empty retrieval query after request extraction.")

    df = load_corpus(args.input.resolve())
    ret_args = retrieval_args_from_generation_args(args)
    rrf_mode = getattr(args, "mode", "") == "weighted_rrf"
    candidates = build_candidates(df, search_query, ret_args, structured_query)
    reranked = rerank_candidates(
        candidates,
        structured_query,
        ingredient_coverage_weight=args.ingredient_coverage_weight,
        metadata_match_weight=args.metadata_match_weight,
        title_match_weight=args.title_match_weight,
        missing_ingredient_penalty=args.missing_ingredient_penalty,
        missing_metadata_penalty=args.missing_metadata_penalty,
        vector_weight=args.vector_weight,
    )
    if rrf_mode:
        reranked = _weighted_rrf_reorder(reranked)
    return reranked, structured_query


# Candidate alignment and selection.
def score_candidate_alignment(
    recipe: dict[str, Any],
    query: str,
    intent: dict[str, Any] | None,
    constraints: dict[str, Any] | None,
    computed: dict[str, Any],
) -> dict[str, Any]:
    intent = intent if isinstance(intent, dict) else {}
    constraints = normalize_constraints(constraints or {})
    candidate_text = candidate_field_text(recipe)
    alignment_score = 0.0
    matched: list[str] = []
    missing: list[str] = []
    penalties: list[str] = []
    dish_intents = detect_dish_intents(query, intent)

    meal_type = normalize_term(intent.get("meal_type") or "")
    if meal_type:
        normalized_meals = normalized_column_values(recipe, "normalized_meal_type")
        if normalized_meals and normalized_values_match(normalized_meals, [meal_type]):
            alignment_score += MEAL_TYPE_STRUCTURED_MATCH_BONUS
            matched.append(f"meal_type:{meal_type}")
        elif text_has_phrase(candidate_text, meal_type):
            alignment_score += MEAL_TYPE_TEXT_MATCH_BONUS
            matched.append(f"meal_type:{meal_type}:text_fallback")

    for dish_intent in dish_intents:
        targets = [dish_intent]
        if dish_intent.endswith(" dish"):
            targets.append(dish_intent[: -len(" dish")].strip())
        targets = list(dict.fromkeys(term for term in targets if term))
        ingredient_text = candidate_field_text(recipe, ["normalized_ingredient_terms", "canonical_ingredients_text", "ingredient_text"])
        category_text = candidate_field_text(recipe, ["normalized_meal_type", "normalized_dish_type", "metadata_text"])
        dish_matched = (
            any_phrase_matches(candidate_text, targets)
            or any_phrase_matches(ingredient_text, targets)
            or any_phrase_matches(category_text, targets)
        )
        if dish_matched:
            alignment_score += DISH_INTENT_MATCH_BONUS
            matched.append(f"dish_intent:{dish_intent}")
        else:
            alignment_score -= DISH_INTENT_MISSING_PENALTY
            missing.append(f"dish_intent:{dish_intent}")

    dish_name = normalize_term(intent.get("dish_name") or "")
    if dish_name:
        title_text = candidate_field_text(recipe, ["title", "title_text"])
        if text_has_phrase(title_text, dish_name):
            alignment_score += DISH_NAME_TITLE_MATCH_BONUS
            matched.append(f"dish_name:{dish_name}")
        else:
            alignment_score -= DISH_NAME_MISSING_PENALTY
            missing.append(f"dish_name:{dish_name}")

    difficulty = normalize_term(intent.get("difficulty") or "")
    if difficulty:
        difficulty_text = candidate_field_text(recipe, ["difficulty_tags_text", "metadata_text"])
        if text_has_phrase(difficulty_text, difficulty):
            alignment_score += DIFFICULTY_MATCH_BONUS
            matched.append(f"difficulty:{difficulty}")

    cost = normalize_term(constraints.get("cost") or "")
    if cost:
        normalized_costs = normalized_column_values(recipe, "normalized_cost_tags")
        cost_text = candidate_field_text(recipe, ["cost_tags_text", "metadata_text"])
        if normalized_costs and normalized_values_match(normalized_costs, [cost]):
            alignment_score += COST_STRUCTURED_MATCH_BONUS
            matched.append(f"cost:{cost}")
        elif text_has_phrase(cost_text, cost):
            alignment_score += COST_TEXT_MATCH_BONUS
            matched.append(f"cost:{cost}:text_fallback")

    max_time = constraints.get("max_time")
    if max_time is not None and str(max_time).strip():
        try:
            max_time_minutes = float(max_time)
            total_time_minutes = float(recipe.get("TotalTime_minutes") or 0)
        except (TypeError, ValueError):
            max_time_minutes = 0.0
            total_time_minutes = 0.0
        if max_time_minutes > 0 and total_time_minutes > 0:
            if total_time_minutes <= max_time_minutes:
                alignment_score += MAX_TIME_MATCH_BONUS
                matched.append(f"max_time:{int(max_time_minutes)}")
            else:
                alignment_score -= MAX_TIME_MISSING_PENALTY
                missing.append(f"max_time:{int(max_time_minutes)}")

    diet = normalize_term(constraints.get("diet") or "")
    if diet:
        normalized_diets = normalized_column_values(recipe, "normalized_diet_tags")
        diet_text = candidate_field_text(recipe, ["diet_tags_text", "metadata_text"])
        if normalized_diets and normalized_values_match(normalized_diets, [diet]):
            alignment_score += DIET_STRUCTURED_MATCH_BONUS
            matched.append(f"diet:{diet}")
        elif text_has_phrase(diet_text, diet):
            alignment_score += DIET_TEXT_MATCH_BONUS
            matched.append(f"diet:{diet}:text_fallback")
        else:
            alignment_score -= DIET_MISSING_PENALTY
            missing.append(f"diet:{diet}")

    cuisine = normalize_term(constraints.get("cuisine") or "")
    if cuisine:
        normalized_cuisines = normalized_column_values(recipe, "normalized_cuisine_tags")
        cuisine_text = candidate_field_text(recipe, ["cuisine_tags_text", "metadata_text", "title"])
        if normalized_cuisines and normalized_values_match(normalized_cuisines, [cuisine]):
            alignment_score += CUISINE_STRUCTURED_MATCH_BONUS
            matched.append(f"cuisine:{cuisine}")
        elif text_has_phrase(cuisine_text, cuisine):
            alignment_score += CUISINE_TEXT_MATCH_BONUS
            matched.append(f"cuisine:{cuisine}:text_fallback")
        else:
            alignment_score -= CUISINE_MISSING_PENALTY
            missing.append(f"cuisine:{cuisine}")

    method_text = candidate_field_text(recipe, ["method_tags_text", "metadata_text", "instruction_text"])
    normalized_methods = normalized_column_values(recipe, "normalized_method_tags")
    for method in constraints.get("method_include", []):
        normalized_method = normalize_term(method)
        if method_include_conflicts_with_recipe(recipe, normalized_method):
            alignment_score -= METHOD_CONFLICT_ALIGNMENT_PENALTY
            missing.append(f"method_include:{normalized_method}:conflict")
            penalties.append(f"method_include_conflict:{normalized_method}")
        elif normalized_methods and normalized_values_match(normalized_methods, [normalized_method]):
            alignment_score += METHOD_STRUCTURED_MATCH_BONUS
            matched.append(f"method_include:{normalized_method}")
        elif text_has_phrase(method_text, normalized_method):
            alignment_score += METHOD_TEXT_MATCH_BONUS
            matched.append(f"method_include:{normalized_method}:text_fallback")
        else:
            alignment_score -= METHOD_MISSING_PENALTY
            missing.append(f"method_include:{normalized_method}")

    if computed["constraint_check"]["has_hard_violation"]:
        alignment_score -= HARD_CONSTRAINT_ALIGNMENT_PENALTY
        penalties.append("hard_constraint_violation")

    return {
        "alignment_score": alignment_score,
        "matched_alignment": matched,
        "missing_alignment": missing,
        "alignment_penalties": penalties,
        "missing_required_method_count": sum(1 for item in missing if item.startswith("method_include:")),
        "method_include_conflict_count": sum(1 for item in penalties if item.startswith("method_include_conflict:")),
        "missing_cuisine_count": sum(1 for item in missing if item.startswith("cuisine:")),
        "missing_meal_type_count": sum(1 for item in missing if item.startswith("meal_type:")),
        "missing_dish_intent_count": sum(1 for item in missing if item.startswith("dish_intent:")),
    }


def select_answer_candidate(
    reranked: pd.DataFrame,
    query: str,
    user_ingredients: str,
    pantry_ingredients: str,
    answer_candidate_k: int,
    intent: dict[str, Any] | None = None,
    constraints: dict[str, Any] | None = None,
    structured_query: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    best_candidate: tuple[tuple[float, ...], dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
    top_candidates_audit: list[dict[str, Any]] = []
    intent = intent if isinstance(intent, dict) else {}
    structured_query = structured_query if isinstance(structured_query, dict) else {}
    explicit_title_terms = structured_query.get("title_terms") if isinstance(structured_query.get("title_terms"), list) else []
    auto_title_terms = structured_query.get("auto_title_terms") if isinstance(structured_query.get("auto_title_terms"), list) else []
    title_match_required = bool(normalize_term(intent.get("dish_name") or "") or explicit_title_terms or auto_title_terms)
    prefer_structured_food_role = should_prefer_structured_food_role(intent, constraints, structured_query)

    # Deterministic HARD-CONSTRAINT GATE over the FULL candidate pool, BEFORE truncating to
    # ANSWER_SELECTION_DEPTH — so a valid recipe beyond the top-30 is not lost behind violators
    # (this is the benchmarked retrieval gate; same checker as run_gate_rrf via to_checker_recipe).
    pool = reranked.head(answer_candidate_k)
    ordered_recs = []
    for orig_rank, (_, row) in enumerate(pool.iterrows(), start=1):
        rec = to_checker_recipe(row.fillna("").to_dict())
        rec["original_retrieval_rank"] = orig_rank  # rank in the pre-gate retrieval pool
        ordered_recs.append(rec)
    survivor_recs, rejected = apply_hard_constraint_gate(ordered_recs, constraints, query)
    gate_summary = {
        "considered_count": len(ordered_recs),
        "rejected_count": len(rejected),
        "survivor_count": len(survivor_recs),
        "violated_constraint_types": sorted({f.split(":", 1)[0] for r in rejected for f in r["violated_fields"]}),
    }
    if has_hard_constraint(constraints, query) and not survivor_recs:
        # No safe candidate: NEVER serve a checker-flagged violator; report explicitly and let
        # the caller skip answer generation. No auto-relaxing of the user's constraints.
        return None, None, {"status": "no_safe_candidate", "gate_summary": gate_summary,
                            "candidate_selection_reason": {"no_safe_candidate": True, **gate_summary}}

    # selection runs ONLY on the survivors (top-N by backend score; order preserved by the gate).
    # hard_constraint_score stays first in the priority tuple below as a cheap 2nd-defence layer
    # (it is 1.0 for every survivor here, so it is a no-op among them).
    for rank, survivor in enumerate(survivor_recs[:ANSWER_SELECTION_DEPTH], start=1):
        recipe = dict(survivor)  # copy: do not mutate the gate's survivor object
        # rank = position among survivors (drives rank_score below). Keep BOTH ranks for audit:
        # original_retrieval_rank (pre-gate pool position) so a survivor that was originally
        # e.g. rank 31 is not silently shown as rank 1; gated_rank = survivor position;
        # retrieval_rank kept as a backward-compat alias for gated_rank.
        recipe["gated_rank"] = rank
        recipe["retrieval_rank"] = rank
        comparison = build_ingredient_comparison(user_ingredients, pantry_ingredients, recipe)
        computed = compute_feasibility(query, recipe, comparison, constraints)
        alignment = score_candidate_alignment(recipe, query, intent, constraints, computed)
        feasibility_rank = FEASIBILITY_RANKS.get(str(computed["computed_feasibility"]), 0.0)
        hard_constraint_score = 0.0 if computed["constraint_check"]["has_hard_violation"] else 1.0
        missing_count = len(computed["computed_missing_core_ingredients"])
        critical_missing_count = len(computed["computed_critical_missing_ingredients"])
        available_match_count = len(comparison.get("matched_core_ingredients", []))
        ingredient_coverage = float(recipe.get("ingredient_coverage") or 0.0)
        metadata_match = float(recipe.get("metadata_match") or 0.0)
        retrieval_score = float(recipe.get("final_score") or 0.0)
        rank_score = -float(rank)
        structured_food_role = 1.0 if recipe_has_structured_food_role(recipe) else 0.0
        structured_food_role_gate = 1.0 if not prefer_structured_food_role or structured_food_role > 0 else 0.0
        if rank <= 10:
            rank_prior = RANK_PRIOR_TOP_10
        elif rank <= 30:
            rank_prior = RANK_PRIOR_TOP_30
        elif rank <= 80:
            rank_prior = RANK_PRIOR_TOP_80
        elif rank <= 120:
            rank_prior = RANK_PRIOR_TOP_120
        else:
            rank_prior = RANK_PRIOR_DEEP
        user_ingredient_count = len(comparison.get("available_core_ingredients", []))
        has_user_ingredients = user_ingredient_count > 0
        no_user_ingredient_match_penalty = (
            NO_USER_INGREDIENT_MATCH_PENALTY if has_user_ingredients and available_match_count == 0 else 0.0
        )
        required_method_penalty = (
            float(alignment["missing_required_method_count"]) * MISSING_REQUIRED_METHOD_SELECTION_PENALTY
        )
        method_conflict_penalty = (
            float(alignment["method_include_conflict_count"]) * METHOD_CONFLICT_SELECTION_PENALTY
        )
        cuisine_penalty = float(alignment["missing_cuisine_count"]) * MISSING_CUISINE_SELECTION_PENALTY
        meal_type_penalty = float(alignment["missing_meal_type_count"]) * MISSING_MEAL_TYPE_SELECTION_PENALTY
        # dish_type is hallucination-prone, so it gets a modest soft penalty only —
        # it must not out-weigh a strong cuisine + ingredient match (see vi_vietnamese_easy).
        dish_intent_penalty = (
            float(alignment["missing_dish_intent_count"]) * MISSING_DISH_INTENT_SELECTION_PENALTY
        )
        ingredient_match_gate = 1.0 if not has_user_ingredients or available_match_count > 0 else 0.0
        title_match_gate = 1.0 if not title_match_required or str(recipe.get("matched_title") or "").strip() else 0.0
        required_method_gate = (
            1.0
            if alignment["missing_required_method_count"] == 0 and alignment["method_include_conflict_count"] == 0
            else 0.0
        )
        cuisine_gate = 1.0 if alignment["missing_cuisine_count"] == 0 else 0.0
        dish_intent_gate = 1.0 if alignment["missing_dish_intent_count"] == 0 else 0.0
        feasible_candidate_gate = 1.0 if feasibility_rank > 0 else 0.0
        selection_score = (
            alignment["alignment_score"] * ALIGNMENT_SELECTION_WEIGHT
            + float(available_match_count) * AVAILABLE_MATCH_SELECTION_WEIGHT
            + ingredient_coverage * INGREDIENT_COVERAGE_SELECTION_WEIGHT
            + metadata_match * METADATA_MATCH_SELECTION_WEIGHT
            + feasibility_rank * FEASIBILITY_RANK_SELECTION_WEIGHT
            - float(critical_missing_count) * CRITICAL_MISSING_SELECTION_PENALTY
            - float(missing_count) * MISSING_INGREDIENT_SELECTION_PENALTY
            - no_user_ingredient_match_penalty
            - required_method_penalty
            - method_conflict_penalty
            - cuisine_penalty
            - meal_type_penalty
            - dish_intent_penalty
            + structured_food_role * STRUCTURED_FOOD_ROLE_SELECTION_BONUS
            + retrieval_score * RETRIEVAL_SCORE_SELECTION_WEIGHT
            + rank_prior
        )
        # Dish intent is a strong soft preference, not a hard gate. A missing
        # dish intent lowers both alignment and selection score, while still
        # allowing ingredient-feasible recipes to survive if the extractor's
        # dish_type is too specific or wrong.
        priority = (
            hard_constraint_score,
            required_method_gate,
            title_match_gate,
            cuisine_gate,
            structured_food_role_gate,
            ingredient_match_gate,
            selection_score,
            feasible_candidate_gate,
            feasibility_rank,
            alignment["alignment_score"],
            rank_score,
        )
        computed["candidate_selection_reason"] = {
            "retrieval_rank": rank,
            "gated_rank": rank,
            "original_retrieval_rank": recipe.get("original_retrieval_rank"),
            "priority": list(priority),
            "selection_score": selection_score,
            "feasibility_rank": feasibility_rank,
            "alignment_score": alignment["alignment_score"],
            "matched_alignment": alignment["matched_alignment"],
            "missing_alignment": alignment["missing_alignment"],
            "alignment_penalties": alignment["alignment_penalties"],
            "missing_required_method_count": alignment["missing_required_method_count"],
            "method_include_conflict_count": alignment["method_include_conflict_count"],
            "missing_cuisine_count": alignment["missing_cuisine_count"],
            "missing_meal_type_count": alignment["missing_meal_type_count"],
            "missing_dish_intent_count": alignment["missing_dish_intent_count"],
            "no_user_ingredient_match_penalty": no_user_ingredient_match_penalty,
            "required_method_penalty": required_method_penalty,
            "method_conflict_penalty": method_conflict_penalty,
            "cuisine_penalty": cuisine_penalty,
            "meal_type_penalty": meal_type_penalty,
            "dish_intent_penalty": dish_intent_penalty,
            "ingredient_match_gate": ingredient_match_gate,
            "title_match_gate": title_match_gate,
            "required_method_gate": required_method_gate,
            "cuisine_gate": cuisine_gate,
            "dish_intent_gate": dish_intent_gate,
            "structured_food_role_gate": structured_food_role_gate,
            "structured_food_role": structured_food_role,
            "prefer_structured_food_role": prefer_structured_food_role,
            "feasible_candidate_gate": feasible_candidate_gate,
            "hard_constraint_score": hard_constraint_score,
            "missing_count": missing_count,
            "critical_missing_count": critical_missing_count,
            "available_match_count": available_match_count,
            "ingredient_coverage": ingredient_coverage,
            "metadata_match": metadata_match,
            "retrieval_score": retrieval_score,
            "rank_prior": rank_prior,
        }
        if rank <= 10:
            top_candidates_audit.append(
                {
                    "rank": rank,
                    "title": recipe.get("title", ""),
                    "source": recipe.get("candidate_source", ""),
                    "cuisine": recipe.get("normalized_cuisine_tags", ""),
                    "matched_title": recipe.get("matched_title", ""),
                    "feasibility": computed["computed_feasibility"],
                    "matched_core_ingredients": comparison.get("matched_core_ingredients", []),
                    "missing_core_ingredients": computed["computed_missing_core_ingredients"],
                    "critical_missing_ingredients": computed["computed_critical_missing_ingredients"],
                    "hard_constraint_violations": computed["constraint_check"].get("hard_constraint_violations", []),
                    "selection_score": selection_score,
                    "alignment_score": alignment["alignment_score"],
                    "matched_alignment": alignment["matched_alignment"],
                    "missing_alignment": alignment["missing_alignment"],
                    "alignment_penalties": alignment["alignment_penalties"],
                    "structured_food_role": structured_food_role,
                    "structured_food_role_gate": structured_food_role_gate,
                    "rank_prior": rank_prior,
                    "retrieval_score": retrieval_score,
                    "ingredient_coverage": ingredient_coverage,
                    "metadata_match": metadata_match,
                }
            )
        if best_candidate is None or priority > best_candidate[0]:
            best_candidate = (priority, recipe, comparison, computed)

    if best_candidate is None:
        raise ValueError("No retrieved candidates available for answer generation.")
    _, recipe, comparison, computed = best_candidate
    computed["top_candidates_audit"] = top_candidates_audit
    # Surface the gate audit on the success path too (not just no_safe_candidate), so the
    # transparent-output layer can report how many candidates the hard-constraint gate rejected.
    computed["gate_summary"] = gate_summary
    return recipe, comparison, computed


# Payload builders.
def build_final_payload(
    extracted: dict[str, Any],
    structured_query: dict[str, Any],
    recipe: dict[str, Any],
    comparison: dict[str, Any],
    computed: dict[str, Any],
    raw_answer: str,
    parsed_answer: dict[str, Any] | None,
    validation_issues: list[str],
    final_answer: dict[str, Any] | None,
    timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    payload = {
        "extracted_request": extracted,
        "retrieval": {
            "search_query": structured_query.get("search_query"),
            "ingredients": structured_query.get("ingredients"),
            "metadata_terms": structured_query.get("metadata_terms"),
            "title_terms": structured_query.get("title_terms"),
            "selected_doc_id": recipe.get("doc_id"),
            "selected_rank": recipe.get("retrieval_rank"),
            "selected_gated_rank": recipe.get("gated_rank"),
            "selected_original_retrieval_rank": recipe.get("original_retrieval_rank"),
            "selected_title": recipe.get("title"),
            "selected_normalized_cuisine_tags": recipe.get("normalized_cuisine_tags"),
            "candidate_selection_reason": computed.get("candidate_selection_reason"),
            "candidate_source": recipe.get("candidate_source"),
            "final_score": recipe.get("final_score"),
            "bm25_score": recipe.get("bm25_score"),
            "vector_score": recipe.get("vector_score"),
        },
        "selected_recipe_context": {
            "canonical_ingredients_text": recipe.get("canonical_ingredients_text", ""),
            "normalized_ingredient_terms": recipe.get("normalized_ingredient_terms", ""),
            "ingredients_text": recipe.get("ingredients_text", ""),
            "instruction_text": recipe.get("instruction_text", ""),
            "metadata_text": recipe.get("metadata_text", ""),
            "primary_image_url": recipe.get("primary_image_url", ""),
        },
        "code_checks": {
            "ingredient_comparison": comparison,
            "computed_feasibility": computed,
            "validation_issues": validation_issues,
            "answer_was_repaired": bool(validation_issues),
        },
        "raw_llm_answer": raw_answer,
        "parsed_llm_answer": parsed_answer,
        "final_answer": final_answer,
    }
    if timings is not None:
        payload["timings_seconds"] = timings
    return make_json_safe(payload)


def build_no_safe_candidate_payload(
    extracted: dict[str, Any],
    structured_query: dict[str, Any],
    computed: dict[str, Any],
    timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Stable payload for the no_safe_candidate outcome: every retrieved candidate within the
    pool was flagged by the hard-constraint gate. We do NOT serve a violator and do NOT relax
    the user's constraints; extraction + gate audit summary are retained, recipe is null."""
    gs = computed.get("gate_summary", {}) if isinstance(computed, dict) else {}
    payload = {
        "status": "no_safe_candidate",
        "recipe": None,
        "considered_count": gs.get("considered_count"),
        "rejected_count": gs.get("rejected_count"),
        "violated_constraint_types": gs.get("violated_constraint_types", []),
        "message": "No recipe satisfies the requested hard constraints.",
        "extracted_request": extracted,
        "retrieval": {
            "search_query": structured_query.get("search_query"),
            "ingredients": structured_query.get("ingredients"),
            "selected_doc_id": None,
            "selected_title": None,
        },
        "code_checks": {"gate_summary": gs},
    }
    if timings is not None:
        payload["timings_seconds"] = timings
    return make_json_safe(payload)


