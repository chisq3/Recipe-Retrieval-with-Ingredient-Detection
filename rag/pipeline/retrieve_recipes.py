#!/usr/bin/env python3
"""Recipe retrieval helpers with BM25/vector candidate gathering and reranking."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rag.pipeline.ingredient_normalization import availability_covers, normalize_recipe_terms, parse_ingredient_terms
from rag.pipeline.bm25_index import load_indexed_bm25
from rag.pipeline.bm25_search import build_field_bm25, normalize_query, tokenize

NO_AVAILABLE_INGREDIENT_PENALTY = 2.5
RECIPE_MISSING_RATIO_PENALTY = 1.5
MINOR_ADJUSTMENT_BONUS = 0.8
COOKABLE_AS_IS_BONUS = 1.2


def split_csv_terms(value: str) -> list[str]:
    terms: list[str] = []
    for part in value.split(","):
        cleaned = " ".join(tokenize(part))
        if cleaned:
            terms.append(normalize_query(cleaned))
    return list(dict.fromkeys(terms))


def normalized_terms(text: str) -> list[str]:
    normalized = normalize_query(text)
    terms = tokenize(normalized)
    return list(dict.fromkeys(terms))


def build_metadata_term_groups(terms: list[str]) -> dict[str, list[str]]:
    return {term: [term] for term in terms}


def build_structured_query(
    query: str,
    ingredients: str,
    intent: str,
    constraints: str,
    must_use_ingredients: str = "",
    title_intent: str = "",
    pantry_ingredients: str = "",
    excluded_metadata: str = "",
) -> dict[str, list[str] | str]:
    ingredient_terms = split_csv_terms(ingredients)
    must_use_terms = split_csv_terms(must_use_ingredients)
    pantry_terms = split_csv_terms(pantry_ingredients)
    constraint_terms = split_csv_terms(constraints)
    intent_terms = normalized_terms(intent)
    title_terms = split_csv_terms(title_intent)

    excluded_metadata_terms = split_csv_terms(excluded_metadata)

    metadata_terms = list(dict.fromkeys([*intent_terms, *constraint_terms]))
    metadata_terms = [term for term in metadata_terms if term not in excluded_metadata_terms]
    return {
        "raw_query": query,
        "search_query": " ".join([query, ingredients, must_use_ingredients, intent, constraints]).strip(),
        "ingredients": ingredient_terms,
        "available_ingredients": ingredient_terms,
        "must_use_ingredients": must_use_terms,
        "pantry_ingredients": pantry_terms,
        "metadata_terms": metadata_terms,
        "metadata_term_groups": build_metadata_term_groups(metadata_terms),
        "excluded_metadata_terms": excluded_metadata_terms,
        "excluded_metadata_term_groups": build_metadata_term_groups(excluded_metadata_terms),
        "title_terms": title_terms,
    }


def tokens_from_terms(terms: list[str]) -> list[str]:
    tokens: list[str] = []
    for term in terms:
        tokens.extend(tokenize(normalize_query(term)))
    return list(dict.fromkeys(tokens))


def weighted_bm25_scores_by_field(
    df: pd.DataFrame,
    title_tokens: list[str],
    ingredient_tokens: list[str],
    metadata_tokens: list[str],
    title_weight: float,
    ingredient_weight: float,
    metadata_weight: float,
) -> list[float]:
    title_field = "title_text" if "title_text" in df.columns else "title"
    ingredient_field = "ingredient_text" if "ingredient_text" in df.columns else "canonical_ingredients_text"
    weighted_fields = [
        (title_field, title_weight, title_tokens),
        (ingredient_field, ingredient_weight, ingredient_tokens),
        ("metadata_text", metadata_weight, metadata_tokens),
    ]
    scores = [0.0] * len(df)
    for field_name, weight, field_tokens in weighted_fields:
        if weight <= 0 or not field_tokens:
            continue
        bm25 = build_field_bm25(df, field_name)
        if bm25 is None:
            continue
        field_scores = bm25.get_scores(field_tokens)
        scores = [base + weight * field_score for base, field_score in zip(scores, field_scores)]
    return scores


def title_phrase_count(df: pd.DataFrame, title_field: str, phrase: str) -> int:
    pattern = rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"
    normalized_titles = df[title_field].fillna("").map(normalize_query)
    return int(normalized_titles.str.contains(pattern, regex=True).sum())


def strip_generic_title_suffix(phrase: str) -> str:
    normalized = normalize_query(phrase)
    suffixes = (" recipes", " recipe", " dishes", " dish", " meals", " meal")
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)].strip()
                changed = True
                break
    return normalized


def filter_specific_title_terms(
    df: pd.DataFrame,
    title_terms: list[str],
    *,
    max_title_frequency: int = 1000,
) -> list[str]:
    if not title_terms:
        return []
    title_field = "title_text" if "title_text" in df.columns else "title"
    filtered: list[str] = []
    for term in title_terms:
        phrase = strip_generic_title_suffix(str(term))
        if not phrase:
            continue
        title_count = title_phrase_count(df, title_field, phrase)
        if title_count == 0 or title_count > max_title_frequency:
            continue
        filtered.append(phrase)
    return list(dict.fromkeys(filtered))


def auto_title_terms_from_query(df: pd.DataFrame, raw_query: str) -> list[str]:
    tokens = tokenize(normalize_query(raw_query))
    if not tokens:
        return []

    title_field = "title_text" if "title_text" in df.columns else "title"
    bm25 = build_field_bm25(df, title_field)
    if bm25 is None:
        return []

    scores = bm25.get_scores(tokens)
    if len(scores) == 0 or max(scores) <= 0:
        return []

    top_indices = np.argsort(scores)[-30:]
    top_titles = [normalize_query(df.iloc[int(index)].get(title_field, "")) for index in top_indices if scores[index] > 0]
    candidates: list[tuple[float, str]] = []
    seen: set[str] = set()
    max_ngram = min(3, len(tokens))
    for ngram_size in range(max_ngram, 0, -1):
        for start in range(0, len(tokens) - ngram_size + 1):
            phrase = " ".join(tokens[start : start + ngram_size])
            if phrase in seen or len(phrase.replace(" ", "")) < 3:
                continue
            seen.add(phrase)
            pattern = rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"
            top_match_count = sum(bool(re.search(pattern, title)) for title in top_titles)
            if top_match_count < 3:
                continue
            if ngram_size == 1:
                title_df = bm25.document_frequencies.get(phrase, 0)
                max_single_token_df = 200 if len(phrase) == 3 else 1000
                if title_df < 50 or title_df > max_single_token_df:
                    continue
                corpus_count = title_df
            else:
                corpus_count = title_phrase_count(df, title_field, phrase)
                if corpus_count < 20 or corpus_count > 1000:
                    continue
            candidates.append((top_match_count * ngram_size + min(corpus_count / 100.0, 5.0), phrase))

    selected: list[str] = []
    for _, phrase in sorted(candidates, reverse=True):
        if any(phrase in existing or existing in phrase for existing in selected):
            continue
        selected.append(phrase)
        if len(selected) >= 2:
            break
    return selected


def bm25_field_tokens(
    df: pd.DataFrame,
    search_query: str,
    structured_query: dict[str, list[str] | str] | None,
    *,
    auto_title_terms_enabled: bool = True,
    raw_query_metadata_enabled: bool = True,
) -> tuple[list[str], list[str], list[str]]:
    if structured_query is None:
        query_tokens = tokenize(normalize_query(search_query))
        return query_tokens, query_tokens, query_tokens

    raw_query = str(structured_query.get("raw_query", ""))
    title_terms = structured_query.get("title_terms", [])
    ingredient_terms = [
        *structured_query.get("available_ingredients", []),
        *structured_query.get("must_use_ingredients", []),
    ]
    metadata_terms = structured_query.get("metadata_terms", [])
    assert isinstance(title_terms, list)
    assert isinstance(ingredient_terms, list)
    assert isinstance(metadata_terms, list)

    title_terms = filter_specific_title_terms(df, title_terms)
    structured_query["title_terms"] = title_terms
    auto_title_terms = [] if title_terms or not auto_title_terms_enabled else auto_title_terms_from_query(df, raw_query)
    if auto_title_terms:
        non_title_terms = set(metadata_terms) | set(ingredient_terms)
        auto_title_terms = [term for term in auto_title_terms if term not in non_title_terms]
    structured_query["auto_title_terms"] = auto_title_terms
    title_tokens = tokens_from_terms(title_terms or auto_title_terms)
    ingredient_tokens = tokens_from_terms(ingredient_terms)
    metadata_tokens = tokens_from_terms(metadata_terms)
    if raw_query_metadata_enabled:
        metadata_tokens = list(dict.fromkeys([*metadata_tokens, *tokenize(normalize_query(raw_query))]))
    return title_tokens, ingredient_tokens, metadata_tokens


def qdrant_vector_scores_for_query(
    query: str,
    qdrant_path: Path,
    qdrant_url: str,
    collection: str,
    top_k: int,
    model_override: str,
    local_files_only: bool,
) -> pd.DataFrame:
    """Encode the query and fetch top_k vector scores from Qdrant."""
    from rag.pipeline.qdrant_search import create_qdrant_client, load_config, search
    from rag.pipeline.embedding_encoder import encode_query

    config = load_config(qdrant_path, collection)
    model_name = model_override or config.get("model", "")
    if not model_name:
        raise ValueError("No vector model specified and Qdrant config has no model field")

    vector = encode_query(query, model_name, device=None, local_files_only=local_files_only).tolist()

    client = create_qdrant_client(qdrant_path, qdrant_url)

    points = search(client, collection, vector, top_k)

    rows: list[dict[str, Any]] = []
    for point in points:
        payload = point.payload or {}
        doc_id = payload.get("doc_id")
        if not doc_id:
            continue
        rows.append({"doc_id": doc_id, "vector_score": point.score})
    return pd.DataFrame(rows)


def build_candidates(
    df: pd.DataFrame,
    search_query: str,
    args: argparse.Namespace,
    structured_query: dict[str, list[str] | str] | None = None,
) -> pd.DataFrame:
    score_frame = df[["doc_id"]].copy()
    candidate_doc_ids: set[str] = set()
    source_by_doc_id: dict[str, set[str]] = {}
    indexed_bm25_scores: np.ndarray | None = None

    if args.mode in {"bm25", "hybrid", "weighted_rrf"}:
        title_tokens, ingredient_tokens, metadata_tokens = bm25_field_tokens(
            df,
            search_query,
            structured_query,
            auto_title_terms_enabled=not getattr(args, "no_auto_title_terms", False),
            raw_query_metadata_enabled=not getattr(args, "structured_bm25", False),
        )
        bm25_index_dir = getattr(args, "bm25_index_dir", None)
        if bm25_index_dir is not None:
            corpus_path = Path(getattr(args, "input"))
            bm25_index = load_indexed_bm25(
                Path(bm25_index_dir),
                corpus_path=corpus_path,
            )
            title_field = "title_text" if "title_text" in df.columns else "title"
            ingredient_field = (
                "ingredient_text"
                if "ingredient_text" in df.columns
                else "canonical_ingredients_text"
            )
            metadata_field = str(getattr(args, "bm25_metadata_field", "metadata_text") or "metadata_text")
            indexed_bm25_scores = bm25_index.weighted_scores(
                [
                    (title_field, args.title_weight, title_tokens),
                    (
                        ingredient_field,
                        args.ingredient_weight,
                        ingredient_tokens,
                    ),
                    (
                        metadata_field,
                        args.metadata_weight,
                        metadata_tokens,
                    ),
                ]
            )
            if len(indexed_bm25_scores) != len(df):
                raise ValueError(
                    "Indexed BM25 document count does not match loaded corpus"
                )
            bm25_top_indices = bm25_index.top_k_indices(
                indexed_bm25_scores,
                args.candidate_k,
            )
            for doc_id in df.iloc[bm25_top_indices]["doc_id"].astype(str):
                candidate_doc_ids.add(doc_id)
                source_by_doc_id.setdefault(doc_id, set()).add("bm25")
        else:
            bm25_scores = weighted_bm25_scores_by_field(
                df,
                title_tokens,
                ingredient_tokens,
                metadata_tokens,
                title_weight=args.title_weight,
                ingredient_weight=args.ingredient_weight,
                metadata_weight=args.metadata_weight,
            )
            score_frame["bm25_score"] = bm25_scores
            bm25_top = score_frame.nlargest(args.candidate_k, "bm25_score")
            for doc_id in bm25_top["doc_id"].astype(str):
                candidate_doc_ids.add(doc_id)
                source_by_doc_id.setdefault(doc_id, set()).add("bm25")
    else:
        score_frame["bm25_score"] = 0.0

    vector_score_by_doc_id: dict[str, float] = {}
    if args.mode in {"vector", "hybrid", "weighted_rrf"}:
        vector_scores = qdrant_vector_scores_for_query(
            search_query,
            args.qdrant_path.resolve(),
            getattr(args, "qdrant_url", ""),
            args.collection,
            args.candidate_k,
            args.vector_model,
            args.local_files_only,
        )
        if indexed_bm25_scores is not None:
            vector_top = vector_scores.nlargest(args.candidate_k, "vector_score")
            vector_score_by_doc_id = {
                str(row.doc_id): float(row.vector_score)
                for row in vector_scores.itertuples(index=False)
            }
        else:
            score_frame = score_frame.merge(vector_scores, on="doc_id", how="left")
            score_frame["vector_score"] = score_frame["vector_score"].fillna(0.0)
            vector_top = score_frame.nlargest(args.candidate_k, "vector_score")
        for doc_id in vector_top["doc_id"].astype(str):
            candidate_doc_ids.add(doc_id)
            source_by_doc_id.setdefault(doc_id, set()).add("vector")
    else:
        score_frame["vector_score"] = 0.0

    if not candidate_doc_ids:
        raise ValueError(f"No candidate retriever enabled for mode: {args.mode}")

    if indexed_bm25_scores is not None:
        doc_ids = df["doc_id"].astype(str)
        selected_mask = doc_ids.isin(candidate_doc_ids).to_numpy()
        selected_positions = np.flatnonzero(selected_mask)
        candidates = df.iloc[selected_positions].copy()
        candidates["bm25_score"] = indexed_bm25_scores[selected_positions]
        candidates["vector_score"] = [
            vector_score_by_doc_id.get(doc_id, 0.0)
            for doc_id in doc_ids.iloc[selected_positions]
        ]
    else:
        candidate_scores = score_frame[
            score_frame["doc_id"].astype(str).isin(candidate_doc_ids)
        ].copy()
        candidates = candidate_scores.merge(df, on="doc_id", how="inner")
    candidates["candidate_source"] = candidates["doc_id"].astype(str).map(
        lambda doc_id: "+".join(sorted(source_by_doc_id.get(doc_id, {"unknown"})))
    )
    return candidates


def contains_term(text: str, term: str) -> bool:
    if not term:
        return False
    if " " in term:
        return term in text
    return bool(re.search(rf"\b{re.escape(term)}\b", text))


def row_text(row: pd.Series, columns: list[str]) -> str:
    values = [str(row.get(column, "") or "").lower() for column in columns]
    return " ".join(values)


def recipe_ingredient_terms(row: pd.Series) -> list[str]:
    terms = parse_ingredient_terms(row.get("normalized_ingredient_terms", ""))
    if terms:
        return terms
    return normalize_recipe_terms(
        " ".join(
            str(row.get(column, "") or "")
            for column in ["canonical_ingredients_text", "ingredient_text", "ingredients_text"]
        )
    )


def matched_available_terms(recipe_terms: list[str], available_terms: list[str]) -> list[str]:
    return [
        term
        for term in available_terms
        if any(availability_covers(term, recipe_term) for recipe_term in recipe_terms)
    ]


def missing_recipe_terms(recipe_terms: list[str], available_terms: list[str], pantry_terms: list[str]) -> list[str]:
    available_for_comparison = [*available_terms, *pantry_terms]
    return [
        term
        for term in recipe_terms
        if not any(availability_covers(available, term) for available in available_for_comparison)
    ]


def min_max_normalize(values: pd.Series) -> pd.Series:
    min_value = values.min()
    max_value = values.max()
    if math.isclose(float(min_value), float(max_value)):
        return pd.Series([0.0] * len(values), index=values.index)
    return (values - min_value) / (max_value - min_value)


def rerank_candidates(
    candidates: pd.DataFrame,
    structured_query: dict[str, list[str] | str],
    ingredient_coverage_weight: float,
    metadata_match_weight: float,
    title_match_weight: float,
    missing_ingredient_penalty: float,
    missing_metadata_penalty: float,
    vector_weight: float,
) -> pd.DataFrame:
    ingredient_terms = structured_query["available_ingredients"]
    must_use_terms = structured_query.get("must_use_ingredients", [])
    pantry_terms = structured_query.get("pantry_ingredients", [])
    metadata_terms = structured_query["metadata_terms"]
    metadata_term_groups = structured_query.get("metadata_term_groups", {})
    excluded_metadata_terms = structured_query.get("excluded_metadata_terms", [])
    excluded_metadata_term_groups = structured_query.get("excluded_metadata_term_groups", {})
    title_terms = structured_query["title_terms"]
    auto_title_terms = structured_query.get("auto_title_terms", [])
    assert isinstance(ingredient_terms, list)
    assert isinstance(must_use_terms, list)
    assert isinstance(pantry_terms, list)
    assert isinstance(metadata_terms, list)
    assert isinstance(metadata_term_groups, dict)
    assert isinstance(excluded_metadata_terms, list)
    assert isinstance(excluded_metadata_term_groups, dict)
    assert isinstance(title_terms, list)
    assert isinstance(auto_title_terms, list)
    title_terms = title_terms or auto_title_terms

    rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        recipe_terms = recipe_ingredient_terms(row)
        metadata_text = row_text(row, ["metadata_text"])
        title_text = row_text(row, ["title", "title_text"])

        matched_ingredients = matched_available_terms(recipe_terms, ingredient_terms)
        matched_must_use = matched_available_terms(recipe_terms, must_use_terms)
        missing_ingredients = [term for term in must_use_terms if term not in matched_must_use]
        missing_recipe_ingredients = missing_recipe_terms(recipe_terms, ingredient_terms, pantry_terms)
        matched_metadata: list[str] = []
        missing_metadata: list[str] = []
        for term in metadata_terms:
            variants = metadata_term_groups.get(term, [term])
            matched_variant = next((variant for variant in variants if contains_term(metadata_text, variant)), "")
            if matched_variant:
                matched_metadata.append(term if matched_variant == term else f"{term}~{matched_variant}")
            else:
                missing_metadata.append(term)
        excluded_metadata_matches: list[str] = []
        for term in excluded_metadata_terms:
            variants = excluded_metadata_term_groups.get(term, [term])
            matched_variant = next((variant for variant in variants if contains_term(metadata_text, variant)), "")
            if matched_variant:
                excluded_metadata_matches.append(term if matched_variant == term else f"{term}~{matched_variant}")
        matched_title = [term for term in title_terms if contains_term(title_text, term)]

        ingredient_coverage = len(matched_ingredients) / len(ingredient_terms) if ingredient_terms else 0.0
        recipe_missing_ratio = (
            len(missing_recipe_ingredients) / len(recipe_terms)
            if recipe_terms and ingredient_terms
            else 0.0
        )
        no_available_ingredient_penalty = (
            NO_AVAILABLE_INGREDIENT_PENALTY if ingredient_terms and not matched_ingredients else 0.0
        )
        recipe_missing_penalty = recipe_missing_ratio * RECIPE_MISSING_RATIO_PENALTY if ingredient_terms else 0.0
        minor_adjustment_bonus = (
            MINOR_ADJUSTMENT_BONUS
            if ingredient_terms and matched_ingredients and 0 < len(missing_recipe_ingredients) <= 2
            else 0.0
        )
        as_is_bonus = COOKABLE_AS_IS_BONUS if ingredient_terms and recipe_terms and not missing_recipe_ingredients else 0.0
        metadata_match = len(matched_metadata) / len(metadata_terms) if metadata_terms else 0.0
        title_match = len(matched_title) / len(title_terms) if title_terms else 0.0
        specific_title_penalty = title_match_weight if len(title_terms) == 1 and not matched_title else 0.0
        missing_penalty = len(missing_ingredients) * missing_ingredient_penalty
        metadata_penalty = len(missing_metadata) * missing_metadata_penalty
        excluded_metadata_penalty = len(excluded_metadata_matches) * metadata_match_weight

        rows.append(
            {
                "ingredient_coverage": ingredient_coverage,
                "metadata_match": metadata_match,
                "title_match": title_match,
                "matched_ingredients": ", ".join(matched_ingredients),
                "matched_must_use_ingredients": ", ".join(matched_must_use),
                "missing_ingredients": ", ".join(missing_ingredients),
                "recipe_missing_ingredients": ", ".join(missing_recipe_ingredients),
                "recipe_missing_ratio": recipe_missing_ratio,
                "matched_metadata": ", ".join(matched_metadata),
                "missing_metadata": ", ".join(missing_metadata),
                "matched_title": ", ".join(matched_title),
                "excluded_metadata_matches": ", ".join(excluded_metadata_matches),
                "rerank_adjustment": (
                    ingredient_coverage_weight * ingredient_coverage
                    + metadata_match_weight * metadata_match
                    + title_match_weight * title_match
                    + minor_adjustment_bonus
                    + as_is_bonus
                    - missing_penalty
                    - recipe_missing_penalty
                    - no_available_ingredient_penalty
                    - metadata_penalty
                    - specific_title_penalty
                    - excluded_metadata_penalty
                ),
            }
        )

    features = pd.DataFrame(rows, index=candidates.index)
    reranked = pd.concat([candidates, features], axis=1)
    if "bm25_score" not in reranked.columns:
        reranked["bm25_score"] = 0.0
    if "vector_score" not in reranked.columns:
        reranked["vector_score"] = 0.0
    reranked["bm25_score_norm"] = min_max_normalize(reranked["bm25_score"])
    reranked["vector_score_norm"] = min_max_normalize(reranked["vector_score"])
    reranked["final_score"] = (
        reranked["bm25_score_norm"]
        + vector_weight * reranked["vector_score_norm"]
        + reranked["rerank_adjustment"]
    )
    return reranked.sort_values("final_score", ascending=False)


