#!/usr/bin/env python3
"""Build the final retrieval corpus for BM25-first recipe retrieval."""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from rag.paths import RULES_DIR
from rag.build.ingredient_line_normalization import canonical_phrase_confirmed_terms, use_corpus_build_ingredient_rules
from rag.pipeline.ingredient_normalization import serialize_ingredient_terms

LOGGER = logging.getLogger("build_retrieval_corpus")

WHITESPACE_RE = re.compile(r"\s+")
NON_FOOD_CATEGORIES = {"Bath/Beauty", "Household Cleaner", "Homeopathy/Remedies"}
DEFAULT_TAXONOMY_RULES_PATH = RULES_DIR / "buildtime_retrieval_taxonomy_rules.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final retrieval corpus from cleaned recipes and aggregated reviews")
    parser.add_argument(
        "--recipes",
        type=Path,
        default=Path("outputs/recipes_cleaned.csv"),
        help="Path to cleaned recipe dataset",
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("outputs/reviews_aggregated.csv"),
        help="Path to aggregated review dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for retrieval corpus outputs",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="retrieval_corpus",
        help="Output filename stem without extension",
    )
    parser.add_argument(
        "--compact-output-stem",
        type=str,
        default="retrieval_corpus_compact",
        help="Output filename stem for compact runtime corpus",
    )
    parser.add_argument(
        "--runtime-output-stem",
        type=str,
        default="retrieval_corpus_runtime",
        help="Output filename stem for final demo/runtime corpus",
    )
    parser.add_argument(
        "--no-compact",
        action="store_true",
        help="Skip writing the compact runtime corpus",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "parquet", "both"),
        default="both",
        help="Output format",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Optional row limit for debugging",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--taxonomy-rules",
        type=Path,
        default=DEFAULT_TAXONOMY_RULES_PATH,
        help="JSON rules for normalized retrieval taxonomy columns",
    )
    parser.add_argument(
        "--taxonomy-profile",
        type=Path,
        default=None,
        help="Optional path for distinct counts of normalized taxonomy columns",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = html.unescape(str(value))
    text = (
        text.replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2019", "'")
        .replace("â€“", "-")
        .replace("â€œ", '"')
        .replace("â€", '"')
        .replace("â€™", "'")
    )
    return WHITESPACE_RE.sub(" ", text).strip()


def join_text_parts(*parts: Any) -> str:
    cleaned = [clean_text(part) for part in parts]
    cleaned = [part for part in cleaned if part]
    return " ".join(cleaned)


def join_unique_text_parts(*parts: Any) -> str:
    seen: set[str] = set()
    unique_parts: list[str] = []
    for part in parts:
        cleaned = clean_text(part)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_parts.append(cleaned)
    return " ".join(unique_parts)


def split_taxonomy_values(value: Any) -> list[str]:
    text = clean_text(value).lower()
    if not text:
        return []
    parts = re.split(r"\s*\|\s*|\s*;\s*|\s*,\s*", text)
    return [part.strip() for part in parts if part.strip()]


def normalize_for_match(value: Any) -> str:
    return clean_text(value).lower().replace("_", " ")


def text_contains_phrase(text: str, phrase: str) -> bool:
    phrase = normalize_for_match(phrase)
    if not phrase:
        return False
    if "/" in phrase or " " in phrase or "." in phrase:
        return phrase in text
    return bool(re.search(rf"\b{re.escape(phrase)}\b", text))


def load_taxonomy_rules(path: Path) -> dict[str, dict[str, list[str]]]:
    if not path.exists():
        LOGGER.warning("Taxonomy rules file not found: %s", path)
        return {}
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    rules: dict[str, dict[str, list[str]]] = {}
    for column, mapping in raw.items():
        if not isinstance(mapping, dict):
            continue
        normalized_mapping: dict[str, list[str]] = {}
        for target, aliases in mapping.items():
            if isinstance(aliases, list):
                normalized_mapping[str(target)] = [normalize_for_match(alias) for alias in aliases]
        rules[column] = normalized_mapping
    return rules


def map_taxonomy_values(text: str, mapping: dict[str, list[str]]) -> str:
    normalized_text = normalize_for_match(text)
    matches: list[str] = []
    for target, aliases in mapping.items():
        if any(text_contains_phrase(normalized_text, alias) for alias in aliases):
            matches.append(target)
    return " | ".join(dict.fromkeys(matches))


def add_normalized_taxonomy_columns(
    recipes: pd.DataFrame,
    rules: dict[str, dict[str, list[str]]],
) -> pd.DataFrame:
    if not rules:
        return recipes

    source_text = recipes.apply(
        lambda row: join_unique_text_parts(
            row.get("source_primary_category", ""),
            row.get("broad_meal_category", ""),
            row.get("specific_dish_type", ""),
            row.get("main_ingredient_category", ""),
            row.get("difficulty_tags_text", ""),
            row.get("diet_tags_text", ""),
            row.get("method_tags_text", ""),
            row.get("cost_tags_text", ""),
            row.get("meal_context_tags_text", ""),
            row.get("occasion_tags_text", ""),
            row.get("flavor_tags_text", ""),
            row.get("cuisine_tags_text", ""),
        ),
        axis=1,
    )
    for column, mapping in rules.items():
        recipes[column] = source_text.map(lambda text: map_taxonomy_values(text, mapping))
    return recipes


def build_taxonomy_profile(df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    profile: dict[str, Any] = {"row_count": int(len(df)), "columns": {}}
    for column in columns:
        if column not in df.columns:
            continue
        counter: Counter[str] = Counter()
        non_empty = 0
        for value in df[column].fillna(""):
            values = split_taxonomy_values(value)
            if values:
                non_empty += 1
            counter.update(values)
        profile["columns"][column] = {
            "non_empty_rows": non_empty,
            "distinct_count": len(counter),
            "distinct_values": [
                {"value": value, "count": int(count)}
                for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
            ],
        }
    return profile


def write_taxonomy_profile(df: pd.DataFrame, output_path: Path) -> None:
    normalized_columns = [column for column in df.columns if column.startswith("normalized_")]
    profile = build_taxonomy_profile(df, normalized_columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Writing taxonomy profile: %s", output_path)


def normalize_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def repair_time_columns(df: pd.DataFrame) -> None:
    required = {"PrepTime_minutes", "CookTime_minutes", "TotalTime_minutes"}
    if not required.issubset(df.columns):
        return

    prep = df["PrepTime_minutes"]
    cook = df["CookTime_minutes"]
    total = df["TotalTime_minutes"]
    component_sum = prep.fillna(0) + cook.fillna(0)
    has_component_time = prep.notna() | cook.notna()
    repair_mask = has_component_time & (component_sum > 0) & (
        total.isna() | (total <= 0) | (total < component_sum)
    )
    repaired_count = int(repair_mask.sum())
    if repaired_count:
        df.loc[repair_mask, "TotalTime_minutes"] = component_sum[repair_mask]
        LOGGER.info("Repaired TotalTime_minutes from PrepTime+CookTime for %d rows", repaired_count)


def build_bm25_text(row: pd.Series) -> str:
    return join_unique_text_parts(
        row.get("title_text", ""),
        row.get("ingredient_text", ""),
        row.get("metadata_text", ""),
    )


def build_title_text(row: pd.Series) -> str:
    return join_unique_text_parts(row.get("title_clean", ""))


def build_ingredient_text(row: pd.Series) -> str:
    return join_unique_text_parts(row.get("canonical_ingredients_text", ""))


def build_normalized_ingredient_terms(row: pd.Series) -> str:
    return serialize_ingredient_terms(
        canonical_phrase_confirmed_terms(
            row.get("canonical_ingredients_text", ""),
            row.get("ingredients_list_json", ""),
        )
    )


def build_metadata_text(row: pd.Series) -> str:
    return join_unique_text_parts(
        row.get("normalized_meal_type", ""),
        row.get("normalized_dish_type", ""),
        row.get("normalized_method_tags", ""),
        row.get("normalized_diet_tags", ""),
        row.get("normalized_cost_tags", ""),
        row.get("normalized_context_tags", ""),
        row.get("source_primary_category", ""),
        row.get("broad_meal_category", ""),
        row.get("specific_dish_type", ""),
        row.get("main_ingredient_category", ""),
        row.get("time_tags_text", ""),
        row.get("difficulty_tags_text", ""),
        row.get("diet_tags_text", ""),
        row.get("audience_tags_text", ""),
        row.get("method_tags_text", ""),
        row.get("cost_tags_text", ""),
        row.get("meal_context_tags_text", ""),
        row.get("occasion_tags_text", ""),
        row.get("flavor_tags_text", ""),
        row.get("cuisine_tags_text", ""),
    )


def build_instruction_text(row: pd.Series) -> str:
    return join_text_parts(row.get("directions_text", ""))


def build_vector_text(row: pd.Series) -> str:
    return join_text_parts(
        row.get("title_text", ""),
        row.get("ingredient_text", ""),
        row.get("metadata_text", ""),
    )


def build_debug_full_text(row: pd.Series) -> str:
    # Debug-only field for inspecting what information a recipe contributes to search.
    return join_text_parts(
        row.get("title_text", ""),
        row.get("ingredient_text", ""),
        row.get("metadata_text", ""),
        row.get("instruction_text", ""),
    )


def build_answer_context_text(row: pd.Series) -> str:
    return join_text_parts(
        row.get("title_text", ""),
        row.get("canonical_ingredients_text", ""),
        row.get("ingredients_text", ""),
        row.get("instruction_text", ""),
    )


def filter_retrieval_rows(recipes: pd.DataFrame) -> pd.DataFrame:
    before = len(recipes)
    source_category = recipes.get(
        "source_primary_category",
        pd.Series([""] * len(recipes), index=recipes.index),
    ).fillna("").astype(str).str.strip()
    directions = recipes.get(
        "directions_text",
        pd.Series([""] * len(recipes), index=recipes.index),
    ).fillna("").astype(str).str.strip()

    keep_mask = ~source_category.isin(NON_FOOD_CATEGORIES) & (directions != "")
    filtered = recipes.loc[keep_mask].copy()
    removed = before - len(filtered)
    if removed:
        LOGGER.info("Filtered %d rows from retrieval corpus (non-food or missing directions)", removed)
    return filtered


def load_recipes(path: Path, sample: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Recipes file not found: {path}")

    LOGGER.info("Loading cleaned recipes: %s", path)
    recipes = pd.read_csv(path)
    if sample > 0:
        recipes = recipes.head(sample).copy()
        LOGGER.info("Using recipe sample: first %d rows", sample)
    return recipes


def load_reviews(path: Path) -> pd.DataFrame:
    if not path.exists():
        LOGGER.warning("Aggregated reviews file not found, continuing without review features: %s", path)
        return pd.DataFrame()

    LOGGER.info("Loading aggregated reviews: %s", path)
    return pd.read_csv(path)


def prepare_recipes(
    recipes: pd.DataFrame,
    taxonomy_rules: dict[str, dict[str, list[str]]] | None = None,
) -> pd.DataFrame:
    normalize_numeric(
        recipes,
        [
            "CookTime_minutes",
            "PrepTime_minutes",
            "TotalTime_minutes",
            "Calories",
            "ProteinContent",
            "AggregatedRating",
            "ReviewCount",
            "RecipeServings",
        ],
    )
    repair_time_columns(recipes)

    for col in [
        "doc_id",
        "title",
        "title_clean",
        "canonical_ingredients_text",
        "ingredients_text",
        "directions_text",
        "ner_text",
        "description_clean",
        "source_primary_category",
        "broad_meal_category",
        "specific_dish_type",
        "main_ingredient_category",
        "time_tags_text",
        "difficulty_tags_text",
        "diet_tags_text",
        "audience_tags_text",
        "method_tags_text",
        "cost_tags_text",
        "meal_context_tags_text",
        "occasion_tags_text",
        "flavor_tags_text",
        "cuisine_tags_text",
        "other_tags_text",
    ]:
        if col in recipes.columns:
            recipes[col] = recipes[col].map(clean_text)

    recipes = add_normalized_taxonomy_columns(recipes, taxonomy_rules or {})
    recipes["title_text"] = recipes.apply(build_title_text, axis=1)
    recipes["ingredient_text"] = recipes.apply(build_ingredient_text, axis=1)
    recipes["normalized_ingredient_terms"] = recipes.apply(build_normalized_ingredient_terms, axis=1)
    recipes["metadata_text"] = recipes.apply(build_metadata_text, axis=1)
    recipes["instruction_text"] = recipes.apply(build_instruction_text, axis=1)
    recipes["bm25_text"] = recipes.apply(build_bm25_text, axis=1)
    recipes["vector_text"] = recipes.apply(build_vector_text, axis=1)
    recipes["debug_full_text"] = recipes.apply(build_debug_full_text, axis=1)
    recipes["answer_context_text"] = recipes.apply(build_answer_context_text, axis=1)
    return recipes


def merge_reviews(recipes: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    if reviews.empty:
        return recipes

    review_cols = [
        "doc_id",
        "review_count_actual",
        "mean_rating_actual",
        "median_rating_actual",
        "positive_review_count",
        "negative_review_count",
        "modification_review_count",
        "review_support_score",
        "positive_review_snippets_json",
        "negative_review_snippets_json",
        "modification_review_snippets_json",
        "substitution_examples_json",
    ]
    available = [col for col in review_cols if col in reviews.columns]
    merged = recipes.merge(reviews[available], on="doc_id", how="left")
    return merged


def finalize_corpus(df: pd.DataFrame) -> pd.DataFrame:
    # Keep broad trace/debug columns in the full corpus.
    desired_columns = [
        "doc_id",
        "source",
        "link",
        "RecipeId",
        "title",
        "title_clean",
        "Images_list_json",
        "canonical_ingredients_json",
        "canonical_ingredients_text",
        "normalized_ingredient_terms",
        "title_text",
        "ingredient_text",
        "metadata_text",
        "instruction_text",
        "ingredients_text",
        "ner_text",
        "directions_text",
        "description_clean",
        "source_primary_category",
        "broad_meal_category",
        "specific_dish_type",
        "main_ingredient_category",
        "normalized_meal_type",
        "normalized_dish_type",
        "normalized_method_tags",
        "normalized_diet_tags",
        "normalized_cost_tags",
        "normalized_cuisine_tags",
        "normalized_context_tags",
        "time_tags_json",
        "difficulty_tags_json",
        "diet_tags_json",
        "audience_tags_json",
        "method_tags_json",
        "cost_tags_json",
        "meal_context_tags_json",
        "occasion_tags_json",
        "flavor_tags_json",
        "cuisine_tags_json",
        "other_tags_json",
        "time_tags_text",
        "difficulty_tags_text",
        "diet_tags_text",
        "audience_tags_text",
        "method_tags_text",
        "cost_tags_text",
        "meal_context_tags_text",
        "occasion_tags_text",
        "flavor_tags_text",
        "cuisine_tags_text",
        "other_tags_text",
        "CookTime_minutes",
        "PrepTime_minutes",
        "TotalTime_minutes",
        "Calories",
        "ProteinContent",
        "FatContent",
        "SaturatedFatContent",
        "CholesterolContent",
        "SodiumContent",
        "CarbohydrateContent",
        "FiberContent",
        "SugarContent",
        "AggregatedRating",
        "ReviewCount",
        "RecipeServings",
        "review_count_actual",
        "mean_rating_actual",
        "median_rating_actual",
        "positive_review_count",
        "negative_review_count",
        "modification_review_count",
        "review_support_score",
        "positive_review_snippets_json",
        "negative_review_snippets_json",
        "modification_review_snippets_json",
        "substitution_examples_json",
        "bm25_text",
        "vector_text",
        "debug_full_text",
        "answer_context_text",
    ]
    ordered = [col for col in desired_columns if col in df.columns]
    final = df[ordered].copy()
    return final


def finalize_compact_corpus(df: pd.DataFrame) -> pd.DataFrame:
    # Runtime-oriented schema: keep only fields used by retrieval, reranking, and answer generation.
    desired_columns = [
        "doc_id",
        "source",
        "link",
        "RecipeId",
        "title",
        "title_clean",
        "canonical_ingredients_text",
        "normalized_ingredient_terms",
        "ingredients_text",
        "directions_text",
        "title_text",
        "ingredient_text",
        "metadata_text",
        "instruction_text",
        "bm25_text",
        "vector_text",
        "answer_context_text",
        "normalized_meal_type",
        "normalized_dish_type",
        "normalized_method_tags",
        "normalized_diet_tags",
        "normalized_cost_tags",
        "normalized_cuisine_tags",
        "normalized_context_tags",
        "CookTime_minutes",
        "PrepTime_minutes",
        "TotalTime_minutes",
        "Calories",
        "ProteinContent",
        "AggregatedRating",
        "ReviewCount",
        "RecipeServings",
        "review_count_actual",
        "mean_rating_actual",
        "median_rating_actual",
        "positive_review_count",
        "negative_review_count",
        "modification_review_count",
        "review_support_score",
        "substitution_examples_json",
    ]
    ordered = [col for col in desired_columns if col in df.columns]
    return df[ordered].copy()


def first_list_item(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, list):
        return str(value[0]).strip() if value else ""
    text = str(value).strip()
    if not text or text.lower() == "nan" or text == "[]":
        return ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return ""
    if isinstance(parsed, list) and parsed:
        return str(parsed[0]).strip()
    return ""


def finalize_runtime_corpus(df: pd.DataFrame) -> pd.DataFrame:
    # Final runtime/demo schema: remove exact duplicate technical fields but keep retrieval,
    # answer, normalized matching, image, nutrition, and basic quality signals.
    runtime = pd.DataFrame(index=df.index)
    runtime["doc_id"] = df.get("doc_id", "")
    runtime["source"] = df.get("source", "")
    runtime["link"] = df.get("link", "")
    runtime["RecipeId"] = df.get("RecipeId", "")
    runtime["title"] = df.get("title_clean", df.get("title", ""))
    if "Images_list_json" in df.columns:
        runtime["primary_image_url"] = df["Images_list_json"].map(first_list_item)
    else:
        runtime["primary_image_url"] = ""

    direct_columns = [
        "canonical_ingredients_text",
        "normalized_ingredient_terms",
        "ingredients_text",
        "instruction_text",
        "metadata_text",
        "vector_text",
        "normalized_meal_type",
        "normalized_dish_type",
        "normalized_method_tags",
        "normalized_diet_tags",
        "normalized_cost_tags",
        "normalized_cuisine_tags",
        "normalized_context_tags",
        "CookTime_minutes",
        "PrepTime_minutes",
        "TotalTime_minutes",
        "Calories",
        "ProteinContent",
        "FatContent",
        "SaturatedFatContent",
        "CholesterolContent",
        "SodiumContent",
        "CarbohydrateContent",
        "FiberContent",
        "SugarContent",
        "AggregatedRating",
        "ReviewCount",
        "RecipeServings",
        "review_support_score",
    ]
    for column in direct_columns:
        if column in df.columns:
            runtime[column] = df[column]
    return runtime


def save_outputs(df: pd.DataFrame, output_dir: Path, output_stem: str, output_format: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{output_stem}.csv"
    parquet_path = output_dir / f"{output_stem}.parquet"

    if output_format in {"csv", "both"}:
        LOGGER.info("Writing CSV: %s", csv_path)
        df.to_csv(csv_path, index=False, encoding="utf-8")

    if output_format in {"parquet", "both"}:
        LOGGER.info("Writing Parquet: %s", parquet_path)
        try:
            df.to_parquet(parquet_path, index=False)
        except ImportError as exc:
            LOGGER.warning("Parquet write skipped because no engine is installed: %s", exc)
            if output_format == "parquet":
                raise


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    use_corpus_build_ingredient_rules()

    try:
        taxonomy_rules = load_taxonomy_rules(args.taxonomy_rules.resolve())
        recipes = load_recipes(args.recipes.resolve(), args.sample)
        reviews = load_reviews(args.reviews.resolve())
        recipes = prepare_recipes(recipes, taxonomy_rules)
        recipes = filter_retrieval_rows(recipes)
        corpus = merge_reviews(recipes, reviews)
        corpus = finalize_corpus(corpus)
        LOGGER.info("Final retrieval corpus shape: %s", corpus.shape)
        save_outputs(corpus, args.output_dir.resolve(), args.output_stem, args.format)
        if not args.no_compact:
            compact = finalize_compact_corpus(corpus)
            LOGGER.info("Compact retrieval corpus shape: %s", compact.shape)
            save_outputs(compact, args.output_dir.resolve(), args.compact_output_stem, args.format)
            runtime = finalize_runtime_corpus(corpus)
            LOGGER.info("Runtime retrieval corpus shape: %s", runtime.shape)
            save_outputs(runtime, args.output_dir.resolve(), args.runtime_output_stem, args.format)
        profile_path = args.taxonomy_profile
        if profile_path is None:
            profile_path = args.output_dir.resolve() / f"{args.output_stem}_taxonomy_profile.json"
        write_taxonomy_profile(corpus, profile_path.resolve())
    except Exception as exc:
        LOGGER.error("Retrieval corpus build failed: %s", exc, exc_info=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
