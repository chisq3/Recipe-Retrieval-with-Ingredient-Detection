#!/usr/bin/env python3
"""Aggregate raw recipe reviews into recipe-level features for RAG reranking."""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger("aggregate_reviews")

WHITESPACE_RE = re.compile(r"\s+")
HTML_BREAK_RE = re.compile(r"<br\s*/?>", re.I)
POSITIVE_PATTERNS = {
    "success": re.compile(r"\b(?:will make again|keeper|loved it|excellent|perfect|great|delicious|very good|outstanding)\b", re.I),
    "easy": re.compile(r"\b(?:easy|simple|quick)\b", re.I),
}
NEGATIVE_PATTERNS = {
    "failure": re.compile(r"\b(?:dry|bland|rubber|too salty|too sweet|burnt|burned|didn't work|did not work|failed)\b", re.I),
}
MODIFICATION_PATTERNS = {
    "substitution": re.compile(r"\b(?:substitute|substituted|instead of|replaced)\b", re.I),
    "addition": re.compile(r"\b(?:i added|added some|added extra|threw in)\b", re.I),
    "omission": re.compile(r"\b(?:without|omitted|left out|skipped|deleted)\b", re.I),
}
MIN_SNIPPET_CHARS = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate reviews into recipe-level features")
    parser.add_argument(
        "--recipes",
        type=Path,
        default=Path("outputs/recipes_cleaned.csv"),
        help="Path to cleaned recipes CSV",
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("rag_dataset/reviews.csv"),
        help="Path to raw reviews CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for aggregated outputs",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="reviews_aggregated",
        help="Output filename stem without extension",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "parquet", "both"),
        default="both",
        help="Output format",
    )
    parser.add_argument(
        "--sample-recipe-limit",
        type=int,
        default=0,
        help="Optional limit on number of cleaned recipes used for join debugging",
    )
    parser.add_argument(
        "--max-review-snippets",
        type=int,
        default=3,
        help="Maximum positive/negative example snippets stored per recipe",
    )
    parser.add_argument(
        "--max-review-summary-chars",
        type=int,
        default=1200,
        help="Maximum characters in concatenated review summary",
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


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = html.unescape(str(value))
    text = HTML_BREAK_RE.sub(" ", text)
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    text = text.replace("\u00a0", " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def load_recipes(path: Path, sample_recipe_limit: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cleaned recipes file not found: {path}")

    LOGGER.info("Loading cleaned recipes: %s", path)
    recipes = pd.read_csv(path, usecols=lambda c: c in {"doc_id", "RecipeId", "title", "source", "link"})
    recipes["RecipeId"] = pd.to_numeric(recipes["RecipeId"], errors="coerce")
    recipes = recipes.dropna(subset=["RecipeId"]).copy()
    recipes["RecipeId"] = recipes["RecipeId"].astype("int64")

    if sample_recipe_limit > 0:
        recipes = recipes.head(sample_recipe_limit).copy()
        LOGGER.info("Using recipe sample limit: %d", sample_recipe_limit)

    return recipes


def load_reviews(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Reviews file not found: {path}")

    LOGGER.info("Loading reviews: %s", path)
    reviews = pd.read_csv(path)
    reviews["RecipeId"] = pd.to_numeric(reviews["RecipeId"], errors="coerce")
    reviews["Rating"] = pd.to_numeric(reviews["Rating"], errors="coerce")
    reviews["Review_clean"] = reviews["Review"].map(clean_text)
    reviews["DateSubmitted_ts"] = pd.to_datetime(reviews["DateSubmitted"], errors="coerce", utc=True)
    reviews["DateModified_ts"] = pd.to_datetime(reviews["DateModified"], errors="coerce", utc=True)
    reviews = reviews.dropna(subset=["RecipeId"]).copy()
    reviews["RecipeId"] = reviews["RecipeId"].astype("int64")
    return reviews


def extract_pattern_flags(series: pd.Series, patterns: dict[str, re.Pattern[str]]) -> dict[str, int]:
    text = series.fillna("").astype(str)
    return {
        name: int(text.str.contains(pattern, regex=True).sum())
        for name, pattern in patterns.items()
    }


def collect_snippets(group: pd.DataFrame, max_snippets: int, positive: bool) -> list[str]:
    snippets: list[str] = []
    for _, row in group.iterrows():
        review = clean_text(row.get("Review_clean", ""))
        if not review or len(review) < MIN_SNIPPET_CHARS:
            continue
        rating = row.get("Rating")
        positive_match = bool(POSITIVE_PATTERNS["success"].search(review) or POSITIVE_PATTERNS["easy"].search(review))
        negative_match = bool(NEGATIVE_PATTERNS["failure"].search(review))
        if positive and pd.notna(rating) and float(rating) >= 4 and positive_match:
            snippets.append(review)
        elif not positive and (
            (pd.notna(rating) and float(rating) <= 2 and (negative_match or len(review) > 20))
            or negative_match
        ):
            snippets.append(review)
        if len(snippets) >= max_snippets:
            break
    return snippets


def collect_modification_snippets(group: pd.DataFrame, max_snippets: int) -> list[str]:
    snippets: list[str] = []
    for _, row in group.iterrows():
        review = clean_text(row.get("Review_clean", ""))
        if not review or len(review) < MIN_SNIPPET_CHARS:
            continue
        if any(pattern.search(review) for pattern in MODIFICATION_PATTERNS.values()):
            snippets.append(review)
        if len(snippets) >= max_snippets:
            break
    return snippets


def collect_substitution_examples(group: pd.DataFrame, max_snippets: int) -> list[str]:
    snippets: list[str] = []
    pattern = MODIFICATION_PATTERNS["substitution"]
    for _, row in group.iterrows():
        review = clean_text(row.get("Review_clean", ""))
        if not review or len(review) < MIN_SNIPPET_CHARS:
            continue
        if pattern.search(review):
            snippets.append(review)
        if len(snippets) >= max_snippets:
            break
    return snippets


def build_review_summary(group: pd.DataFrame, max_chars: int, max_snippets: int) -> str:
    snippets = []
    snippets.extend(collect_snippets(group, max_snippets=max_snippets, positive=True))
    if len(snippets) < max_snippets:
        for text in collect_modification_snippets(group, max_snippets=max_snippets):
            if text not in snippets:
                snippets.append(text)
            if len(snippets) >= max_snippets:
                break
    summary = " ".join(snippets)
    return summary[:max_chars].strip()


def compute_review_support_score(
    mean_rating: float | None,
    review_count: int,
    positive_success_count: int,
    negative_failure_count: int,
) -> float:
    if mean_rating is None:
        return 0.0

    rating_component = max(0.0, min(1.0, float(mean_rating) / 5.0))
    volume_component = min(1.0, math.log1p(max(0, review_count)) / math.log1p(50))
    sentiment_component = max(
        0.0,
        min(1.0, (positive_success_count - negative_failure_count + max(1, review_count)) / (2 * max(1, review_count))),
    )
    score = 0.5 * rating_component + 0.3 * volume_component + 0.2 * sentiment_component
    return round(score, 4)


def aggregate_reviews(
    recipes: pd.DataFrame,
    reviews: pd.DataFrame,
    max_review_snippets: int,
    max_review_summary_chars: int,
) -> pd.DataFrame:
    valid_recipe_ids = set(recipes["RecipeId"].tolist())
    reviews = reviews[reviews["RecipeId"].isin(valid_recipe_ids)].copy()
    LOGGER.info("Joinable review rows: %d", len(reviews))

    rows: list[dict[str, Any]] = []
    for recipe_id, group in reviews.groupby("RecipeId", sort=False):
        recipe_meta = recipes.loc[recipes["RecipeId"] == recipe_id].iloc[0]
        group = group.sort_values(["DateSubmitted_ts", "ReviewId"], ascending=[False, False], na_position="last")

        review_text = group["Review_clean"].fillna("").astype(str)
        positive_flags = extract_pattern_flags(review_text, POSITIVE_PATTERNS)
        negative_flags = extract_pattern_flags(review_text, NEGATIVE_PATTERNS)
        modification_flags = extract_pattern_flags(review_text, MODIFICATION_PATTERNS)
        positive_snippets = collect_snippets(group, max_review_snippets, positive=True)
        negative_snippets = collect_snippets(group, max_review_snippets, positive=False)
        modification_snippets = collect_modification_snippets(group, max_review_snippets)
        substitution_examples = collect_substitution_examples(group, max_review_snippets)

        row: dict[str, Any] = {
            "doc_id": recipe_meta["doc_id"],
            "RecipeId": int(recipe_id),
            "title": recipe_meta.get("title", ""),
            "review_count_actual": int(len(group)),
            "mean_rating_actual": round(float(group["Rating"].dropna().mean()), 4) if group["Rating"].dropna().size else None,
            "median_rating_actual": round(float(group["Rating"].dropna().median()), 4) if group["Rating"].dropna().size else None,
            "min_rating_actual": float(group["Rating"].dropna().min()) if group["Rating"].dropna().size else None,
            "max_rating_actual": float(group["Rating"].dropna().max()) if group["Rating"].dropna().size else None,
            "review_summary_text": build_review_summary(group, max_review_summary_chars, max_review_snippets),
            "positive_review_snippets_json": json.dumps(positive_snippets, ensure_ascii=False),
            "negative_review_snippets_json": json.dumps(negative_snippets, ensure_ascii=False),
            "modification_review_snippets_json": json.dumps(modification_snippets, ensure_ascii=False),
            "substitution_examples_json": json.dumps(substitution_examples, ensure_ascii=False),
        }

        for name, value in positive_flags.items():
            row[f"positive_{name}_count"] = value
        for name, value in negative_flags.items():
            row[f"negative_{name}_count"] = value
        for name, value in modification_flags.items():
            row[f"modification_{name}_count"] = value

        row["positive_review_count"] = int(
            group["Review_clean"].fillna("").astype(str).map(
                lambda text: len(clean_text(text)) >= MIN_SNIPPET_CHARS
                and bool(POSITIVE_PATTERNS["success"].search(clean_text(text)) or POSITIVE_PATTERNS["easy"].search(clean_text(text)))
            ).sum()
        )
        row["negative_review_count"] = int(
            group["Review_clean"].fillna("").astype(str).map(
                lambda text: len(clean_text(text)) >= MIN_SNIPPET_CHARS
                and bool(NEGATIVE_PATTERNS["failure"].search(clean_text(text)))
            ).sum()
        )
        row["modification_review_count"] = int(
            group["Review_clean"].fillna("").astype(str).map(
                lambda text: len(clean_text(text)) >= MIN_SNIPPET_CHARS
                and any(pattern.search(clean_text(text)) for pattern in MODIFICATION_PATTERNS.values())
            ).sum()
        )
        row["has_reviews"] = int(row["review_count_actual"] > 0)
        row["review_support_score"] = compute_review_support_score(
            mean_rating=row["mean_rating_actual"],
            review_count=row["review_count_actual"],
            positive_success_count=row.get("positive_success_count", 0),
            negative_failure_count=row.get("negative_failure_count", 0),
        )

        rows.append(row)

    aggregated = pd.DataFrame(rows)
    LOGGER.info("Aggregated recipe rows: %d", len(aggregated))
    return aggregated


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

    try:
        recipes = load_recipes(args.recipes.resolve(), args.sample_recipe_limit)
        reviews = load_reviews(args.reviews.resolve())
        aggregated = aggregate_reviews(
            recipes=recipes,
            reviews=reviews,
            max_review_snippets=args.max_review_snippets,
            max_review_summary_chars=args.max_review_summary_chars,
        )
        save_outputs(aggregated, args.output_dir.resolve(), args.output_stem, args.format)
    except Exception as exc:
        LOGGER.error("Review aggregation failed: %s", exc, exc_info=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
