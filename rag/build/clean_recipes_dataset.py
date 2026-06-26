#!/usr/bin/env python3
"""Clean the merged recipe dataset into a retrieval-ready intermediate corpus."""

from __future__ import annotations

import argparse
import ast
import html
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

LOGGER = logging.getLogger("clean_recipes")

LIST_COLUMNS = ("ingredients", "directions", "NER", "Images", "Keywords")
TEXT_COLUMNS = ("title", "Description", "RecipeCategory", "AuthorName", "source", "link")
NUMERIC_COLUMNS = (
    "AggregatedRating",
    "ReviewCount",
    "Calories",
    "FatContent",
    "SaturatedFatContent",
    "CholesterolContent",
    "SodiumContent",
    "CarbohydrateContent",
    "FiberContent",
    "SugarContent",
    "ProteinContent",
    "RecipeServings",
)
TIME_COLUMNS = ("CookTime", "PrepTime", "TotalTime")

WHITESPACE_RE = re.compile(r"\s+")
KEYWORD_SPLIT_RE = re.compile(r"'([^']*)'")
JSON_LIST_LIKE_RE = re.compile(r"^\s*\[.*\]\s*$", re.DOTALL)
HTTP_URL_RE = re.compile(r"https?://[^\s'\"]+")
TIME_CATEGORY_RE = re.compile(r"^<\s*\d+\s*(mins|min|minutes|hrs|hours)\s*$", re.IGNORECASE)
TITLE_ALPHA_RE = re.compile(r"[A-Za-z]")
TITLE_ONLY_SYMBOLS_RE = re.compile(r"^[^A-Za-z0-9]+$")
TITLE_URL_LIKE_RE = re.compile(r"https?://|www\.|\.(jpg|jpeg|png|gif|webp)\b", re.IGNORECASE)
FRACTION_RE = re.compile(r"\b\d+\s*/\s*\d+\b")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
PAREN_CONTENT_RE = re.compile(r"\([^)]*\)")
MEASURE_WORD_RE = re.compile(
    r"\b(?:cups?|tablespoons?|tbsp|teaspoons?|tsp|ounces?|oz|pounds?|lbs?|grams?|g|kg|ml|"
    r"liters?|litres?|l|packages?|packets?|cans?|jars?|bottles?|boxes?|containers?|"
    r"fluid|fl(?:\.)?|ounce|to taste|taste)\b",
    re.IGNORECASE,
)

DEFAULT_TAG_TAXONOMY: dict[str, list[str]] = {
    "difficulty_tags": ["easy", "beginner cook"],
    "audience_tags": ["kid friendly", "toddler friendly", "for large groups"],
    "diet_tags": [
        "low protein",
        "low cholesterol",
        "low saturated fat",
        "low sodium",
        "low carb",
        "low calorie",
        "vegan",
        "vegetarian",
        "egg free",
        "free of...",
        "healthy",
    ],
    "method_tags": ["microwave", "small appliance", "from scratch", "oven", "stove top", "refrigerator", "no cook", "mixer"],
    "cost_tags": ["inexpensive"],
    "meal_context_tags": ["weeknight", "brunch"],
    "occasion_tags": ["christmas", "thanksgiving", "potluck", "summer", "winter"],
    "flavor_tags": ["sweet", "savory", "spicy"],
    "cuisine_tags": [
        "greek",
        "australian",
        "european",
        "iraqi",
        "palestinian",
        "southwest asia (middle east)",
    ],
    "non_food_category_labels": [
        "easy",
        "beginner cook",
        "kid friendly",
        "for large groups",
        "low protein",
        "low cholesterol",
        "low saturated fat",
        "low sodium",
        "low carb",
        "low calorie",
        "healthy",
    ],
}

DEFAULT_CATEGORY_TAXONOMY: dict[str, list[str]] = {
    "broad_meal_categories": [
        "dessert",
        "beverages",
        "lunch/snacks",
        "breakfast",
        "breads",
        "one dish meal",
        "sauces",
        "soup",
        "salad",
        "appetizer",
        "main dish",
        "side dishes",
    ],
    "specific_dish_types": [
        "smoothies",
        "spreads",
        "ice cream",
        "frozen desserts",
        "gelatin",
        "quick breads",
        "drop cookies",
        "bar cookie",
        "candy",
        "pies and tarts",
        "savory pies",
    ],
    "main_ingredient_categories": [
        "chicken",
        "chicken breast",
        "meat",
        "pork",
        "beef",
        "fish",
        "seafood",
        "shrimp",
        "cheese",
        "potato",
        "spinach",
        "short grain rice",
        "rice",
        "beans",
        "vegetable",
        "fruit",
        "pumpkin",
        "turkey",
        "duck",
        "lamb/sheep",
        "pasta",
        "noodles",
        "broccoli",
        "corn",
        "tomato",
        "mushroom",
        "egg",
    ],
    "category_aliases": {
        "dessert": "Dessert",
        "desserts": "Dessert",
        "beverages": "Beverages",
        "lunch/snacks": "Lunch/Snacks",
        "breakfast": "Breakfast",
        "breads": "Breads",
        "one dish meal": "One Dish Meal",
        "sauces": "Sauces",
        "soup": "Soup",
        "salad": "Salad",
        "appetizer": "Appetizer",
        "appetizers": "Appetizer",
        "main dish": "Main Dish",
        "side dishes": "Side Dishes",
        "smoothies": "Smoothies",
        "spreads": "Spreads",
        "ice cream": "Ice Cream",
        "frozen desserts": "Frozen Desserts",
        "gelatin": "Gelatin",
        "quick breads": "Quick Breads",
        "drop cookies": "Drop Cookies",
        "bar cookie": "Bar Cookie",
        "candy": "Candy",
        "pies and tarts": "Pies and Tarts",
        "savory pies": "Savory Pies",
        "chicken": "Chicken",
        "chicken breast": "Chicken Breast",
        "meat": "Meat",
        "pork": "Pork",
        "beef": "Beef",
        "fish": "Fish",
        "seafood": "Seafood",
        "shrimp": "Shrimp",
        "cheese": "Cheese",
        "potato": "Potato",
        "spinach": "Spinach",
        "short grain rice": "Short Grain Rice",
        "rice": "Rice",
        "beans": "Beans",
        "vegetable": "Vegetable",
        "fruit": "Fruit",
        "pumpkin": "Pumpkin",
        "turkey": "Turkey",
        "duck": "Duck",
        "lamb/sheep": "Lamb/Sheep",
        "pasta": "Pasta",
        "noodles": "Noodles",
        "broccoli": "Broccoli",
        "corn": "Corn",
        "tomato": "Tomato",
        "mushroom": "Mushroom",
        "egg": "Egg",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean merged recipe dataset for RAG preprocessing")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("rag_dataset/dataset_merged.csv"),
        help="Path to the merged recipe CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for cleaned outputs",
    )
    parser.add_argument(
        "--output-stem",
        type=str,
        default="recipes_cleaned",
        help="Output filename stem without extension",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=Path("tag_taxonomy.json"),
        help="Path to tag taxonomy JSON config",
    )
    parser.add_argument(
        "--category-taxonomy",
        type=Path,
        default=Path("category_taxonomy.json"),
        help="Path to category taxonomy JSON config",
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
        help="Optional number of rows to process for debugging",
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


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


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
    text = normalize_whitespace(text)
    return "" if text.lower() == "nan" else text


def strip_bracketed_text(text: str) -> str:
    return text.strip().strip("[]")


def parse_keyword_like_string(raw: str) -> list[str]:
    matches = [normalize_whitespace(m) for m in KEYWORD_SPLIT_RE.findall(raw)]
    matches = [m for m in matches if m and m.lower() != "none"]
    if matches:
        return matches

    stripped = strip_bracketed_text(raw)
    if not stripped:
        return []

    parts = [normalize_whitespace(part) for part in stripped.split(",")]
    return [part for part in parts if part and part.lower() != "none"]


def split_concatenated_urls(raw: str) -> list[str]:
    urls = [normalize_whitespace(match) for match in HTTP_URL_RE.findall(raw)]
    return [url for url in urls if url]


def parse_maybe_list(value: Any, column: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, list):
        return [clean_text(item) for item in value if clean_text(item)]

    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return []

    if column == "Keywords":
        return parse_keyword_like_string(raw)
    if column == "Images":
        urls = split_concatenated_urls(raw)
        if urls:
            return urls

    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        if column == "Images" and JSON_LIST_LIKE_RE.match(raw):
            return parse_keyword_like_string(raw)
        return [clean_text(raw)] if clean_text(raw) else []

    if isinstance(parsed, (list, tuple, set)):
        return [clean_text(item) for item in parsed if clean_text(item)]
    if parsed is None:
        return []
    return [clean_text(parsed)] if clean_text(parsed) else []


def parse_time_to_minutes(value: Any) -> float | None:
    text = clean_text(value)
    if not text:
        return None

    match = re.fullmatch(
        r"P(?:(?P<days>\d+)D)?T?(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?",
        text,
    )
    if not match:
        return None

    days = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    total_minutes = days * 24 * 60 + hours * 60 + minutes + (seconds / 60.0)
    return round(total_minutes, 2)


def load_tag_taxonomy(path: Path) -> dict[str, set[str]]:
    taxonomy: dict[str, Any] = DEFAULT_TAG_TAXONOMY
    if path.exists():
        LOGGER.info("Loading tag taxonomy: %s", path)
        taxonomy = json.loads(path.read_text(encoding="utf-8"))
    else:
        LOGGER.warning("Tag taxonomy file not found, using defaults: %s", path)

    normalized: dict[str, set[str]] = {}
    for key, values in taxonomy.items():
        if not isinstance(values, list):
            continue
        normalized[key] = {clean_text(value).lower() for value in values if clean_text(value)}
    return normalized


def load_category_taxonomy(path: Path) -> dict[str, Any]:
    taxonomy: dict[str, Any] = DEFAULT_CATEGORY_TAXONOMY
    if path.exists():
        LOGGER.info("Loading category taxonomy: %s", path)
        taxonomy = json.loads(path.read_text(encoding="utf-8"))
    else:
        LOGGER.warning("Category taxonomy file not found, using defaults: %s", path)

    normalized: dict[str, Any] = {}
    for key, values in taxonomy.items():
        if key == "category_aliases" and isinstance(values, dict):
            normalized[key] = {
                clean_text(alias_key).lower(): clean_text(alias_value)
                for alias_key, alias_value in values.items()
                if clean_text(alias_key) and clean_text(alias_value)
            }
            continue
        if not isinstance(values, list):
            continue
        normalized[key] = {clean_text(value).lower() for value in values if clean_text(value)}
    return normalized


def canonicalize_category_label(value: str, category_taxonomy: dict[str, Any]) -> str:
    cleaned = clean_text(value)
    if not cleaned:
        return ""

    alias_map = category_taxonomy.get("category_aliases", {})
    if isinstance(alias_map, dict):
        canonical = alias_map.get(cleaned.lower(), "")
        if canonical:
            return canonical
    return cleaned


def clean_category_with_taxonomy(value: Any, taxonomy: dict[str, set[str]]) -> str:
    category = clean_text(value)
    if not category:
        return ""

    lowered = category.lower()
    if TIME_CATEGORY_RE.fullmatch(category):
        return ""
    if lowered in taxonomy.get("non_food_category_labels", set()):
        return ""
    return category


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = clean_text(value)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def classify_metadata_tags(
    recipe_category: Any,
    keywords: list[str],
    taxonomy: dict[str, set[str]],
) -> dict[str, list[str]]:
    category_text = clean_text(recipe_category)
    candidates: list[str] = []
    if category_text:
        candidates.append(category_text)
    candidates.extend(keywords)
    candidates = unique_preserve_order(candidates)

    buckets: dict[str, list[str]] = {
        "time_tags": [],
        "difficulty_tags": [],
        "diet_tags": [],
        "audience_tags": [],
        "method_tags": [],
        "cost_tags": [],
        "meal_context_tags": [],
        "occasion_tags": [],
        "flavor_tags": [],
        "cuisine_tags": [],
        "other_tags": [],
    }

    for tag in candidates:
        lowered = tag.lower()
        if TIME_CATEGORY_RE.fullmatch(tag):
            buckets["time_tags"].append(tag)
        elif lowered in taxonomy.get("difficulty_tags", set()):
            buckets["difficulty_tags"].append(tag)
        elif lowered in taxonomy.get("diet_tags", set()):
            buckets["diet_tags"].append(tag)
        elif lowered in taxonomy.get("audience_tags", set()):
            buckets["audience_tags"].append(tag)
        elif lowered in taxonomy.get("method_tags", set()):
            buckets["method_tags"].append(tag)
        elif lowered in taxonomy.get("cost_tags", set()):
            buckets["cost_tags"].append(tag)
        elif lowered in taxonomy.get("meal_context_tags", set()):
            buckets["meal_context_tags"].append(tag)
        elif lowered in taxonomy.get("occasion_tags", set()):
            buckets["occasion_tags"].append(tag)
        elif lowered in taxonomy.get("flavor_tags", set()):
            buckets["flavor_tags"].append(tag)
        elif lowered in taxonomy.get("cuisine_tags", set()):
            buckets["cuisine_tags"].append(tag)
        else:
            buckets["other_tags"].append(tag)

    return buckets


def derive_category_fields(
    source_primary_category: Any,
    other_tags: list[str],
    category_taxonomy: dict[str, Any],
) -> dict[str, str]:
    source_value = clean_text(source_primary_category)
    other_values = unique_preserve_order(other_tags)

    candidates = []
    if source_value:
        candidates.append(source_value)
    candidates.extend(other_values)

    broad_meal_category = ""
    specific_dish_type = ""
    main_ingredient_category = ""
    broad_meal_categories = category_taxonomy.get("broad_meal_categories", set())
    specific_dish_types = category_taxonomy.get("specific_dish_types", set())
    main_ingredient_categories = category_taxonomy.get("main_ingredient_categories", set())

    for candidate in candidates:
        canonical = canonicalize_category_label(candidate, category_taxonomy)
        lowered = canonical.lower()
        if not broad_meal_category and lowered in broad_meal_categories:
            broad_meal_category = canonical
        if not specific_dish_type and lowered in specific_dish_types:
            specific_dish_type = canonical
        if not main_ingredient_category and lowered in main_ingredient_categories:
            main_ingredient_category = canonical
        if broad_meal_category and specific_dish_type and main_ingredient_category:
            break

    return {
        "source_primary_category": source_value,
        "broad_meal_category": broad_meal_category,
        "specific_dish_type": specific_dish_type,
        "main_ingredient_category": main_ingredient_category,
    }


def clean_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def list_to_text(values: Iterable[str]) -> str:
    cleaned = [clean_text(value) for value in values]
    cleaned = [value for value in cleaned if value]
    return " ".join(cleaned)


def classify_title_quality(title: Any) -> str:
    text = clean_text(title)
    if not text:
        return "blank"
    if TITLE_URL_LIKE_RE.search(text):
        return "url_like"
    if TITLE_ONLY_SYMBOLS_RE.fullmatch(text):
        return "symbols_only"
    if not TITLE_ALPHA_RE.search(text):
        return "no_alpha"
    return "ok"


def build_clean_title(row: pd.Series) -> str:
    title = clean_text(row.get("title_clean", ""))
    if classify_title_quality(title) == "ok":
        return title

    category = next(
        (
            clean_text(row.get(column, ""))
            for column in (
                "specific_dish_type",
                "broad_meal_category",
                "source_primary_category",
                "main_ingredient_category",
            )
            if clean_text(row.get(column, ""))
        ),
        "",
    )
    ingredients = row.get("NER_list", [])
    if not isinstance(ingredients, list):
        ingredients = []
    ingredient_text = ", ".join(unique_preserve_order(ingredients)[:3])

    if category and ingredient_text:
        return f"{category} with {ingredient_text}"
    if ingredient_text:
        return f"Recipe with {ingredient_text}"
    if category:
        return f"{category} Recipe"
    return "Untitled Recipe"


def clean_fallback_ingredient(value: Any) -> str:
    text = clean_text(value).lower()
    if not text:
        return ""

    text = PAREN_CONTENT_RE.sub(" ", text)
    text = FRACTION_RE.sub(" ", text)
    text = NUMBER_RE.sub(" ", text)
    text = MEASURE_WORD_RE.sub(" ", text)
    text = re.sub(r"\b(?:of|or|and|depending on the number of|depending on)\b", " ", text)
    text = re.sub(r"[^a-z0-9/&' -]+", " ", text)
    return normalize_whitespace(text).strip(" ,-")


def choose_canonical_ingredients(row: pd.Series) -> list[str]:
    ner_values = row.get("NER_list", [])
    if isinstance(ner_values, list) and ner_values:
        return unique_preserve_order(ner_values)

    ingredient_values = row.get("ingredients_list", [])
    if isinstance(ingredient_values, list):
        cleaned_values = [clean_fallback_ingredient(value) for value in ingredient_values]
        return unique_preserve_order(value for value in cleaned_values if value)
    return []


def build_doc_id(source: str, link: str) -> str:
    return f"{source}::{link}"


def load_dataset(input_path: Path, sample: int) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    LOGGER.info("Loading dataset: %s", input_path)
    df = pd.read_csv(input_path)
    if sample > 0:
        df = df.head(sample).copy()
        LOGGER.info("Using sample mode: first %d rows", sample)
    return df


def clean_dataset(
    df: pd.DataFrame,
    taxonomy: dict[str, set[str]],
    category_taxonomy: dict[str, set[str]],
) -> pd.DataFrame:
    LOGGER.info("Initial shape: %s", df.shape)

    for column in TEXT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(clean_text)

    for column in LIST_COLUMNS:
        if column in df.columns:
            LOGGER.info("Parsing list-like column: %s", column)
            df[f"{column}_list"] = df[column].map(lambda value, c=column: parse_maybe_list(value, c))

    for column in ("NER", "Images", "Keywords"):
        list_col = f"{column}_list"
        if list_col in df.columns:
            df[list_col] = df[list_col].map(unique_preserve_order)

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = clean_numeric_series(df[column])

    for column in TIME_COLUMNS:
        if column in df.columns:
            df[f"{column}_minutes"] = df[column].map(parse_time_to_minutes)

    if "DatePublished" in df.columns:
        df["DatePublished_ts"] = pd.to_datetime(df["DatePublished"], errors="coerce", utc=True)

    if "title" in df.columns:
        df["title_clean"] = df["title"].map(clean_text)
        df["title_quality"] = df["title_clean"].map(classify_title_quality)
    if "Description" in df.columns:
        df["description_clean"] = df["Description"].map(clean_text)
    if "RecipeCategory" in df.columns:
        df["recipe_category_clean"] = df["RecipeCategory"].map(
            lambda value: clean_category_with_taxonomy(value, taxonomy)
        )

    if "ingredients_list" in df.columns:
        df["ingredients_text"] = df["ingredients_list"].map(list_to_text)
    if "directions_list" in df.columns:
        df["directions_text"] = df["directions_list"].map(list_to_text)
    if "NER_list" in df.columns:
        df["ner_text"] = df["NER_list"].map(list_to_text)
    if "Keywords_list" in df.columns:
        df["keywords_text"] = df["Keywords_list"].map(list_to_text)

    if "Keywords_list" in df.columns:
        LOGGER.info("Classifying metadata tags from RecipeCategory and Keywords")
        tag_data = [
            classify_metadata_tags(
                recipe_category=recipe_category,
                keywords=keywords,
                taxonomy=taxonomy,
            )
            for recipe_category, keywords in zip(
                df.get("RecipeCategory", pd.Series([""] * len(df))),
                df["Keywords_list"],
            )
        ]
        for bucket in (
            "time_tags",
            "difficulty_tags",
            "diet_tags",
            "audience_tags",
            "method_tags",
            "cost_tags",
            "meal_context_tags",
            "occasion_tags",
            "flavor_tags",
            "cuisine_tags",
            "other_tags",
        ):
            df[f"{bucket}_list"] = [entry[bucket] for entry in tag_data]
            df[f"{bucket}_json"] = df[f"{bucket}_list"].map(json.dumps)
            df[f"{bucket}_text"] = df[f"{bucket}_list"].map(list_to_text)

    if "recipe_category_clean" in df.columns:
        category_rows = [
            derive_category_fields(
                source_primary_category=source_primary_category,
                other_tags=other_tags,
                category_taxonomy=category_taxonomy,
            )
            for source_primary_category, other_tags in zip(
                df["recipe_category_clean"],
                df.get("other_tags_list", pd.Series([[] for _ in range(len(df))])),
            )
        ]
        df["source_primary_category"] = [row["source_primary_category"] for row in category_rows]
        df["broad_meal_category"] = [row["broad_meal_category"] for row in category_rows]
        df["specific_dish_type"] = [row["specific_dish_type"] for row in category_rows]
        df["main_ingredient_category"] = [row["main_ingredient_category"] for row in category_rows]

    if "title_clean" in df.columns:
        df["title_clean"] = df.apply(build_clean_title, axis=1)

    if {"NER_list", "ingredients_list"}.intersection(df.columns):
        df["canonical_ingredients"] = df.apply(choose_canonical_ingredients, axis=1)
        df["canonical_ingredients_json"] = df["canonical_ingredients"].map(
            lambda values: json.dumps(values, ensure_ascii=False)
        )
        df["canonical_ingredients_text"] = df["canonical_ingredients"].map(list_to_text)

    if {"source", "link"}.issubset(df.columns):
        df["doc_id"] = [
            build_doc_id(source=source, link=link)
            for source, link in zip(df["source"], df["link"])
        ]

    before = len(df)
    if "link" in df.columns:
        df = df.drop_duplicates(subset=["link"], keep="first").copy()
        LOGGER.info("Exact dedup by link removed %d rows", before - len(df))

    list_columns = [f"{column}_list" for column in LIST_COLUMNS if f"{column}_list" in df.columns]
    for column in list_columns:
        df[f"{column}_json"] = df[column].map(json.dumps)

    preferred_columns = [
        "doc_id",
        "source",
        "link",
        "RecipeId",
        "title",
        "title_clean",
        "title_quality",
        "canonical_ingredients_json",
        "canonical_ingredients_text",
        "ingredients_list_json",
        "directions_list_json",
        "NER_list_json",
        "Images_list_json",
        "Keywords_list_json",
        "ingredients_text",
        "directions_text",
        "ner_text",
        "keywords_text",
        "Description",
        "description_clean",
        "RecipeCategory",
        "source_primary_category",
        "broad_meal_category",
        "specific_dish_type",
        "main_ingredient_category",
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
        "CookTime",
        "CookTime_minutes",
        "PrepTime",
        "PrepTime_minutes",
        "TotalTime",
        "TotalTime_minutes",
        "DatePublished",
        "DatePublished_ts",
        "AggregatedRating",
        "ReviewCount",
        "Calories",
        "ProteinContent",
        "RecipeServings",
    ]
    ordered = [column for column in preferred_columns if column in df.columns]
    remainder = [column for column in df.columns if column not in ordered]
    df = df[ordered + remainder]

    redundant_output_columns = [
        "canonical_ingredients",
        "recipe_category_clean",
        "dish_type_category_text",
        "main_ingredient_category_text",
    ]
    redundant_output_columns.extend(
        column for column in df.columns if column.endswith("_list")
    )
    df = df.drop(columns=[col for col in redundant_output_columns if col in df.columns])

    LOGGER.info("Cleaned shape: %s", df.shape)
    return df


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
        df = load_dataset(args.input.resolve(), args.sample)
        taxonomy = load_tag_taxonomy(args.taxonomy.resolve())
        category_taxonomy = load_category_taxonomy(args.category_taxonomy.resolve())
        cleaned = clean_dataset(df, taxonomy, category_taxonomy)
        save_outputs(cleaned, args.output_dir.resolve(), args.output_stem, args.format)
    except Exception as exc:
        LOGGER.error("Dataset cleaning failed: %s", exc, exc_info=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
