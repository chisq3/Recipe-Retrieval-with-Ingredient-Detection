#!/usr/bin/env python3
"""Ingredient term parsing and availability matching helpers.

Recipe ingredient terms are canonicalized during corpus build into the
``normalized_ingredient_terms`` column. Runtime code parses that column and
compares it with request ingredients canonicalized by request normalization.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rag.paths import RULES_DIR

RUNTIME_RULES_PATH = RULES_DIR / "runtime_ingredient_reasoning_rules.json"


def load_ingredient_rules(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


RUNTIME_RULES = load_ingredient_rules(RUNTIME_RULES_PATH)
PANTRY_EQUIVALENTS: dict[str, list[str]] = {}
KNOWN_INGREDIENT_PHRASES: list[str] = []
MEASURE_TOKEN_TERMS: set[str] = set()
NON_INGREDIENT_TERMS: set[str] = set()
PROCESSED_PRODUCT_SUFFIXES: tuple[str, ...] = ()
SINGULAR_INGREDIENTS: dict[str, str] = {}


def apply_ingredient_rules(rules: dict[str, Any]) -> None:
    """Activate one ingredient-rule profile for the shared normalization helpers."""
    global PANTRY_EQUIVALENTS, KNOWN_INGREDIENT_PHRASES, MEASURE_TOKEN_TERMS, NON_INGREDIENT_TERMS
    global PROCESSED_PRODUCT_SUFFIXES, SINGULAR_INGREDIENTS
    PANTRY_EQUIVALENTS = rules.get("pantry_equivalents", {})
    KNOWN_INGREDIENT_PHRASES = rules.get("known_ingredient_phrases", [])
    MEASURE_TOKEN_TERMS = set(rules.get("measure_token_terms", []))
    NON_INGREDIENT_TERMS = set(rules.get("non_ingredient_terms", []))
    PROCESSED_PRODUCT_SUFFIXES = tuple(rules.get("processed_product_suffixes", []))
    SINGULAR_INGREDIENTS = rules.get("singular_ingredients", {})


def use_runtime_ingredient_rules() -> None:
    apply_ingredient_rules(RUNTIME_RULES)


# Default import behavior is runtime-safe. Raw recipe-line cleanup lives in
# rag.build.ingredient_line_normalization for corpus construction.
use_runtime_ingredient_rules()


def normalize_basic_text(text: Any) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", str(text or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def singularize_term(term: str) -> str:
    term = term.strip().lower()
    if term in SINGULAR_INGREDIENTS:
        return str(SINGULAR_INGREDIENTS[term])
    if term.endswith("ies") and len(term) > 3:
        return f"{term[:-3]}y"
    if term.endswith("s") and len(term) > 3 and not term.endswith("ss"):
        return term[:-1]
    return term


def normalize_ingredient_term(term: str) -> str:
    """Canonicalize one short ingredient term for build/request boundaries."""
    cleaned = normalize_basic_text(term)
    if not cleaned:
        return ""
    tokens: list[str] = []
    for token in cleaned.split():
        singular = singularize_term(token)
        if (
            token.isdigit()
            or token in NON_INGREDIENT_TERMS
            or singular in NON_INGREDIENT_TERMS
            or token in MEASURE_TOKEN_TERMS
            or singular in MEASURE_TOKEN_TERMS
        ):
            continue
        tokens.append(token)
    if not tokens:
        return ""
    return " ".join(singularize_term(token) for token in tokens)


def normalize_recipe_terms(text: Any) -> list[str]:
    normalized_text = normalize_basic_text(text)
    if not normalized_text:
        return []

    phrase_terms: list[str] = []
    remainder = f" {normalized_text} "
    for phrase in sorted(KNOWN_INGREDIENT_PHRASES, key=len, reverse=True):
        normalized_phrase = normalize_basic_text(phrase)
        if not normalized_phrase:
            continue
        pattern = rf"\b{re.escape(normalized_phrase)}\b"
        if re.search(pattern, remainder):
            normalized_term = normalize_ingredient_term(normalized_phrase)
            if normalized_term:
                phrase_terms.append(normalized_term)
            remainder = re.sub(pattern, " ", remainder)

    raw_terms = [normalize_ingredient_term(term) for term in remainder.split() if term]
    filtered_terms = [term for term in raw_terms if term and len(term) > 1]
    return list(dict.fromkeys([*phrase_terms, *filtered_terms]))


def serialize_ingredient_terms(terms: list[str]) -> str:
    return " | ".join(dict.fromkeys(term for term in terms if term))


def parse_ingredient_terms(value: Any) -> list[str]:
    """Parse the serialized ``normalized_ingredient_terms`` column.

    The column already contains normalized ingredient phrases separated by
    ``|``. This function only parses/deduplicates that build-time output; it
    does not re-normalize full recipe text.
    """
    text = str(value or "").strip()
    if not text:
        return []
    return list(dict.fromkeys(part.strip() for part in text.split("|") if part.strip()))


def contains_term(text: str, term: str) -> bool:
    normalized_text = f" {normalize_basic_text(text)} "
    normalized_term = normalize_basic_text(term)
    if not normalized_term:
        return False
    return bool(re.search(rf"\b{re.escape(normalized_term)}\b", normalized_text))


def equivalent_ingredient(term: str, available: str) -> bool:
    normalized_term = normalize_ingredient_term(term)
    normalized_available = normalize_ingredient_term(available)
    if not normalized_term or not normalized_available:
        return False
    if normalized_term == normalized_available:
        return True
    if contains_term(normalized_term, normalized_available) or contains_term(normalized_available, normalized_term):
        return True
    variants = PANTRY_EQUIVALENTS.get(normalized_available, [])
    return any(
        normalized_term == normalize_ingredient_term(variant)
        or contains_term(normalized_term, variant)
        or contains_term(variant, normalized_term)
        for variant in variants
    )


def _is_processed_extension(base: str, other: str) -> bool:
    """True if ``other`` is ``base`` followed by a processed-product suffix,
    e.g. base='fish', other='fish sauce'."""
    parts = other.split()
    return (
        len(parts) >= 2
        and parts[-1] in PROCESSED_PRODUCT_SUFFIXES
        and " ".join(parts[:-1]) == base
    )


def availability_covers(user_term: str, recipe_term: str) -> bool:
    """Phase-D availability matcher: does having ``user_term`` cover a recipe's need
    for ``recipe_term``?

    Narrow on purpose (cookability must not be over-estimated): a raw ingredient does
    NOT cover its processed product (fish vs fish sauce, chicken vs chicken broth),
    but generic<->cut/form still covers both ways (chicken <-> chicken breast) via
    containment. Separate from ``equivalent_ingredient`` (kept for grounding).
    """
    # Request and recipe terms are canonicalized upstream; this guard keeps the
    # matcher robust for direct calls, tests, and hand-built fixtures.
    u = normalize_ingredient_term(user_term)
    r = normalize_ingredient_term(recipe_term)
    if not u or not r:
        return False
    if u == r:
        return True
    # Block the raw<->processed-product relation in either direction.
    if _is_processed_extension(u, r) or _is_processed_extension(r, u):
        return False
    # Otherwise fall back to bidirectional containment (generic <-> specific cut/form).
    return equivalent_ingredient(user_term, recipe_term) or equivalent_ingredient(recipe_term, user_term)
