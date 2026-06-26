#!/usr/bin/env python3
"""Build-time normalization for raw recipe ingredient lines.

The online pipeline matches already-normalized ingredient phrases. This module is
only for corpus construction, where raw recipe lines still contain quantities,
units, parenthetical notes, and serving/preparation fragments.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rag.paths import RULES_DIR
from rag.pipeline.ingredient_normalization import (
    RUNTIME_RULES,
    apply_ingredient_rules,
    load_ingredient_rules,
    normalize_ingredient_term,
    normalize_recipe_terms,
)

BUILD_TIME_RULES_PATH = RULES_DIR / "buildtime_ingredient_preprocessing_rules.json"


def merge_ingredient_rule_profiles(*profiles: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for profile in profiles:
        for key, value in profile.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            elif isinstance(value, list) and isinstance(merged.get(key), list):
                merged[key] = list(dict.fromkeys([*merged[key], *value]))
            else:
                merged[key] = value
    return merged


def use_corpus_build_ingredient_rules() -> None:
    apply_ingredient_rules(
        merge_ingredient_rule_profiles(load_ingredient_rules(BUILD_TIME_RULES_PATH), RUNTIME_RULES)
    )


QUANTITY_RE = re.compile(
    "^\\s*(?:\\d+\\s*/\\s*\\d+|\\d+(?:\\.\\d+)?|[¼½¾⅓⅔⅛⅜⅝⅞]+)\\s*",
    re.IGNORECASE,
)
MEASURE_RE = re.compile(
    r"^\s*(?:"
    r"cups?|c|tablespoons?|tbsp|teaspoons?|tsp|ounces?|oz|pounds?|lbs?|grams?|g|kg|"
    r"fluid\s+ounces?|fl\s*oz|milliliters?|millilitres?|ml|liters?|litres?|l|"
    r"packages?|packets?|cans?|jars?|bottles?|boxes?|containers?|"
    r"slices?|pieces?|strips?|heads?|stalks?|quarts?|qt|pints?|gallons?|drops?|pinch|dash"
    r")\b\.?\s*(?:of\s+)?",
    re.IGNORECASE,
)
PAREN_RE = re.compile(r"\([^)]*\)")


def parse_json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]


def strip_leading_amount_and_unit(text: str) -> str:
    previous = None
    cleaned = text.strip()
    while cleaned and previous != cleaned:
        previous = cleaned
        cleaned = QUANTITY_RE.sub("", cleaned).strip()
        cleaned = MEASURE_RE.sub("", cleaned).strip()
    return cleaned


def normalize_ingredient_line(line: Any) -> list[str]:
    text = str(line or "").strip().lower()
    if not text:
        return []
    text = re.split(r"[,;]", text, maxsplit=1)[0]
    text = PAREN_RE.sub(" ", text)
    text = text.replace("-", " ")
    parts = re.split(r"\s+\bor\b\s+", text)
    terms: list[str] = []
    for part in parts:
        stripped = strip_leading_amount_and_unit(part)
        stripped = re.split(r"\s+\bfor\b\s+", stripped, maxsplit=1)[0].strip()
        stripped = re.split(r"\s+\benough\s+to\b\s+", stripped, maxsplit=1)[0].strip()
        if not stripped:
            continue
        normalized = normalize_ingredient_term(stripped)
        if normalized:
            terms.append(normalized)
    return list(dict.fromkeys(terms))


def normalize_ingredient_lines(value: Any) -> list[str]:
    lines = parse_json_list(value)
    terms: list[str] = []
    for line in lines:
        terms.extend(normalize_ingredient_line(line))
    return list(dict.fromkeys(terms))


def term_token_set(term: str) -> set[str]:
    return set(normalize_ingredient_term(term).split())


def phrase_matches_canonical(phrase: str, canonical_terms: list[str]) -> bool:
    phrase_tokens = term_token_set(phrase)
    if not phrase_tokens:
        return False
    canonical_token_sets = [term_token_set(term) for term in canonical_terms if term_token_set(term)]
    if phrase_tokens in canonical_token_sets:
        return True
    return any(tokens and tokens.issubset(phrase_tokens) for tokens in canonical_token_sets)


def canonical_sequence_text(canonical_terms: list[str]) -> str:
    return f" {' '.join(canonical_terms)} "


def term_sort_key(term: str) -> tuple[int, int]:
    tokens = term_token_set(term)
    return (len(tokens), len(term))


def remove_overlapping_subterms(terms: list[str]) -> list[str]:
    """Drop single/sub terms already represented by a longer ingredient phrase."""
    unique_terms = list(dict.fromkeys(term for term in terms if term))
    token_sets = {term: term_token_set(term) for term in unique_terms}
    filtered: list[str] = []
    for term in unique_terms:
        tokens = token_sets[term]
        if not tokens:
            continue
        is_subterm = any(
            term != other and tokens < other_tokens
            for other, other_tokens in token_sets.items()
            if other_tokens
        )
        if not is_subterm:
            filtered.append(term)
    return filtered


def should_trust_line_terms_as_fallback(canonical_terms: list[str], line_terms: list[str]) -> bool:
    """Use ingredient lines as fallback when canonical NER is clearly too sparse.

    The canonical field is normally the safer source, but some rows contain
    broken canonical text such as only a product name plus a unit token. In those
    cases the structured ingredient lines recover important ingredients without
    changing the retrieval text.
    """
    if not line_terms:
        return False
    if not canonical_terms:
        return True
    return len(canonical_terms) <= 1 and len(line_terms) >= 3


def longest_canonical_subsequence_phrase(phrase: str, canonical_terms: list[str]) -> str:
    phrase_words = normalize_ingredient_term(phrase).split()
    canonical_text = canonical_sequence_text(canonical_terms)
    canonical_tokens = set().union(*(term_token_set(term) for term in canonical_terms if term_token_set(term)))
    best = ""
    for start in range(len(phrase_words)):
        for end in range(len(phrase_words), start, -1):
            candidate = normalize_ingredient_term(" ".join(phrase_words[start:end]))
            if not candidate:
                continue
            if f" {candidate} " in canonical_text and len(candidate) > len(best):
                best = candidate
                break
            candidate_tokens = term_token_set(candidate)
            if (
                len(candidate_tokens) >= 2
                and candidate_tokens.issubset(canonical_tokens)
                and len(candidate) > len(best)
            ):
                best = candidate
    return best


def canonical_phrase_confirmed_terms(canonical_text: Any, ingredient_lines: Any) -> list[str]:
    """Normalize ingredients with canonical terms as source of truth.

    Recipe1M/Food.com NER-style canonical text is usually less noisy but may split
    multi-word ingredients. Ingredient lines retain phrase boundaries but include
    quantities and preparation notes. This function keeps only phrases supported
    by canonical terms, then appends remaining canonical terms as fallback.
    """
    canonical_terms = normalize_recipe_terms(canonical_text)
    if not canonical_terms:
        return normalize_ingredient_lines(ingredient_lines)

    line_terms = normalize_ingredient_lines(ingredient_lines)
    confirmed: list[str] = []
    covered_tokens: set[str] = set()
    canonical_tokens = set().union(*(term_token_set(term) for term in canonical_terms))

    for phrase in sorted(line_terms, key=term_sort_key, reverse=True):
        phrase_tokens = term_token_set(phrase)
        if not phrase_tokens:
            continue
        overlap = phrase_tokens & canonical_tokens
        if not overlap:
            continue
        if phrase_tokens.issubset(covered_tokens):
            continue
        confirmed_phrase = longest_canonical_subsequence_phrase(phrase, canonical_terms)
        if confirmed_phrase and (
            phrase_matches_canonical(confirmed_phrase, canonical_terms)
            or len(term_token_set(confirmed_phrase)) >= 2
        ):
            confirmed_tokens = term_token_set(confirmed_phrase)
            if confirmed_tokens.issubset(covered_tokens):
                continue
            confirmed.append(confirmed_phrase)
            covered_tokens.update(term_token_set(confirmed_phrase))

    if should_trust_line_terms_as_fallback(canonical_terms, line_terms):
        for phrase in sorted(line_terms, key=term_sort_key, reverse=True):
            phrase_tokens = term_token_set(phrase)
            if not phrase_tokens or phrase_tokens.issubset(covered_tokens):
                continue
            confirmed.append(phrase)
            covered_tokens.update(phrase_tokens)

    for term in canonical_terms:
        tokens = term_token_set(term)
        if not tokens or tokens.issubset(covered_tokens):
            continue
        confirmed.append(term)
        covered_tokens.update(tokens)

    return remove_overlapping_subterms(confirmed)
