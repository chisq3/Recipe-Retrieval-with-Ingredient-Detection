#!/usr/bin/env python3
"""Rule-based constraint checks for retrieved recipes.

The LLM extracts structured constraints. This module decides whether a
retrieved recipe violates those constraints. Keeping this separate prevents
method-specific checks from spreading across the pipeline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


from rag.paths import RULES_DIR

RULES_PATH = RULES_DIR / "runtime_constraint_checker_rules.json"


def load_constraint_rules(path: Path = RULES_PATH) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


CONSTRAINT_RULES = load_constraint_rules()
# Generic method-exclude detectors: {method: {recipe_terms: [...]}}. Adding a new
# excludable cooking method is now a rules-file edit, not a code change.
METHOD_EXCLUDE_DETECTORS = CONSTRAINT_RULES.get("method_exclude_detectors", {})
DIET_CONFLICT_TERMS = CONSTRAINT_RULES.get("diet_conflict_terms", {})
INGREDIENT_EXCLUDE_GROUPS = CONSTRAINT_RULES.get("ingredient_exclude_groups", {})


def term_variants(term: str) -> list[str]:
    cleaned = term.lower().strip()
    if not cleaned:
        return []
    variants = [cleaned]
    if cleaned.endswith("ies") and len(cleaned) > 3:
        variants.append(f"{cleaned[:-3]}y")
    elif cleaned.endswith("es") and len(cleaned) > 2:
        variants.append(cleaned[:-2])
    elif cleaned.endswith("s") and len(cleaned) > 1:
        variants.append(cleaned[:-1])
    else:
        variants.append(f"{cleaned}s")
    return list(dict.fromkeys(variants))


def contains_term(text: str, term: str) -> bool:
    if not term:
        return False
    lowered = text.lower()
    return any(bool(re.search(rf"\b{re.escape(variant)}\b", lowered)) for variant in term_variants(term))


# Negation cues that mean an ingredient is ABSENT even though its name appears in
# the text: "peanut-free", "peanut free", "no peanut", "without peanut",
# "free of/from peanut". General linguistic patterns, NOT per-ingredient rules.
_NEG_AFTER = re.compile(r"^[\s-]*(?:free|less)\b")
_NEG_BEFORE = re.compile(r"(?:\b(?:no|without)\s+|\bfree\s+(?:of|from)\s+)$")


def term_present_unnegated(text: str, term: str) -> bool:
    """Like contains_term, but an occurrence in a negated context does not count.
    Present iff at least one occurrence of the term is NOT negated. This keeps
    derivative matches (e.g. 'peanut' in 'peanut butter') while ignoring negated
    claims (e.g. 'Peanut-Free Cookies', 'dairy-free')."""
    if not term:
        return False
    lowered = text.lower()
    for variant in term_variants(term):
        for m in re.finditer(rf"\b{re.escape(variant)}\b", lowered):
            after = lowered[m.end():m.end() + 8]
            before = lowered[max(0, m.start() - 14):m.start()]
            if _NEG_AFTER.match(after) or _NEG_BEFORE.search(before):
                continue
            return True
    return False


# Domain vocabularies live in runtime_constraint_checker_rules.json. The code
# below only applies the matching logic.
NON_DAIRY_QUALIFIERS = tuple(CONSTRAINT_RULES.get("non_dairy_qualifiers", []))
DAIRY_TERMS_WITH_PLANT_ANALOGUE = tuple(CONSTRAINT_RULES.get("dairy_terms_with_plant_analogue", []))


def _is_non_dairy_compound(after_tokens: list[str], prev_word: str, dairy_term: str) -> bool:
    """Whether a specific occurrence of ``dairy_term`` is a NON-dairy compound:
    '<plant> milk/cream/butter' (qualifier right before) or 'cream of <plant>'."""
    if prev_word in NON_DAIRY_QUALIFIERS:
        return True
    if (
        dairy_term == "cream"
        and len(after_tokens) >= 2
        and after_tokens[0] == "of"
        and after_tokens[1].strip(",.;:") in NON_DAIRY_QUALIFIERS
    ):
        return True
    return False


def _dairy_term_present_as_dairy(text: str, dairy_term: str) -> bool:
    """True if ``dairy_term`` appears as REAL dairy at least once, i.e. an occurrence
    that is not a non-dairy compound."""
    lowered = text.lower()
    for m in re.finditer(rf"\b{re.escape(dairy_term)}\b", lowered):
        prev = lowered[: m.start()].split()
        prev_word = prev[-1] if prev else ""
        after_tokens = lowered[m.end():].split()
        if not _is_non_dairy_compound(after_tokens, prev_word, dairy_term):
            return True
    return False


def recipe_contains_excluded(text: str, term: str) -> bool:
    """term_present_unnegated plus the non-dairy-compound exception. The exception is
    keyed on the DAIRY term being checked (milk/cream/butter), so it never weakens
    allergen exclusion of the qualifier itself: excluding 'dairy' allows peanut butter,
    but excluding 'peanut' still blocks it. cheese/yogurt and all other terms keep the
    broad word-boundary behaviour, so safety recall is preserved."""
    if not term_present_unnegated(text, term):
        return False
    normalized = str(term).strip().lower()
    if normalized in DAIRY_TERMS_WITH_PLANT_ANALOGUE:
        return _dairy_term_present_as_dairy(text, normalized)
    return True


def expanded_excluded_ingredients(ingredients: list[str]) -> list[str]:
    expanded: list[str] = []
    for ingredient in ingredients:
        normalized = ingredient.lower().strip()
        if not normalized:
            continue
        expanded.append(normalized)
        expanded.extend(str(item).lower().strip() for item in INGREDIENT_EXCLUDE_GROUPS.get(normalized, []))
    return list(dict.fromkeys(item for item in expanded if item))


def recipe_uses_method(recipe: dict[str, Any], method: str) -> bool:
    """Whether the recipe appears to use the given cooking method, per its detector terms."""
    terms = [str(t).strip() for t in METHOD_EXCLUDE_DETECTORS.get(method, {}).get("recipe_terms", []) if str(t).strip()]
    if not terms:
        return False
    instruction_text = str(recipe.get("instruction_text", "") or "").lower()
    method_text = str(recipe.get("method_tags_text", "") or "").lower()
    combined = f"{instruction_text} {method_text}"
    return any(contains_term(combined, term) for term in terms)


def recipe_text_for_constraints(recipe: dict[str, Any]) -> str:
    fields = [
        "title",
        "canonical_ingredients_text",
        "ingredient_text",
        "metadata_text",
        "diet_tags_text",
        "method_tags_text",
        "instruction_text",
    ]
    return " ".join(str(recipe.get(field, "") or "").lower() for field in fields)


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip().lower()] if value.strip() else []
    return [str(value).strip().lower()] if str(value).strip() else []


def normalize_constraints(constraints: dict[str, Any] | None, query: str = "") -> dict[str, Any]:
    constraints = constraints if isinstance(constraints, dict) else {}
    method_include = as_list(constraints.get("method_include"))
    method_exclude = as_list(constraints.get("method_exclude"))
    ingredient_exclude = as_list(constraints.get("ingredient_exclude"))

    diet = constraints.get("diet")
    cuisine = constraints.get("cuisine")
    max_time = constraints.get("max_time")
    return {
        "method_include": list(dict.fromkeys(method_include)),
        "method_exclude": list(dict.fromkeys(method_exclude)),
        "ingredient_exclude": list(dict.fromkeys(ingredient_exclude)),
        "diet": str(diet).strip().lower() if diet is not None and str(diet).strip() else None,
        "cuisine": str(cuisine).strip().lower() if cuisine is not None and str(cuisine).strip() else None,
        "max_time": max_time,
        "cost": constraints.get("cost"),
    }


def check_constraints(recipe: dict[str, Any], constraints: dict[str, Any] | None, query: str = "") -> dict[str, Any]:
    normalized = normalize_constraints(constraints, query=query)
    text = recipe_text_for_constraints(recipe)
    violations: list[str] = []
    violated_fields: list[str] = []

    violated_methods: list[str] = []
    for method in normalized["method_exclude"]:
        if recipe_uses_method(recipe, method):
            violations.append(f"User excluded cooking method '{method}', but the recipe appears to use it.")
            violated_fields.append(f"method_exclude:{method}")
            violated_methods.append(method)

    for ingredient in expanded_excluded_ingredients(normalized["ingredient_exclude"]):
        if recipe_contains_excluded(text, ingredient):
            violations.append(f"User excluded ingredient '{ingredient}', but the recipe contains it.")
            violated_fields.append(f"ingredient_exclude:{ingredient}")

    if normalized["diet"] in DIET_CONFLICT_TERMS:
        matched_conflicts = [term for term in DIET_CONFLICT_TERMS[normalized["diet"]] if recipe_contains_excluded(text, term)]
        if matched_conflicts:
            violations.append(f"User requested {normalized['diet']}, but the recipe appears to contain conflicting ingredients.")
            violated_fields.append(f"diet:{normalized['diet']}")

    return {
        "normalized_constraints": normalized,
        "hard_constraint_violations": violations,
        "violated_fields": violated_fields,
        "has_hard_violation": bool(violations),
        "violated_methods": violated_methods,
    }


def has_hard_constraint(constraints: dict[str, Any] | None, query: str = "") -> bool:
    """True iff the (normalized) constraints contain a HARD component the gate acts on:
    diet OR ingredient_exclude OR method_exclude. method_include is a PREFERENCE and is
    never part of the hard gate (handled later by candidate selection)."""
    n = normalize_constraints(constraints, query=query)
    return bool(n["diet"] or n["ingredient_exclude"] or n["method_exclude"])


def to_checker_recipe(row: dict[str, Any]) -> dict[str, Any]:
    """Map a corpus/retrieval row to the field names check_constraints reads, so the runtime
    gate decides IDENTICALLY to the benchmark. Mirrors run_gate_rrf.recipe_for_checker; the
    benchmark and runtime both feed gate inputs through this. Returns a new dict (no mutation)."""
    r = dict(row)
    r["method_tags_text"] = row.get("normalized_method_tags", "") or row.get("method_tags_text", "") or ""
    r["diet_tags_text"] = row.get("normalized_diet_tags", "") or row.get("diet_tags_text", "") or ""
    r["ingredient_text"] = (
        row.get("ingredients_text", "") or row.get("ingredient_text", "")
        or row.get("canonical_ingredients_text", "") or ""
    )
    return r


def apply_hard_constraint_gate(
    ordered_candidates: list[dict[str, Any]],
    constraints: dict[str, Any] | None,
    query: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic hard-constraint gate SHARED by the Phase-F benchmark and the runtime.

    Filters an ALREADY-ORDERED candidate list, dropping recipes the deterministic checker
    flags as violating a HARD constraint (diet / ingredient_exclude / method_exclude ONLY -
    method_include is a preference and is NEVER gated). It does NOT retrieve, rank, score
    availability, or apply preferences; the CALLER owns retrieval depth and any post-filter
    ranking/selection (benchmark: take top-10 survivors; runtime: rich priority-selection on
    survivors).

    Args:
        ordered_candidates: ordered list of recipe dicts (checker-ready fields per
            recipe_text_for_constraints: title, canonical_ingredients_text / ingredient_text,
            instruction_text, method_tags_text, diet_tags_text, metadata_text), ranked best-first.
        constraints: raw extracted constraints dict (normalized internally via
            normalize_constraints).
        query: raw user query, passed UNCHANGED to check_constraints (no raw-query keyword
            rules are introduced here).

    Returns:
        (survivors, rejected)
        - survivors: the SAME recipe dict objects that passed, in input order.
        - rejected: one dict per dropped candidate, in input order, each with field names
          aligned to gate_decision_log so the benchmark log can be built directly:
            {"doc_id", "original_rank" (1-based), "violated_fields", "violated_methods"}.

    Guarantees:
        - No hard constraint after normalization -> NO-OP: survivors == list(ordered_candidates)
          (same order), rejected == [].
        - Does NOT mutate the input list or any candidate dict; order is preserved.
        - Pure / deterministic: no LLM, no I/O beyond the already-loaded rules asset.

    Safety scope (deliberately bounded, NOT a universal guarantee):
        A candidate is dropped iff check_constraints reports has_hard_violation. This bounds
        what is served to "candidates the checker did NOT flag"; it does NOT guarantee the
        absence of TRUE violations the checker fails to detect (checker false-negatives).
    """
    candidates = list(ordered_candidates)  # shallow copy: never mutate the caller's list
    if not has_hard_constraint(constraints, query=query):
        return candidates, []  # NO-OP: preserve order, nothing rejected

    survivors: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for rank, recipe in enumerate(candidates, start=1):
        result = check_constraints(recipe, constraints, query)
        if result["has_hard_violation"]:
            rejected.append({
                "doc_id": recipe.get("doc_id"),
                "original_rank": rank,
                "violated_fields": result["violated_fields"],
                "violated_methods": result["violated_methods"],
            })
        else:
            survivors.append(recipe)  # same object, order preserved
    return survivors, rejected
