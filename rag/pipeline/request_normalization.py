"""Normalize LLM-extracted user requests before retrieval.

The LLM gives us a schema-shaped request, but small local models still make
schema mistakes: plural ingredient forms, generic dish names, and negation
phrases in the wrong field. This module keeps those deterministic corrections
in one place so answer generation and evaluation share the same contract.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from rag.pipeline.constraint_checker import CONSTRAINT_RULES, normalize_constraints
from rag.pipeline.ingredient_normalization import normalize_ingredient_term
from rag.pipeline.llm_service import split_terms

# The only diet values our deterministic constraint checker understands. Anything
# else the LLM puts in constraints.diet (e.g. "healthy", "low-carb") is misrouted.
VALID_DIETS = set(CONSTRAINT_RULES.get("diet_conflict_terms", {}).keys())
# Matches a negation phrase the LLM may leave inside a goal/diet value, e.g.
# "no sugar", "without alcohol", "dairy-free". A structural grammar pattern that
# generalizes to any X; it operates on extracted text only, never the raw query.
EXTRACTED_NEGATION_RE = re.compile(r"^(?:no|without|avoid)\s+(.+)$|^(.+?)[-\s]free$")

GENERIC_DISH_SUFFIXES = (" recipes", " recipe", " dishes", " dish", " meals", " meal")
# Tokens that make a dish_name generic only when every other token is already
# explained by typed fields such as diet, cuisine, meal_type, or difficulty.
GENERIC_DISH_NAME_TOKENS = {"dish", "dishes", "meal", "meals", "recipe", "recipes", "food"}


def singularize_token(token: str) -> str:
    """Conservatively singularize request metadata tokens.

    Ingredient fields use ``normalize_ingredient_term``. This helper supports
    dish names, dish types, goals, and generic suffix stripping.
    """
    if len(token) <= 3 or token.endswith(("ss", "us", "is", "ous")):
        return token
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("ves") and len(token) > 4:
        return f"{token[:-3]}f"
    if token.endswith("oes") and len(token) > 4:
        return token[:-2]
    if token.endswith(("ches", "shes", "sses", "xes", "zes")):
        return token[:-2]
    if token.endswith("s"):
        return token[:-1]
    return token


def normalize_term(term: Any) -> str:
    cleaned = str(term).strip().lower().replace("_", " ")
    if not cleaned:
        return ""
    parts = cleaned.split()
    parts[-1] = singularize_token(parts[-1])
    return " ".join(parts)


def normalize_ingredient_field(term: Any) -> str:
    """Canonicalize one user/LLM ingredient before retrieval."""
    return normalize_ingredient_term(term) or normalize_term(term)


def normalize_ingredient_list(items: Any) -> list[str]:
    if not isinstance(items, list):
        return []
    return list(
        dict.fromkeys(
            normalized
            for item in items
            if str(item).strip()
            if (normalized := normalize_ingredient_field(item))
        )
    )


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    stripped = "".join(char for char in normalized if not unicodedata.combining(char))
    return stripped.replace("\u0111", "d").replace("\u0110", "D")


def normalized_text(value: Any) -> str:
    return strip_accents(str(value or "").lower().replace("_", " "))


def strip_generic_dish_suffix(dish_name: str) -> tuple[str, str]:
    normalized = normalize_term(dish_name)
    removed_suffix = ""
    changed = True
    while changed:
        changed = False
        for suffix in GENERIC_DISH_SUFFIXES:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)].strip()
                removed_suffix = suffix.strip()
                changed = True
                break
    return normalized, removed_suffix


def typed_generic_tokens(
    intent: dict[str, Any],
    constraints: dict[str, Any],
    normalized_dish: str,
) -> set[str]:
    """Tokens that can explain a generic dish_name such as 'vegan dinner'.

    If a multi-token cuisine value exactly equals the dish name (e.g. the LLM
    emits cuisine='pad thai'), do not let that explain away the whole dish.
    """
    tokens = set(GENERIC_DISH_NAME_TOKENS)
    for value in (
        constraints.get("diet"),
        constraints.get("cuisine"),
        intent.get("cuisine"),
        intent.get("meal_type"),
        intent.get("difficulty"),
    ):
        normalized_value = normalize_term(value or "")
        if normalized_value == normalized_dish and len(normalized_dish.split()) > 1:
            continue
        for tok in normalized_value.split():
            if tok:
                tokens.add(tok)
    return tokens


def dish_is_all_typed_generic(normalized_dish: str, intent: dict[str, Any], constraints: dict[str, Any]) -> bool:
    """Return True when dish_name is only a typed generic phrase."""
    tokens = normalized_dish.split()
    if not tokens:
        return False
    generic = typed_generic_tokens(intent, constraints, normalized_dish)
    return all(tok in generic for tok in tokens)


def negated_target(text: Any) -> str | None:
    """Return X from a negation phrase such as 'no X' / 'without X' / 'X-free'."""
    cleaned = normalize_term(text)
    if not cleaned:
        return None
    match = EXTRACTED_NEGATION_RE.match(cleaned)
    if not match:
        return None
    target = (match.group(1) or match.group(2) or "").strip()
    return target or None


def method_exclude_target(target: str) -> str | None:
    """Map a negated cooking method/appliance to the canonical method name."""
    normalized_target = normalize_term(target)
    if not normalized_target:
        return None
    for method, config in CONSTRAINT_RULES.get("method_exclude_detectors", {}).items():
        normalized_method = normalize_term(method)
        recipe_terms = [
            normalize_term(term)
            for term in config.get("recipe_terms", [])
            if str(term).strip()
        ]
        if normalized_target == normalized_method or normalized_target in recipe_terms:
            return normalized_method
    return None


def append_excluded_target(constraints: dict[str, Any], target: str) -> None:
    """Route a negated target to method_exclude or ingredient_exclude."""
    method = method_exclude_target(target)
    if method:
        method_exclude = constraints.get("method_exclude")
        method_exclude = method_exclude if isinstance(method_exclude, list) else []
        constraints["method_exclude"] = list(dict.fromkeys([*method_exclude, method]))
        return

    ingredient_exclude = constraints.get("ingredient_exclude")
    ingredient_exclude = ingredient_exclude if isinstance(ingredient_exclude, list) else []
    constraints["ingredient_exclude"] = list(
        dict.fromkeys([*ingredient_exclude, normalize_ingredient_field(target)])
    )


def validate_extracted_schema(intent: dict[str, Any], constraints: dict[str, Any]) -> None:
    """Deterministic schema validation of LLM-extracted fields (mutates in place)."""
    diet = normalize_term(constraints.get("diet") or "")
    if diet and diet not in VALID_DIETS:
        negated = negated_target(diet)
        if negated:
            append_excluded_target(constraints, negated)
        else:
            goals = intent.get("goal") if isinstance(intent.get("goal"), list) else []
            if diet not in [normalize_term(goal) for goal in goals]:
                goals.append(diet)
            intent["goal"] = goals
        constraints["diet"] = None

    goals = intent.get("goal") if isinstance(intent.get("goal"), list) else []
    kept_goals: list[str] = []
    for goal in goals:
        target = negated_target(goal)
        if target:
            append_excluded_target(constraints, target)
        else:
            kept_goals.append(goal)
    intent["goal"] = kept_goals
    ingredient_exclude = constraints.get("ingredient_exclude")
    ingredient_exclude = ingredient_exclude if isinstance(ingredient_exclude, list) else []
    constraints["ingredient_exclude"] = list(dict.fromkeys(ingredient_exclude))


def normalize_extracted_request(
    parsed: dict[str, Any] | None,
    query: str,
    ingredients: str,
) -> dict[str, Any]:
    """Normalize the LLM extraction payload into the retrieval contract."""
    input_ingredients = [
        normalized
        for item in split_terms(ingredients)
        if (normalized := normalize_ingredient_field(item))
    ]
    if not parsed:
        return {
            "dish_name": None,
            "available_ingredients": input_ingredients,
            "must_use_ingredients": [],
            "intent": {"text": query},
            "constraints": normalize_constraints({}, query=query),
        }

    constraints = parsed.get("constraints")
    if not isinstance(constraints, dict):
        constraints = {}
    intent = parsed.get("intent")
    if isinstance(intent, str):
        intent = {"text": intent}
    elif not isinstance(intent, dict):
        intent = {"text": query}

    if "meal_type" in constraints and not intent.get("meal_type"):
        intent["meal_type"] = constraints.get("meal_type")
    if constraints.get("difficulty") and not intent.get("difficulty"):
        intent["difficulty"] = constraints.get("difficulty")
    if "dish_type" not in intent:
        intent["dish_type"] = None
    if not isinstance(intent.get("main_ingredient_focus"), list):
        intent["main_ingredient_focus"] = []

    difficulty = normalize_term(intent.get("difficulty") or "")
    goals = intent.get("goal") if isinstance(intent.get("goal"), list) else []
    if difficulty in {"few", "few ingredient", "few ingredients"}:
        if "few_ingredients" not in goals:
            goals.append("few_ingredients")
        intent["difficulty"] = None
        intent["goal"] = goals
    elif difficulty:
        intent["difficulty"] = difficulty

    dish_name = parsed.get("dish_name")
    normalized_dish, stripped_suffix = strip_generic_dish_suffix(str(dish_name or ""))
    if stripped_suffix in {"dish", "dishes", "meal", "meals"} and len(normalized_dish.split()) == 1:
        normalized_dish = ""

    if normalized_dish and dish_is_all_typed_generic(normalized_dish, intent, constraints):
        normalized_dish = ""
    intent["dish_name"] = normalized_dish or None

    intent["dish_type"] = normalize_term(intent.get("dish_type") or "") or None
    intent["main_ingredient_focus"] = [
        normalize_ingredient_field(item)
        for item in intent.get("main_ingredient_focus", [])
        if str(item).strip()
    ]
    focus_terms = list(intent.get("main_ingredient_focus", []))

    available = parsed.get("available_ingredients")
    parsed_available_terms = normalize_ingredient_list(available)
    available_terms = parsed_available_terms or input_ingredients

    must_use = parsed.get("must_use_ingredients")
    if not isinstance(must_use, list):
        must_use = []
    must_use_terms = normalize_ingredient_list(must_use)
    intent["main_ingredient_focus"] = focus_terms

    validate_extracted_schema(intent, constraints)

    normalized_constraints = normalize_constraints(constraints, query=query)
    normalized_constraints["ingredient_exclude"] = normalize_ingredient_list(
        normalized_constraints.get("ingredient_exclude", [])
    )
    cuisine = normalize_term(normalized_constraints.get("cuisine") or "")
    dish_type = normalize_term(intent.get("dish_type") or "")
    if cuisine and dish_type:
        dish_without_cuisine = normalize_term(dish_type.replace(cuisine, ""))
        if not dish_without_cuisine or dish_without_cuisine in {"dish", "food", "meal", "cuisine", "recipe"}:
            intent["dish_type"] = None
    if cuisine and isinstance(intent.get("goal"), list):
        intent["goal"] = [
            goal
            for goal in intent["goal"]
            if normalize_term(goal) != cuisine and normalized_text(goal) != normalized_text(cuisine)
        ]

    return {
        "dish_name": normalized_dish or None,
        "available_ingredients": available_terms,
        "must_use_ingredients": must_use_terms,
        "intent": intent,
        "constraints": normalized_constraints,
    }


def intent_to_text(intent: Any) -> str:
    if isinstance(intent, str):
        return intent.strip()
    if not isinstance(intent, dict):
        return ""
    parts: list[str] = []
    for key in ["text", "meal_type", "dish_type", "difficulty", "cost"]:
        value = intent.get(key)
        if value and str(value).strip().lower() not in " ".join(parts).lower():
            parts.append(str(value))
    focus_terms = intent.get("main_ingredient_focus", [])
    if isinstance(focus_terms, list):
        for term in focus_terms:
            cleaned_term = str(term).replace("_", " ").strip()
            if cleaned_term and cleaned_term.lower() not in " ".join(parts).lower():
                parts.append(cleaned_term)
    goals = intent.get("goal", [])
    if isinstance(goals, list):
        for goal in goals:
            cleaned_goal = str(goal).replace("_", " ").strip()
            if cleaned_goal and cleaned_goal.lower() not in " ".join(parts).lower():
                parts.append(cleaned_goal)
    elif goals:
        cleaned_goals = str(goals).replace("_", " ").strip()
        if cleaned_goals and cleaned_goals.lower() not in " ".join(parts).lower():
            parts.append(cleaned_goals)
    return " ".join(dict.fromkeys(part.strip() for part in parts if part.strip()))


def build_retrieval_inputs(extracted: dict[str, Any], fallback_query: str, fallback_ingredients: str) -> dict[str, str]:
    available_ingredients = ", ".join(extracted.get("available_ingredients", [])) or fallback_ingredients
    must_use_ingredients = ", ".join(extracted.get("must_use_ingredients", []))
    constraints = extracted.get("constraints", {})
    constraint_terms = [
        str(value)
        for key, value in constraints.items()
        if key not in {"method_exclude", "ingredient_exclude"}
        and value is not None
        and not isinstance(value, list)
        and str(value).strip()
    ]
    method_includes = constraints.get("method_include") if isinstance(constraints.get("method_include"), list) else []
    constraint_terms.extend(str(item) for item in method_includes if str(item).strip())
    dish_name = str(extracted.get("dish_name") or "").strip()
    method_excludes = constraints.get("method_exclude") if isinstance(constraints.get("method_exclude"), list) else []
    intent_text = intent_to_text(extracted.get("intent"))
    intent = " ".join(part for part in [dish_name, intent_text, " ".join(constraint_terms)] if part)
    retrieval_query = " ".join(part for part in [dish_name, intent_text] if part)
    return {
        "query": fallback_query or retrieval_query,
        "ingredients": available_ingredients,
        "must_use_ingredients": must_use_ingredients,
        "excluded_metadata_terms": ", ".join(str(item) for item in method_excludes if str(item).strip()),
        "title_intent": dish_name,
        "intent": intent,
        "constraints": "",
    }
