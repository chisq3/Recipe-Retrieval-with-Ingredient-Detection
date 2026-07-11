#!/usr/bin/env python3
"""Ollama LLM helpers for request extraction and grounded answer text."""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from rag.pipeline.constraint_checker import (
    check_constraints,
    expanded_excluded_ingredients,
    normalize_constraints,
)
from rag.pipeline.ingredient_normalization import (
    availability_covers,
    equivalent_ingredient,
    normalize_ingredient_term,
    normalize_recipe_terms,
    parse_ingredient_terms,
)


from rag.paths import RULES_DIR

INGREDIENT_RULES_PATH = RULES_DIR / "runtime_ingredient_reasoning_rules.json"
# Preserve Vietnamese ingredient chips instead of corrupting them with the
# English-only ingredient normalizer. Query text is still handled by LLM extraction.
VIETNAMESE_CHAR_RE = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]", re.IGNORECASE)


def load_json_rules(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


INGREDIENT_RULES = load_json_rules(INGREDIENT_RULES_PATH)
STRUCTURAL_INGREDIENT_TERMS = set(INGREDIENT_RULES.get("structural_ingredient_terms", []))
MINOR_MISSING_LIMIT = 2


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def call_ollama(
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
    temperature: float,
    num_predict: int | None = None,
    output_schema: dict[str, Any] | None = None,
    seed: int | None = None,
    think: bool | None = None,
) -> str:
    options: dict[str, Any] = {
        "temperature": temperature,
    }
    if num_predict is not None and num_predict > 0:
        options["num_predict"] = num_predict
    if seed is not None:
        options["seed"] = seed
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": options,
    }
    # Constrained decoding: force the model to emit JSON matching this schema
    # (Ollama -> llama.cpp GBNF grammar). Guarantees structure/enums at decode
    # time instead of repairing the output afterwards.
    if output_schema is not None:
        payload["format"] = output_schema
    if think is not None:
        payload["think"] = think
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        content_parts: list[str] = []
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                if time.perf_counter() - start > timeout:
                    break
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                data = json.loads(line)
                content_parts.append(str(data.get("message", {}).get("content", "")))
                if data.get("done"):
                    break
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach local LLM endpoint. Start Ollama/server first, for example:\n"
            "  ollama serve\n"
            f"Endpoint: {endpoint}"
        ) from exc
    return "".join(content_parts).strip()


def extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


def split_terms(text: str) -> list[str]:
    terms: list[str] = []
    for part in re.split(r"[,;|/]+", text):
        cleaned = re.sub(r"\s+", " ", part.strip().lower())
        if cleaned:
            terms.append(cleaned)
    return list(dict.fromkeys(terms))


def normalize_user_ingredient_term(term: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(term or "").strip().lower())
    if not cleaned:
        return ""
    if VIETNAMESE_CHAR_RE.search(cleaned):
        return cleaned
    return normalize_ingredient_term(cleaned) or cleaned


def contains_term(text: str, term: str) -> bool:
    if not term:
        return False
    normalized_text = f" {text.lower()} "
    normalized_term = term.lower().strip()
    return f" {normalized_term} " in normalized_text or normalized_term in normalized_text


def extraction_output_schema() -> dict[str, Any]:
    """JSON schema for constrained decoding of the extraction call.

    The schema enforces structure and required fields, but intentionally leaves
    values open. Value enums caused a small model to coerce uncertain inputs into
    incorrect labels. Semantic values are checked after decoding instead.
    """
    nullable_string = {"type": ["string", "null"]}
    string_list = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "dish_name": nullable_string,
            "available_ingredients": string_list,
            "must_use_ingredients": string_list,
            "intent": {
                "type": "object",
                "properties": {
                    "meal_type": nullable_string,
                    "dish_type": nullable_string,
                    "main_ingredient_focus": string_list,
                    "difficulty": nullable_string,
                    "goal": string_list,
                },
                "required": ["meal_type", "dish_type", "main_ingredient_focus", "difficulty", "goal"],
            },
            "constraints": {
                "type": "object",
                "properties": {
                    "diet": nullable_string,
                    "cuisine": nullable_string,
                    "method_include": string_list,
                    "method_exclude": string_list,
                    "ingredient_exclude": string_list,
                    "cost": nullable_string,
                    "max_time": {"type": ["integer", "null"]},
                },
                "required": [
                    "diet",
                    "cuisine",
                    "method_include",
                    "method_exclude",
                    "ingredient_exclude",
                    "cost",
                    "max_time",
                ],
            },
        },
        "required": [
            "dish_name",
            "available_ingredients",
            "must_use_ingredients",
            "intent",
            "constraints",
        ],
    }


def build_extraction_messages(query: str, user_ingredients: str) -> list[dict[str, str]]:
    schema = {
        "dish_name": "",
        "available_ingredients": [],
        "must_use_ingredients": [],
        "intent": {
            "meal_type": None,
            "dish_type": None,
            "main_ingredient_focus": [],
            "difficulty": None,
            "goal": [],
        },
        "constraints": {
            "diet": None,
            "cuisine": None,
            "method_include": [],
            "method_exclude": [],
            "ingredient_exclude": [],
            "cost": None,
            "max_time": None,
        },
    }
    examples = [
        {
            "query": "Tôi muốn món tráng miệng dễ làm, ít nguyên liệu, không cần lò. Tôi có sữa, táo và rau cải.",
            "ingredients": "milk, apple, cabbage",
            "output": {
                "dish_name": None,
                "available_ingredients": ["milk", "apple", "cabbage"],
                "must_use_ingredients": [],
                "intent": {
                    "meal_type": "dessert",
                    "dish_type": "dessert",
                    "main_ingredient_focus": [],
                    "difficulty": "easy",
                    "goal": ["few_ingredients"],
                },
                "constraints": {
                    "diet": None,
                    "cuisine": None,
                    "method_include": [],
                    "method_exclude": ["oven"],
                    "ingredient_exclude": [],
                    "cost": None,
                    "max_time": None,
                },
            },
        },
        {
            "query": "Quick vegan breakfast under 30 minutes.",
            "ingredients": "oats, banana",
            "output": {
                "dish_name": None,
                "available_ingredients": ["oats", "banana"],
                "must_use_ingredients": [],
                "intent": {
                    "meal_type": "breakfast",
                    "dish_type": None,
                    "main_ingredient_focus": [],
                    "difficulty": None,
                    "goal": [],
                },
                "constraints": {
                    "diet": "vegan",
                    "cuisine": None,
                    "method_include": [],
                    "method_exclude": [],
                    "ingredient_exclude": [],
                    "cost": None,
                    "max_time": 30,
                },
            },
        },
        {
            "query": "Tôi muốn nấu canh nhưng không dùng nước mắm.",
            "ingredients": "fish, tomato, pineapple",
            "output": {
                "dish_name": None,
                "available_ingredients": ["fish", "tomato", "pineapple"],
                "must_use_ingredients": [],
                "intent": {
                    "meal_type": None,
                    "dish_type": "soup",
                    "main_ingredient_focus": [],
                    "difficulty": None,
                    "goal": [],
                },
                "constraints": {
                    "diet": None,
                    "cuisine": None,
                    "method_include": [],
                    "method_exclude": [],
                    "ingredient_exclude": ["fish sauce"],
                    "cost": None,
                    "max_time": None,
                },
            },
        },
        {
            "query": "I want a vegetarian soup for dinner.",
            "ingredients": "tomato, beans",
            "output": {
                "dish_name": None,
                "available_ingredients": ["tomato", "beans"],
                "must_use_ingredients": [],
                "intent": {
                    "meal_type": "dinner",
                    "dish_type": "soup",
                    "main_ingredient_focus": [],
                    "difficulty": None,
                    "goal": [],
                },
                "constraints": {
                    "diet": "vegetarian",
                    "cuisine": None,
                    "method_include": [],
                    "method_exclude": [],
                    "ingredient_exclude": [],
                    "cost": None,
                    "max_time": None,
                },
            },
        },
        {
            "query": "Cheap chicken dinner for a weeknight.",
            "ingredients": "chicken, rice",
            "output": {
                "dish_name": None,
                "available_ingredients": ["chicken", "rice"],
                "must_use_ingredients": [],
                "intent": {
                    "meal_type": "dinner",
                    "dish_type": None,
                    "main_ingredient_focus": ["chicken"],
                    "difficulty": None,
                    "goal": ["weeknight"],
                },
                "constraints": {
                    "diet": None,
                    "cuisine": None,
                    "method_include": [],
                    "method_exclude": [],
                    "ingredient_exclude": [],
                    "cost": "inexpensive",
                    "max_time": None,
                },
            },
        },
        {
            "query": "Tôi muốn món khoai tây làm bằng lò vi sóng.",
            "ingredients": "potato, cheese",
            "output": {
                "dish_name": None,
                "available_ingredients": ["potato", "cheese"],
                "must_use_ingredients": [],
                "intent": {
                    "meal_type": None,
                    "dish_type": "potato dish",
                    "main_ingredient_focus": ["potato"],
                    "difficulty": None,
                    "goal": [],
                },
                "constraints": {
                    "diet": None,
                    "cuisine": None,
                    "method_include": ["microwave"],
                    "method_exclude": [],
                    "ingredient_exclude": [],
                    "cost": None,
                    "max_time": None,
                },
            },
        },
        {
            "query": "I want an easy dinner with mushrooms.",
            "ingredients": "mushroom, chicken, rice",
            "output": {
                "dish_name": None,
                "available_ingredients": ["mushroom", "chicken", "rice"],
                "must_use_ingredients": ["mushroom"],
                "intent": {
                    "meal_type": "dinner",
                    "dish_type": None,
                    "main_ingredient_focus": ["mushroom"],
                    "difficulty": "easy",
                    "goal": [],
                },
                "constraints": {
                    "diet": None,
                    "cuisine": None,
                    "method_include": [],
                    "method_exclude": [],
                    "ingredient_exclude": [],
                    "cost": None,
                    "max_time": None,
                },
            },
        },
        {
            "query": "I want Vietnamese food with chicken.",
            "ingredients": "chicken, rice",
            "output": {
                "dish_name": None,
                "available_ingredients": ["chicken", "rice"],
                "must_use_ingredients": ["chicken"],
                "intent": {
                    "meal_type": None,
                    "dish_type": None,
                    "main_ingredient_focus": ["chicken"],
                    "difficulty": None,
                    "goal": [],
                },
                "constraints": {
                    "diet": None,
                    "cuisine": "Vietnamese",
                    "method_include": [],
                    "method_exclude": [],
                    "ingredient_exclude": [],
                    "cost": None,
                    "max_time": None,
                },
            },
        },
    ]
    examples_text = "\n\n".join(
        (
            f"Example {index}:\n"
            f"Input query: {example['query']}\n"
            f"Input ingredients: {example['ingredients']}\n"
            "Output JSON:\n"
            f"{json.dumps(example['output'], ensure_ascii=False, indent=2)}"
        )
        for index, example in enumerate(examples, start=1)
    )
    return [
        {
            "role": "system",
            "content": (
                "You extract structured recipe-search requests for a recipe retrieval system. "
                "Return only valid JSON. Do not explain. Use English canonical terms. "
                "Translate or canonicalize user-provided ingredient names from any language into common English food terms "
                "in available_ingredients; if unsure, keep the original ingredient phrase. Then extract only information "
                "that is directly stated or strongly entailed by the user query. "
                "If the user did not specify a value, use null or an empty list. "
                "Do not invent a dish name, ingredient, diet, cuisine, cost, or method."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
                f"Few-shot examples:\n\n{examples_text}\n\n"
                f"User query: {query}\n"
                f"User provided ingredients: {user_ingredients}\n\n"
                "Rules:\n"
                "1. dish_name must be null unless the user names a specific dish.\n"
                "2. meal_type is the eating context such as breakfast, dinner, dessert, snack, or beverage. dish_type is the concrete dish/category such as soup, potato dish, cake, pasta, salad, stew, or drink.\n"
                "3. Put user provided ingredients in available_ingredients by default, using common English canonical ingredient names when the meaning is clear; they are not mandatory to all appear in the recipe.\n"
                "4. Do not put cooking equipment or method terms into constraints.ingredient_exclude.\n"
                "5. Add a cooking method to method_exclude only when the user explicitly rejects that method.\n"
                "6. Canonicalize cooking methods into English schema values.\n"
                "7. Add an ingredient to ingredient_exclude only when the user explicitly excludes it or states an allergy.\n"
                "8. Soft preferences such as meal type, difficulty, health goals, budget, or few ingredients should influence intent or the matching constraint field, but must not become ingredient exclusions.\n"
                "9. Hard constraints are only: required diet, included/excluded cooking method, excluded ingredient/allergen, max_time, and cost/budget. Cuisine is a matching constraint, not an ingredient or dish type.\n"
                "10. If the query states a dietary requirement, fill constraints.diet. Do not leave it only in intent.\n"
                "11. If the query states affordability or budget, fill constraints.cost with an economic value such as inexpensive. Do not use non-cost words as cost.\n"
                "12. If the query states a cooking method to use, fill method_include. If it rejects a method, fill method_exclude. Never put the same method in both.\n"
                "12a. Translate cooking equipment or method constraints from any language into English schema values. Phrases meaning not needed, not using, without, avoiding, or no + a cooking appliance/method are rejections and must go in method_exclude, not goal or difficulty.\n"
                "12b. Distinguish oven from microwave: oven, baking oven, bake/no-bake, and Vietnamese lo/nuong constraints map to oven; microwave and Vietnamese lo vi song map to microwave.\n"
                "13. Put explicitly excluded ingredients or allergens into constraints.ingredient_exclude.\n"
                "14. Never infer hard exclusions from soft preferences.\n"
                "15. Put an ingredient in must_use_ingredients only when the user explicitly asks to use it or names a specific dish that requires it.\n"
                "16. Never put user-provided ingredients into constraints.ingredient_exclude unless the query explicitly excludes them.\n"
                "17. Preserve dish intent even when meal context is also present: 'vegetarian soup for dinner' means meal_type='dinner' and dish_type='soup'.\n"
                "18. main_ingredient_focus should include ingredients emphasized in the query itself, such as potato in 'microwave potato dish', not every user-provided ingredient.\n"
                "19. Put an ingredient in must_use_ingredients and main_ingredient_focus only when it is tied to the desired DISH: English 'a dish with/using/made with/made from X', Vietnamese 'mon co/voi/dung/lam tu X'. Ingredients the user only says they possess or have on hand — English 'I have X, Y', Vietnamese 'toi co/minh co X, Y' — go into available_ingredients only and must stay out of must_use_ingredients.\n"
                "20. Do not treat every available ingredient as must_use; only promote ingredients emphasized in the query text itself or when the query says to use all ingredients. If the emphasized query ingredient is written in another language but matches one available ingredient after canonicalization, use the same English canonical ingredient term in must_use_ingredients.\n"
                "21. Before returning, internally verify that every explicit diet, cuisine, cost, method, allergy, and excluded ingredient in the user query appears in the correct constraints field.\n"
                "22. Before returning, internally verify that constraints fields do not contain unrelated words copied from meal context, goals, or ingredients.\n"
                "23. Excluding one meat or ingredient does not imply a vegetarian or vegan diet.\n"
                "24. Set constraints.diet only to vegetarian or vegan when the user explicitly asks for that diet, plant-based food, no meat, or no animal products. Do not use dairy-free, gluten-free, nut-free, or similar ingredient exclusions as diet values; put the excluded group in ingredient_exclude instead.\n"
                "24a. For phrases like no cheese, without cheese, no dairy, dairy-free, or not using dairy, put cheese/dairy in ingredient_exclude and do not infer vegan by itself.\n"
                "25. If the query states a country, region, regional adjective, nationality, or cuisine style, fill constraints.cuisine with the English canonical cuisine name. In short noun phrases, treat a country/region/nationality modifier before a dish/category/food/cooking/meal as cuisine. Vietnamese phrases such as món Việt, món Việt Nam, đồ ăn Việt, or đồ ăn Việt Nam mean constraints.cuisine='vietnamese'. Do not drop multi-word cuisine or region phrases; plain English or lowercase snake_case is acceptable. Do not put cuisine in meal_type, dish_type, main_ingredient_focus, or goal.\n"
                "26. If the query does not explicitly mention a country, region, or cuisine style, constraints.cuisine must be null. Never infer cuisine from the user's language or from examples.\n"
                "27. dish_type must be the dish/category only. Do not include cuisine, diet, cost, time, or difficulty words inside dish_type.\n"
                "28. goal should contain only preference labels such as few_ingredients, healthy, spicy, weeknight, or kid_friendly. Do not put meal type, cuisine, dish type, or full query phrases inside goal.\n"
                "29. Difficulty describes cooking complexity and may contain values such as easy, medium, hard, or beginner. Few ingredients is a goal, not a difficulty, and must be written as few_ingredients. Do not place quick in difficulty or goal. Only fill constraints.max_time when the user provides a specific time limit; otherwise, preserve quick only in the original query.\n"
                "30. If the user names a specific dish, including Vietnamese dish names such as pho, banh mi, bun bo hue, or thit kho, put that name in dish_name using plain lowercase ASCII when possible.\n"
                "Return exactly one JSON object matching the schema."
            ),
        },
    ]


def build_ingredient_comparison(
    user_ingredients: str,
    pantry_ingredients: str,
    recipe: dict[str, Any],
) -> dict[str, Any]:
    available_core_terms = [
        normalized
        for term in split_terms(user_ingredients)
        if (normalized := normalize_user_ingredient_term(term))
    ]
    assumed_pantry_terms = [
        normalized
        for term in split_terms(pantry_ingredients)
        if (normalized := normalize_user_ingredient_term(term))
    ]
    available_for_comparison = list(dict.fromkeys([*available_core_terms, *assumed_pantry_terms]))
    recipe_text = str(recipe.get("canonical_ingredients_text", "") or "").lower()
    recipe_terms = parse_ingredient_terms(recipe.get("normalized_ingredient_terms", "")) or normalize_recipe_terms(recipe_text)

    matched_core = [
        term
        for term in available_core_terms
        if any(availability_covers(term, recipe_term) for recipe_term in recipe_terms)
    ]
    matched_pantry = [
        term
        for term in assumed_pantry_terms
        if any(availability_covers(term, recipe_term) for recipe_term in recipe_terms)
    ]
    missing_terms = [
        term
        for term in recipe_terms
        if not any(
            availability_covers(available, term)
            for available in available_for_comparison
        )
    ]
    return {
        "available_core_ingredients": available_core_terms,
        "assumed_available_pantry": assumed_pantry_terms,
        "available_for_comparison": available_for_comparison,
        "recipe_ingredients": list(dict.fromkeys(recipe_terms)),
        "matched_core_ingredients": matched_core,
        "matched_pantry_ingredients": matched_pantry,
        "initial_missing_core_terms": list(dict.fromkeys(missing_terms)),
        "initial_missing_pantry_terms": [],
        "note": (
            "This comparison is lexical and may be imperfect. The model may group terms into ingredient phrases, "
            "but must not invent ingredients outside recipe context. Default pantry ingredients are assumed available "
            "and should not be reported as missing unless the user explicitly excludes them."
        ),
    }


def compute_feasibility(
    query: str,
    recipe: dict[str, Any],
    ingredient_comparison: dict[str, Any],
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    missing_core = list(ingredient_comparison.get("initial_missing_core_terms", []))
    recipe_ingredients = list(ingredient_comparison.get("recipe_ingredients", []))
    matched_core = list(ingredient_comparison.get("matched_core_ingredients", []))
    available_core = list(ingredient_comparison.get("available_core_ingredients", []))
    blocking_reasons: list[str] = []
    warnings: list[str] = []
    critical_missing: list[str] = []
    constraint_result = check_constraints(recipe, constraints, query=query)

    if not available_core:
        # No inventory => inventory feasibility is not evaluated. BUT a hard
        # constraint violation (excluded diet/method/ingredient) must still block:
        # previously this branch returned cookable_as_is unconditionally (bug #5),
        # mislabelling e.g. an oven recipe as cookable for a "no oven" request.
        hard_violations = (
            list(constraint_result["hard_constraint_violations"])
            if constraint_result["has_hard_violation"]
            else []
        )
        return {
            "computed_feasibility": "not_recommended" if hard_violations else "cookable_as_is",
            "blocking_reasons": hard_violations,
            "warnings": ["No ingredients were provided, so cookability from your kitchen was not evaluated."],
            "computed_missing_core_ingredients": [],
            "computed_missing_pantry_ingredients": [],
            "computed_critical_missing_ingredients": [],
            "constraint_check": constraint_result,
        }

    if constraint_result["has_hard_violation"]:
        blocking_reasons.extend(constraint_result["hard_constraint_violations"])

    structural_missing = [term for term in missing_core if term in STRUCTURAL_INGREDIENT_TERMS]
    if structural_missing:
        critical_missing.extend(structural_missing)
        blocking_reasons.append("This recipe is missing ingredients that are central to the dish.")

    matched_count = len(matched_core)
    missing_count = len(missing_core)
    too_many_missing_for_inventory = (
        missing_count > MINOR_MISSING_LIMIT
        and (matched_count == 0 or missing_count > max(MINOR_MISSING_LIMIT + 1, matched_count + 2))
    )
    if too_many_missing_for_inventory:
        critical_missing.extend(missing_core)
        blocking_reasons.append("This recipe needs too many additional ingredients.")

    if recipe_ingredients and not matched_core:
        critical_missing.extend(missing_core)
        blocking_reasons.append("This recipe does not use the ingredients you provided.")

    critical_missing = list(dict.fromkeys(critical_missing))
    if blocking_reasons:
        feasibility = "not_recommended"
    elif missing_core:
        feasibility = "cookable_with_minor_adjustment"
    else:
        feasibility = "cookable_as_is"

    if missing_core:
        warnings.append("Some recipe ingredients are not in your available ingredients or basic pantry.")
        if missing_count > MINOR_MISSING_LIMIT and not too_many_missing_for_inventory:
            warnings.append("This recipe needs several additional ingredients, but it still uses ingredients you provided.")

    return {
        "computed_feasibility": feasibility,
        "blocking_reasons": blocking_reasons,
        "warnings": warnings,
        "computed_missing_core_ingredients": missing_core,
        "computed_missing_pantry_ingredients": list(ingredient_comparison.get("initial_missing_pantry_terms", [])),
        "computed_critical_missing_ingredients": critical_missing,
        "constraint_check": constraint_result,
    }


def build_answer_plan(
    query: str,
    user_ingredients: str,
    pantry_ingredients: str,
    recipe: dict[str, Any],
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    comparison = build_ingredient_comparison(user_ingredients, pantry_ingredients, recipe)
    computed = compute_feasibility(query, recipe, comparison, constraints)
    missing_items = [
        *computed["computed_missing_core_ingredients"],
        *computed["computed_missing_pantry_ingredients"],
    ]
    blocking_reasons = list(computed.get("blocking_reasons", []))
    warnings = list(computed.get("warnings", []))
    feasibility = computed["computed_feasibility"]
    inventory_evaluated = bool(comparison.get("available_core_ingredients"))

    warning_facts: list[str] = []
    fallback_why = ""
    fallback_warning = ""

    if feasibility == "cookable_as_is":
        if inventory_evaluated:
            matched = comparison.get("matched_core_ingredients", [])
            fallback_why = fallback_as_is_reason(matched)
        else:
            fallback_why = fallback_recommendation_reason_without_inventory()
            fallback_warning = fallback_no_inventory_warning()
            warning_facts.append(fallback_warning)
    elif feasibility == "cookable_with_minor_adjustment" and not blocking_reasons:
        fallback_why = fallback_minor_adjustment_reason(missing_items)
        warning_facts.extend(warnings)
        fallback_warning = " ".join(warning_facts)
    else:
        fallback_why = fallback_not_recommended_reason(blocking_reasons, missing_items)
        warning_facts.extend([*blocking_reasons, *warnings])
        fallback_warning = " ".join(warning_facts)

    return {
        "language": "English",
        "recipe_title": recipe.get("title", ""),
        "feasibility": feasibility,
        "inventory_evaluated": inventory_evaluated,
        "matched_core_ingredients": comparison.get("matched_core_ingredients", []),
        "missing_ingredients": missing_items,
        "blocking_reasons": blocking_reasons,
        "warning_facts": warning_facts,
        "fallback_why_recommended": fallback_why,
        "fallback_warning": fallback_warning,
    }


# Few-shot examples for grounded answer generation: RAW recipe situation -> why.
# Lean prompt + these examples + answer_output_schema() scored 19/19 faithful + genuinely
# generated on the English answer-G diagnostic (eval/answer_g_eval_summary.json). The
# not-recommended example keeps many-missing cases from being over-claimed.
_ANSWER_FEWSHOT = [
    {"situation": {"feasibility": "cookable_as_is", "inventory_evaluated": True,
                   "you_have": ["chicken", "rice"], "to_buy": [], "warnings": []},
     "output": {"why_recommended": "Both chicken and rice are already in your kitchen, so you can cook this right away."}},
    {"situation": {"feasibility": "cookable_with_minor_adjustment", "inventory_evaluated": True,
                   "you_have": ["pasta"], "to_buy": ["basil", "parmesan"],
                   "warnings": ["Some recipe ingredients are not in your available ingredients."]},
     "output": {"why_recommended": "This fits your pasta; you would just need to add basil and parmesan."}},
    {"situation": {"feasibility": "cookable_as_is", "inventory_evaluated": False,
                   "you_have": [], "to_buy": [],
                   "warnings": ["No available ingredients were provided, so cookability was not evaluated."]},
     "output": {"why_recommended": "This matches the dish you asked for. You did not list any ingredients, so I have not checked what you would need to buy."}},
    {"situation": {"feasibility": "not_recommended", "inventory_evaluated": True,
                   "you_have": ["egg"], "to_buy": ["flour", "sugar", "butter", "milk", "cocoa"],
                   "warnings": ["This recipe needs too many additional ingredients."]},
     "output": {"why_recommended": "You have egg, but this recipe needs many more ingredients (flour, sugar, butter, and others), so it is not a practical fit right now."}},
]


def answer_output_schema() -> dict[str, Any]:
    """Constrained-decoding schema for the answer call.

    The LLM only drafts a short explanation. Warnings and factual fields are
    deterministic outputs from repair_answer_output().
    """
    return {
        "type": "object",
        "properties": {"why_recommended": {"type": "string"}},
        "required": ["why_recommended"],
    }


def build_answer_messages(
    query: str,
    user_ingredients: str,
    pantry_ingredients: str,
    recipe: dict[str, Any],
    constraints: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Lean + few-shot grounded answer prompt. Gives the LLM the RAW recipe situation (NOT a
    pre-written sentence) so it genuinely phrases a grounded why_recommended. Pair with
    answer_output_schema() at the call site. The deterministic validator + repair still
    guard: the LLM why is kept only if it passes validation; warning and every other field
    stay deterministic."""
    plan = build_answer_plan(query, user_ingredients, pantry_ingredients, recipe, constraints)
    situation = {
        "language": plan["language"],
        "feasibility": plan["feasibility"],
        "inventory_evaluated": plan["inventory_evaluated"],
        "you_have": plan.get("matched_core_ingredients", []),
        "to_buy": plan.get("missing_ingredients", []),
        "warnings": plan.get("warning_facts", []),
    }
    shots = "\n\n".join(
        f"Recipe situation:\n{json.dumps(s['situation'], ensure_ascii=False)}\nOutput:\n{json.dumps(s['output'], ensure_ascii=False)}"
        for s in _ANSWER_FEWSHOT
    )
    return [
        {"role": "system", "content": (
            "You write a SHORT, natural, user-facing explanation (why_recommended) for a recipe "
            "recommendation. Use ONLY the given recipe situation. Do NOT invent ingredients, steps, substitutions, "
            "quantities, or methods, and do NOT overstate how well the recipe fits. why_recommended = one or two "
            "natural sentences. "
            "Speak directly to the user about the recipe and their ingredients - do NOT mention 'facts', 'the system', "
            "'provided information', or that you were given data. Write in the given language. "
            "Return only a JSON object with key why_recommended."
        )},
        {"role": "user", "content": (
            f"Examples:\n\n{shots}\n\n"
            f"Now do this one.\nUser query: {query}\n"
            f"Recipe situation:\n{json.dumps(situation, ensure_ascii=False)}\nOutput:"
        )},
    ]


def validate_answer_output(
    output: dict[str, Any] | None,
    query: str,
    user_ingredients: str,
    pantry_ingredients: str,
    recipe: dict[str, Any],
    constraints: dict[str, Any] | None = None,
) -> list[str]:
    if output is None:
        return ["Output is not valid JSON."]

    issues: list[str] = []
    comparison = build_ingredient_comparison(user_ingredients, pantry_ingredients, recipe)
    computed = compute_feasibility(query, recipe, comparison, constraints)
    available_core = set(comparison["available_core_ingredients"])
    expected_missing_core = set(computed["computed_missing_core_ingredients"])
    expected_missing_pantry = set(computed["computed_missing_pantry_ingredients"])
    expected_critical = set(computed["computed_critical_missing_ingredients"])
    excluded_ingredients = expanded_excluded_ingredients(
        normalize_constraints(constraints or {}, query=query).get("ingredient_exclude", [])
    )

    if "feasibility" in output and output.get("feasibility") != computed["computed_feasibility"]:
        issues.append(
            f"feasibility mismatch: expected {computed['computed_feasibility']}, got {output.get('feasibility')}"
        )

    if "shopping_list" in output:
        shopping_list = set(str(item).lower() for item in output.get("shopping_list", []) if item)
        if shopping_list & available_core:
            issues.append(f"shopping_list includes already available core ingredients: {sorted(shopping_list & available_core)}")
        excluded_shopping = [
            item
            for item in shopping_list
            if any(contains_term(item, excluded) or contains_term(excluded, item) for excluded in excluded_ingredients)
        ]
        if excluded_shopping:
            issues.append(f"shopping_list includes excluded ingredients: {sorted(excluded_shopping)}")

    substitutions = set(str(item).lower() for item in output.get("substitutions", []) if item)
    if substitutions & available_core:
        issues.append(f"substitutions include already available core ingredients: {sorted(substitutions & available_core)}")
    if substitutions:
        issues.append("substitutions provided, but the system has no grounded substitution module.")

    if "missing_core_ingredients" in output:
        missing_core = set(str(item).lower() for item in output.get("missing_core_ingredients", []) if item)
        if missing_core != expected_missing_core:
            issues.append(f"missing_core_ingredients mismatch: expected {sorted(expected_missing_core)}, got {sorted(missing_core)}")

    if "missing_pantry_ingredients" in output:
        missing_pantry = set(str(item).lower() for item in output.get("missing_pantry_ingredients", []) if item)
        if missing_pantry != expected_missing_pantry:
            issues.append(
                f"missing_pantry_ingredients mismatch: expected {sorted(expected_missing_pantry)}, got {sorted(missing_pantry)}"
            )

    if "critical_missing_ingredients" in output:
        critical = set(str(item).lower() for item in output.get("critical_missing_ingredients", []) if item)
        if critical != expected_critical:
            issues.append(
                f"critical_missing_ingredients mismatch: expected {sorted(expected_critical)}, got {sorted(critical)}"
            )

    if output.get("adapted_steps") and (
        computed["computed_feasibility"] == "not_recommended" or computed["constraint_check"]["has_hard_violation"]
    ):
        issues.append("adapted_steps provided despite not_recommended feasibility or a hard constraint violation.")

    evidence = output.get("evidence_used", {})
    if "evidence_used" in output and isinstance(evidence, dict):
        recipe_terms = parse_ingredient_terms(recipe.get("normalized_ingredient_terms", "")) or normalize_recipe_terms(
            recipe.get("canonical_ingredients_text") or recipe.get("ingredients_text") or ""
        )
        evidence_terms = normalize_recipe_terms(str(evidence.get("recipe_ingredients_used") or ""))
        user_core_terms = comparison.get("available_core_ingredients", [])
        unsupported_evidence = [
            term
            for term in user_core_terms
            if any(equivalent_ingredient(term, evidence_term) for evidence_term in evidence_terms)
            if recipe_terms and not any(equivalent_ingredient(recipe_term, term) for recipe_term in recipe_terms)
        ]
        if unsupported_evidence:
            issues.append(f"evidence_used includes ingredients not found in recipe: {unsupported_evidence}")

    # Ground why_recommended: the short LLM reasoning must not claim an available
    # ingredient that the selected recipe does not contain (bug #2: "Apple Rabdi ...
    # cabbage could be added"). If it does, flag it; repair_answer_output then falls
    # back to the deterministic reason (it keeps the LLM why only when issues is empty).
    why_text = str(output.get("why_recommended") or "").lower()
    if why_text:
        why_recipe_terms = parse_ingredient_terms(recipe.get("normalized_ingredient_terms", "")) or normalize_recipe_terms(
            recipe.get("canonical_ingredients_text") or recipe.get("ingredients_text") or ""
        )
        ungrounded_why = [
            term
            for term in comparison.get("available_core_ingredients", [])
            if re.search(rf"\b{re.escape(term)}\b", why_text)
            and why_recipe_terms
            and not any(equivalent_ingredient(recipe_term, term) for recipe_term in why_recipe_terms)
        ]
        if ungrounded_why:
            issues.append(
                f"why_recommended mentions ingredients not found in the recipe: {sorted(set(ungrounded_why))}"
            )

    return issues


def fallback_as_is_reason(matched: list[str]) -> str:
    matched_text = ", ".join(matched) if matched else "some available ingredients"
    return f"This recipe fits because it can be cooked with the available ingredients and uses {matched_text}."


def fallback_recommendation_reason_without_inventory() -> str:
    return "This recipe matches the requested dish, but inventory feasibility was not evaluated because no available ingredients were provided."


def fallback_no_inventory_warning() -> str:
    return "No ingredients were provided, so I did not check what you can cook from your kitchen."


def fallback_minor_adjustment_reason(missing_items: list[str]) -> str:
    missing_text = ", ".join(missing_items)
    return f"This recipe partially fits the available ingredients, but you need to add: {missing_text}."


def fallback_not_recommended_reason(blocking_reasons: list[str], missing_items: list[str]) -> str:
    if blocking_reasons:
        return "This recipe is not recommended because it violates or misses an important requirement."
    if missing_items:
        return "This recipe is not recommended because it is missing too many important ingredients."
    return "This recipe is not recommended for the current request."


# Output-policy guard for LLM why text. The system has no grounded substitution
# module, so a why that suggests swapping/replacing ingredients is not kept.
# This validates generated output, not the raw user query.
_FORBIDDEN_SUBSTITUTION_RE = re.compile(
    r"\b(?:substitut\w*|swap\w*|replac\w*)\b|\binstead\s+of\b|\bor\s+use\b",
    re.IGNORECASE,
)


def _why_suggests_substitution(text: str) -> bool:
    return bool(_FORBIDDEN_SUBSTITUTION_RE.search(text or ""))


def repair_answer_output(
    output: dict[str, Any] | None,
    query: str,
    user_ingredients: str,
    pantry_ingredients: str,
    recipe: dict[str, Any],
    constraints: dict[str, Any] | None = None,
    validation_issues: list[str] | None = None,
) -> dict[str, Any] | None:
    if output is None:
        return None

    repaired = dict(output)
    comparison = build_ingredient_comparison(user_ingredients, pantry_ingredients, recipe)
    computed = compute_feasibility(query, recipe, comparison, constraints)
    available_core = set(comparison["available_core_ingredients"])
    missing_core = list(computed["computed_missing_core_ingredients"])
    missing_pantry = list(computed["computed_missing_pantry_ingredients"])
    critical_missing = list(computed["computed_critical_missing_ingredients"])
    excluded_ingredients = expanded_excluded_ingredients(
        normalize_constraints(constraints or {}, query=query).get("ingredient_exclude", [])
    )

    repaired["recipe_title"] = recipe.get("title", repaired.get("recipe_title", ""))
    repaired["feasibility"] = computed["computed_feasibility"]
    repaired["available_core_ingredients"] = comparison["available_core_ingredients"]
    repaired["assumed_available_pantry"] = comparison["assumed_available_pantry"]
    repaired["missing_core_ingredients"] = missing_core
    repaired["missing_pantry_ingredients"] = missing_pantry
    repaired["critical_missing_ingredients"] = critical_missing
    repaired["evidence_used"] = {
        "recipe_ingredients_used": recipe.get("canonical_ingredients_text", recipe.get("ingredients_text", "")),
        "instruction_source": "instruction_text",
    }
    repaired["adapted_steps"] = []

    shopping_items = [
        str(item).lower()
        for item in [*missing_core, *missing_pantry]
        if str(item).lower() not in available_core
        and not any(
            contains_term(str(item).lower(), excluded) or contains_term(excluded, str(item).lower())
            for excluded in excluded_ingredients
        )
    ]
    repaired["shopping_list"] = list(dict.fromkeys(shopping_items))

    # No grounded substitution module (no substitution field in the corpus), so the
    # system never asserts substitutions. Author-written substitutions still reach the
    # user verbatim via the displayed recipe instruction_text.
    repaired["substitutions"] = []

    answer_plan = build_answer_plan(query, user_ingredients, pantry_ingredients, recipe, constraints)
    feasibility = computed["computed_feasibility"]
    repaired["why_recommended"] = answer_plan["fallback_why_recommended"]
    repaired["warning"] = answer_plan["fallback_warning"]

    if feasibility == "not_recommended":
        repaired["adapted_steps"] = []
    elif computed["constraint_check"]["has_hard_violation"]:
        repaired["adapted_steps"] = []

    # Prefer the LLM's why_recommended only when it passed deterministic validation.
    # All factual fields, feasibility, and warnings remain deterministic; the LLM
    # only phrases the short explanation shown beside the verbatim recipe.
    llm_why = str(output.get("why_recommended") or "").strip()
    why_lang_ok = bool(llm_why)  # English-only: trust the English answer prompt (probe: VN input -> English why)
    # Keep the LLM why_recommended when it passed the validator (grounding/scope) AND inventory
    # was evaluated (no-inventory whys stay deterministic — the LLM
    # tends to falsely imply a pantry check) AND it does not suggest a substitution (no grounded
    # substitution module). This widens past the old cookable_as_is-only gate to all
    # inventory-evaluated feasibility classes; warning + every other field stay deterministic.
    if (not validation_issues and why_lang_ok and answer_plan["inventory_evaluated"]
            and not _why_suggests_substitution(llm_why)):
        repaired["why_recommended"] = llm_why

    return repaired

