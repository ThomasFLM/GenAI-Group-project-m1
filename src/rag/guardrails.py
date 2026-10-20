"""Garde-fous : détection d'injection de prompt (entrée), validation de la sortie.

Défense en profondeur :
1. entrée   : heuristiques FR/EN sur la question (politique `block` ou `warn`) ;
2. contexte : extraits balisés + consignes « données ≠ instructions » (prompts.py) ;
3. sortie   : JSON validé par pydantic, citations limitées aux documents récupérés,
              détection de fuite du prompt système (canari).
"""
from __future__ import annotations

import re

from pydantic import ValidationError

from .llm import parse_json_object
from .prompts import CANARY
from .schemas import LLMAnswer
from .text import normalize

INJECTION_PATTERNS = [
    r"ignor\w*\s+(?:\w+\s+){0,4}(instructions?|consignes?|regles?|directives?|ce qui precede)",
    r"oubli\w*\s+(?:\w+\s+){0,4}(instructions?|consignes?|regles?|directives?)",
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above)",
    r"disregard\s+(all\s+|the\s+)?(previous|prior|above|instructions)",
    r"(prompt|message|instructions?|consignes?)\s+(systeme|system|initia\w*|secret\w*|cache\w*)",
    r"system\s*prompt",
    r"(hidden|secret|internal|system|initial)\s+(instructions?|prompt|rules)",
    r"(print|reveal|show|repeat|output|leak)\s+(me\s+)?(your|the)\s+(instructions?|prompt|rules)",
    r"(revele|affiche|montre|donne|repete|recopie|traduis)\w*\s+(moi\s+)?(tes|vos|les|ton|votre)\s+"
    r"(instructions?|consignes?|regles?|prompt)",
    r"(tu es|vous etes)\s+(desormais|maintenant)",
    r"you are now|act as|jailbreak|\bdan\b|developer mode|mode (developpeur|admin\w*|debug)",
    r"^\s*(system|systeme)\s*:",
    r"\[\s*(system|inst)\s*\]|<\s*/?\s*(system|instructions?)\s*>",
    r"sans (citer|tenir compte|respecter)",
    r"(nouvelle|new)\s+(instruction|regle|rule)s?",
]
_COMPILED = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in INJECTION_PATTERNS]


def detect_injection(text: str) -> list[str]:
    """Renvoie la liste des motifs d'injection détectés (vide si rien)."""
    norm = normalize(text)
    return [p.pattern for p in _COMPILED if p.search(norm)]


def leaks_system_prompt(text: str) -> bool:
    norm = normalize(text)
    return CANARY.lower() in text.lower() or "regles imperatives" in norm or "code interne confidentiel" in norm


DOC_ID_RE = re.compile(r"\b[FNR]\d{2,6}\b")


def validate_answer(raw_text: str, allowed_doc_ids: list[str]) -> tuple[LLMAnswer | None, str]:
    """Valide la sortie structurée. Renvoie (réponse, erreur). Les citations hors contexte sont retirées."""
    try:
        data = parse_json_object(raw_text)
        ans = LLMAnswer.model_validate(data)
    except (ValueError, ValidationError) as e:
        return None, f"sortie JSON invalide : {e}"
    allowed = set(allowed_doc_ids)
    # normalise "F1234#2" ou "[F1234]" -> "F1234"
    cleaned: list[str] = []
    for c in ans.citations:
        m = DOC_ID_RE.search(c)
        if m and m.group(0) not in cleaned:
            cleaned.append(m.group(0))
    invalid = [c for c in cleaned if c not in allowed]
    ans.citations = [c for c in cleaned if c in allowed]
    if ans.answerable and not ans.citations:
        if invalid:
            return None, f"citations hors des extraits fournis : {invalid}"
        # on tente de récupérer des identifiants mentionnés dans le texte de la réponse
        ans.citations = [m for m in dict.fromkeys(DOC_ID_RE.findall(ans.answer)) if m in allowed]
        if not ans.citations:
            return None, "réponse sans citation alors que answerable=true"
    if not ans.answer.strip():
        return None, "réponse vide"
    return ans, ""
