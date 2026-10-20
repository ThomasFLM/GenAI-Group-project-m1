"""Guardrails: prompt-injection detection (input), output validation.

Defence in depth:
1. input   : FR/EN heuristics on the question (`block` or `warn` policy);
2. context : tagged excerpts + "data is not instructions" rules (prompts.py);
3. output  : JSON validated with pydantic, citations restricted to the retrieved documents,
             system-prompt leak detection (canary).

The injection patterns match French and English attacks. Validation error messages stay in
French because they are sent back to the LLM in the (French) retry prompt.
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
    """Return the list of detected injection patterns (empty if none)."""
    norm = normalize(text)
    return [p.pattern for p in _COMPILED if p.search(norm)]


def leaks_system_prompt(text: str) -> bool:
    norm = normalize(text)
    return CANARY.lower() in text.lower() or "regles imperatives" in norm or "code interne confidentiel" in norm


DOC_ID_RE = re.compile(r"\b[FNR]\d{2,6}\b")


def validate_answer(raw_text: str, allowed_doc_ids: list[str]) -> tuple[LLMAnswer | None, str]:
    """Validate the structured output. Returns (answer, error). Out-of-context citations are removed."""
    try:
        data = parse_json_object(raw_text)
        ans = LLMAnswer.model_validate(data)
    except (ValueError, ValidationError) as e:
        return None, f"sortie JSON invalide : {e}"
    allowed = set(allowed_doc_ids)
    # normalise "F1234#2" or "[F1234]" -> "F1234"
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
        # try to recover identifiers mentioned in the answer text
        ans.citations = [m for m in dict.fromkeys(DOC_ID_RE.findall(ans.answer)) if m in allowed]
        if not ans.citations:
            return None, "réponse sans citation alors que answerable=true"
    if not ans.answer.strip():
        return None, "réponse vide"
    return ans, ""
