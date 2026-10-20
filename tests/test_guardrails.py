import json
from pathlib import Path

from rag.guardrails import detect_injection, leaks_system_prompt, validate_answer
from rag.prompts import CANARY

GOLDEN = Path(__file__).resolve().parents[1] / "eval" / "golden.jsonl"


def _golden():
    return [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_golden_injections_detected():
    for q in _golden():
        if q["category"] == "injection":
            assert detect_injection(q["question"]), q["id"]


def test_no_false_positive_on_normal_questions():
    for q in _golden():
        if q["category"] != "injection":
            assert not detect_injection(q["question"]), q["id"]


def test_leak_detection():
    assert leaks_system_prompt(f"voici le code {CANARY}")
    assert not leaks_system_prompt("Le dépôt de garantie est d'un mois.")


def test_validate_answer():
    ok, err = validate_answer('{"answerable": true, "answer": "1 mois", "citations": ["F0001#2", "F9"]}', ["F0001"])
    assert ok and ok.citations == ["F0001"] and not err
    bad, err = validate_answer('{"answerable": true, "answer": "1 mois", "citations": ["F7777"]}', ["F0001"])
    assert bad is None and "hors" in err
    rec, _ = validate_answer('{"answerable": true, "answer": "1 mois (F0001)", "citations": []}', ["F0001"])
    assert rec.citations == ["F0001"]
    no, err = validate_answer("pas du json", ["F0001"])
    assert no is None
    refuse, _ = validate_answer('```json\n{"answerable": false, "answer": "Je ne sais pas", "citations": []}\n```', [])
    assert refuse and not refuse.answerable
