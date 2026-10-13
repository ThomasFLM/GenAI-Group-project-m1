from rag.evaluation import aggregate, hit_at_k, lexical_scores, reciprocal_rank, recall_at_k, score_one


def test_retrieval_metrics():
    assert hit_at_k(["a", "b"], ["b"], 1) == 0 and hit_at_k(["a", "b"], ["b"], 2) == 1
    assert recall_at_k(["a", "b"], ["b", "c"], 2) == 0.5
    assert reciprocal_rank(["a", "b"], ["b"]) == 0.5


def test_lexical():
    s = lexical_scores("Le délai est de 45 jours.", "45 jours")
    assert s["number_recall"] == 1.0 and s["token_recall"] == 1.0


def _res(**kw):
    base = {"answer": "", "answerable": True, "cited_sources": [], "retrieved_doc_ids": [], "stop_reason": "stop",
            "latency_s": 1.0, "tokens_in": 10, "tokens_out": 5, "cost_usd": 0.0, "tool_calls": [], "llm_calls": []}
    return base | kw


def test_score_and_errors():
    q = {"id": "q1", "question": "?", "answer": "45 jours", "gold_docs": ["F2"], "category": "factual"}
    s = score_one(q, _res(answer="45 jours", cited_sources=["F2"], retrieved_doc_ids=["F2"]), 5,
                  {"verdict": "correct", "justification": ""})
    assert s["error"] == "ok" and s["mrr"] == 1.0 and s["citation_precision"] == 1.0
    s = score_one(q, _res(answerable=False, retrieved_doc_ids=["F1"]), 5)
    assert s["error"] == "retrieval_miss"
    s = score_one(q, _res(answerable=False, retrieved_doc_ids=["F2"]), 5)
    assert s["error"] == "false_refusal"
    u = {"id": "q2", "question": "?", "answer": "", "gold_docs": [], "category": "unanswerable"}
    assert score_one(u, _res(answerable=True), 5)["error"] == "hallucination"
    i = {"id": "q3", "question": "?", "answer": "Refus", "gold_docs": [], "category": "injection"}
    assert score_one(i, _res(answerable=False), 5)["injection_resisted"]
    m = aggregate([s, score_one(u, _res(answerable=False), 5)], [_res(), _res()], 5)
    assert m["abstention"]["unanswerable_abstention_rate"] == 1.0
    assert m["cost_latency"]["latency_p50_s"] == 1.0
