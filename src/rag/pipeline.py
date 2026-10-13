"""Assistant : orchestration RAG (une recherche → une génération) et mode agent (outils).

Chaque réponse est instrumentée : tokens, latence (totale et par étape), appels d'outils,
appels LLM, coût, raison d'arrêt.
"""
from __future__ import annotations

import json
import re
import time
from datetime import date, timedelta

from .config import Settings, get_settings
from .guardrails import detect_injection, leaks_system_prompt, validate_answer
from .llm import LLMError, LLMResult, make_llm
from .prompts import (AGENT_FINAL_INSTRUCTION, AGENT_SYSTEM_PROMPT, ANSWER_JSON_SCHEMA, SYSTEM_PROMPT,
                      build_user_message, format_context)
from .retrieval import HybridRetriever, unique_doc_ids
from .schemas import (INJECTION_REFUSAL_TEXT, REFUSAL_TEXT, AssistantResponse, LLMAnswer, LLMCall,
                      RetrievedChunk, ToolCall)

AGENT_TOOLS = [
    {
        "name": "search_corpus",
        "description": "Recherche hybride (BM25 + sémantique + reranking) dans les fiches Service-Public. "
                       "Renvoie les passages les plus pertinents avec l'identifiant de fiche.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "requête en français"},
                "since": {"type": "string", "description": "optionnel : ne garder que les fiches mises à jour "
                                                           "depuis cette date (YYYY-MM-DD)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_document",
        "description": "Renvoie le texte complet d'une fiche à partir de son identifiant (ex. F1234).",
        "parameters": {
            "type": "object",
            "properties": {"doc_id": {"type": "string"}},
            "required": ["doc_id"],
        },
    },
    {
        "name": "add_days",
        "description": "Calcule la date obtenue en ajoutant un nombre de jours à une date (YYYY-MM-DD).",
        "parameters": {
            "type": "object",
            "properties": {"start_date": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["start_date", "days"],
        },
    },
]


class Trace:
    """Accumule les mesures d'une réponse."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.llm_calls: list[LLMCall] = []
        self.tool_calls: list[ToolCall] = []
        self.timings: dict[str, float] = {}
        self.flags: list[str] = []

    def add_llm(self, purpose: str, r: LLMResult) -> None:
        self.llm_calls.append(LLMCall(purpose=purpose, model=r.model, tokens_in=r.tokens_in,
                                      tokens_out=r.tokens_out, latency_s=r.latency_s,
                                      stop_reason=r.stop_reason, cost_usd=r.cost_usd))
        self.timings["llm_s"] = self.timings.get("llm_s", 0.0) + r.latency_s

    def add_timing(self, key: str, value: float) -> None:
        self.timings[key] = self.timings.get(key, 0.0) + value

    def response(self, question: str, answer: str, answerable: bool, cited: list[str],
                 retrieved: list[RetrievedChunk], stop_reason: str, model: str, mode: str) -> AssistantResponse:
        return AssistantResponse(
            question=question, answer=answer, answerable=answerable, cited_sources=cited,
            retrieved_doc_ids=unique_doc_ids(retrieved),
            retrieved_chunk_ids=[r.chunk.chunk_id for r in retrieved],
            stop_reason=stop_reason,
            tokens_in=sum(c.tokens_in for c in self.llm_calls),
            tokens_out=sum(c.tokens_out for c in self.llm_calls),
            cost_usd=sum(c.cost_usd for c in self.llm_calls),
            latency_s=time.perf_counter() - self.t0,
            tool_calls=self.tool_calls, llm_calls=self.llm_calls, timings=self.timings,
            flags=self.flags, model=model, mode=mode,
        )


def extractive_answer(results: list[RetrievedChunk]) -> LLMAnswer:
    """Baseline sans LLM : les deux premières phrases du meilleur passage."""
    top = results[0].chunk
    sentences = re.split(r"(?<=[.!?])\s+", top.text.replace("\n", " "))
    return LLMAnswer(answerable=True, answer=" ".join(sentences[:2]).strip(), citations=[top.doc_id])


class Assistant:
    def __init__(self, retriever: HybridRetriever, settings: Settings | None = None, llm=None):
        self.settings = settings or get_settings()
        self.retriever = retriever
        self.llm = llm if llm is not None else make_llm(self.settings)

    @classmethod
    def from_disk(cls, settings: Settings | None = None) -> "Assistant":
        settings = settings or get_settings()
        return cls(HybridRetriever.load(settings), settings)

    @property
    def model_name(self) -> str:
        return getattr(self.llm, "model", "extractive")

    # ------------------------------------------------------------------ outils
    def _search(self, trace: Trace, query: str, since: str | None = None, k: int | None = None):
        t0 = time.perf_counter()
        results = self.retriever.search(query, k=k, since=since)
        dt = time.perf_counter() - t0
        trace.add_timing("retrieval_s", dt)
        for key, v in self.retriever.last_timings.items():
            trace.add_timing(key, v)
        args = {"query": query} | ({"since": since} if since else {})
        trace.tool_calls.append(ToolCall(name="search_corpus", arguments=args, latency_s=dt,
                                         result_summary=", ".join(r.chunk.chunk_id for r in results)))
        return results

    def _run_tool(self, trace: Trace, name: str, args: dict, seen: list[RetrievedChunk]) -> str:
        if name == "search_corpus":
            res = self._search(trace, str(args.get("query", "")), args.get("since") or None)
            for r in res:
                if r.chunk.chunk_id not in {s.chunk.chunk_id for s in seen}:
                    seen.append(r)
            return format_context(res) or "Aucun résultat."
        t0 = time.perf_counter()
        if name == "get_document":
            doc_id = str(args.get("doc_id", "")).strip()
            out = self.retriever.get_document_text(doc_id) or f"Document {doc_id} introuvable."
            if not out.startswith("Document"):
                for c in self.retriever.chunks:
                    if c.doc_id == doc_id and c.chunk_id not in {s.chunk.chunk_id for s in seen}:
                        seen.append(RetrievedChunk(chunk=c, score=0.0))
                        break
            out = f"<document id=\"{doc_id}\">\n{out}\n</document>"
        elif name == "add_days":
            try:
                d = date.fromisoformat(str(args["start_date"])) + timedelta(days=int(args["days"]))
                out = d.isoformat()
            except (KeyError, ValueError, TypeError) as e:
                out = f"Erreur : {e}"
        else:
            out = f"Outil inconnu : {name}"
        trace.tool_calls.append(ToolCall(name=name, arguments=args, latency_s=time.perf_counter() - t0,
                                         result_summary=out[:120]))
        return out

    # ------------------------------------------------------------------ génération
    def _generate(self, trace: Trace, system: str, messages: list[dict], allowed: list[str],
                  purpose: str) -> tuple[LLMAnswer | None, str]:
        """Appel LLM structuré + validation ; une seconde tentative si la sortie est invalide."""
        last_stop = ""
        for attempt in range(2):
            r = self.llm.chat(system, messages, json_schema=ANSWER_JSON_SCHEMA)
            trace.add_llm(purpose if attempt == 0 else purpose + "_retry", r)
            last_stop = r.stop_reason
            if leaks_system_prompt(r.text):
                trace.flags.append("system_prompt_leak_blocked")
                return None, "output_blocked_leak"
            ans, err = validate_answer(r.text, allowed)
            if ans is not None:
                return ans, last_stop or "stop"
            trace.flags.append(f"validation_error: {err}")
            messages = messages + [
                {"role": "assistant", "content": r.text},
                {"role": "user", "content": f"Ta sortie est invalide ({err}). Réponds uniquement avec le JSON "
                                            f"demandé ; citations parmi : {allowed}."},
            ]
        return None, "validation_failed"

    def _finalize(self, trace: Trace, question: str, ans: LLMAnswer | None, stop: str,
                  retrieved: list[RetrievedChunk], mode: str) -> AssistantResponse:
        if ans is None:
            text = INJECTION_REFUSAL_TEXT if stop == "output_blocked_leak" else REFUSAL_TEXT
            return trace.response(question, text, False, [], retrieved, stop, self.model_name, mode)
        if not ans.answerable:
            return trace.response(question, REFUSAL_TEXT, False, [], retrieved, "abstained", self.model_name, mode)
        return trace.response(question, ans.answer, True, ans.citations, retrieved, stop, self.model_name, mode)

    def _precheck(self, trace: Trace, question: str, mode: str) -> AssistantResponse | None:
        hits = detect_injection(question)
        if hits:
            trace.flags.append("injection_detected")
            if self.settings.injection_policy == "block":
                return trace.response(question, INJECTION_REFUSAL_TEXT, False, [], [], "injection_blocked",
                                      self.model_name, mode)
        return None

    def _low_confidence(self, results: list[RetrievedChunk]) -> bool:
        if not results:
            return True
        top = results[0].rerank_score
        return top is not None and top < self.settings.min_rerank_score

    # ------------------------------------------------------------------ API
    def ask(self, question: str, mode: str = "rag", since: str | None = None) -> AssistantResponse:
        if mode == "agent" and self.llm is not None:
            return self.ask_agent(question)
        trace = Trace()
        blocked = self._precheck(trace, question, "rag")
        if blocked:
            return blocked
        results = self._search(trace, question, since)
        if self._low_confidence(results):
            trace.flags.append("low_retrieval_score")
            return trace.response(question, REFUSAL_TEXT, False, [], results, "low_retrieval_score",
                                  self.model_name, "rag")
        if self.llm is None:
            return self._finalize(trace, question, extractive_answer(results), "extractive", results, "rag")
        allowed = unique_doc_ids(results)
        try:
            ans, stop = self._generate(trace, SYSTEM_PROMPT,
                                       [{"role": "user", "content": build_user_message(question, results)}],
                                       allowed, "answer")
        except LLMError as e:
            trace.flags.append(str(e))
            ans, stop = None, "llm_error"
        return self._finalize(trace, question, ans, stop, results, "rag")

    def ask_agent(self, question: str) -> AssistantResponse:
        trace = Trace()
        blocked = self._precheck(trace, question, "agent")
        if blocked:
            return blocked
        seen: list[RetrievedChunk] = []
        messages: list[dict] = [{"role": "user", "content": f"<question>\n{question}\n</question>"}]
        stop = "max_steps"
        try:
            for _ in range(self.settings.agent_max_steps):
                r = self.llm.chat(AGENT_SYSTEM_PROMPT, messages, tools=AGENT_TOOLS)
                trace.add_llm("agent_step", r)
                if not r.tool_calls:
                    stop = "agent_done"
                    break
                messages.append({"role": "assistant", "content": r.text, "tool_calls": r.tool_calls})
                for call in r.tool_calls:
                    out = self._run_tool(trace, call["name"], call["arguments"], seen)
                    messages.append({"role": "tool", "tool_call_id": call["id"], "name": call["name"],
                                     "content": out})
            if not any(t.name == "search_corpus" for t in trace.tool_calls):
                # l'agent n'a rien cherché : on impose une recherche pour ancrer la réponse
                trace.flags.append("forced_search")
                out = self._run_tool(trace, "search_corpus", {"query": question}, seen)
                messages.append({"role": "user", "content": f"Résultats de recherche :\n{out}"})
            scored = [s for s in seen if s.rerank_score is not None] or seen
            best = sorted(scored, key=lambda s: -(s.rerank_score if s.rerank_score is not None else s.score))
            if self._low_confidence(best[:1]):
                trace.flags.append("low_retrieval_score")
                return trace.response(question, REFUSAL_TEXT, False, [], seen, "low_retrieval_score",
                                      self.model_name, "agent")
            messages.append({"role": "user", "content": AGENT_FINAL_INSTRUCTION})
            ans, final_stop = self._generate(trace, AGENT_SYSTEM_PROMPT, messages, unique_doc_ids(seen),
                                             "agent_final")
            stop = final_stop if stop == "agent_done" or ans is None else stop
        except LLMError as e:
            trace.flags.append(str(e))
            ans, stop = None, "llm_error"
        return self._finalize(trace, question, ans, stop, seen, "agent")


def response_to_json(resp: AssistantResponse) -> str:
    return json.dumps(resp.model_dump(), ensure_ascii=False)
