"""Modèles de données (pydantic) : documents, passages, sorties structurées, traces."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

REFUSAL_TEXT = (
    "Je ne sais pas : cette information ne figure pas dans les documents dont je dispose."
)
INJECTION_REFUSAL_TEXT = (
    "Je ne peux pas suivre cette demande. Je réponds uniquement à des questions "
    "sur les démarches administratives, à partir des fiches Service-Public."
)


class Document(BaseModel):
    doc_id: str                      # identifiant Service-Public, ex. "F1234"
    title: str
    url: str = ""
    theme: str = ""
    subtheme: str = ""
    audience: str = "Particuliers"
    date: str = ""                   # date de dernière modification (YYYY-MM-DD)
    description: str = ""
    sections: list[dict[str, str]] = Field(default_factory=list)  # [{"heading": ..., "text": ...}]

    @property
    def text(self) -> str:
        return "\n\n".join(
            (f"## {s['heading']}\n" if s.get("heading") else "") + s["text"] for s in self.sections
        )


class Chunk(BaseModel):
    chunk_id: str                    # "F1234#3"
    doc_id: str
    title: str
    section: str = ""
    text: str                        # texte brut du passage
    url: str = ""
    theme: str = ""
    date: str = ""

    @property
    def indexed_text(self) -> str:
        """Texte indexé : en-tête contextuel (titre + section) + passage."""
        header = self.title + (f" — {self.section}" if self.section else "")
        return f"{header}\n{self.text}"


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float                     # score final (rerank si actif, sinon RRF)
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rerank_score: float | None = None


class LLMAnswer(BaseModel):
    """Sortie structurée attendue du LLM (validée par pydantic)."""
    answerable: bool = Field(description="true si les extraits permettent de répondre")
    answer: str = Field(description="réponse concise en français, ou explication du refus")
    citations: list[str] = Field(default_factory=list,
                                 description="identifiants des documents cités, ex. ['F1234']")


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    latency_s: float = 0.0
    result_summary: str = ""


class LLMCall(BaseModel):
    purpose: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    stop_reason: str = ""
    cost_usd: float = 0.0


class AssistantResponse(BaseModel):
    question: str
    answer: str
    answerable: bool
    cited_sources: list[str]
    retrieved_doc_ids: list[str]
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    stop_reason: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    tool_calls: list[ToolCall] = Field(default_factory=list)
    llm_calls: list[LLMCall] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)
    model: str = ""
    mode: str = "rag"
