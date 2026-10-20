"""Data models (pydantic): documents, chunks, structured outputs, traces.

The refusal texts stay in French: they are shown to end users of a French-language assistant.
"""
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
    doc_id: str                      # Service-Public identifier, e.g. "F1234"
    title: str
    url: str = ""
    theme: str = ""
    subtheme: str = ""
    audience: str = "Particuliers"
    doc_type: str = ""               # publication type, e.g. "Fiche d'information conditionnée"
    date: str = ""                   # last modification date (YYYY-MM-DD)
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
    text: str                        # raw chunk text
    url: str = ""
    theme: str = ""
    date: str = ""

    @property
    def indexed_text(self) -> str:
        """Indexed text: contextual header (title + section) + chunk."""
        header = self.title + (f" — {self.section}" if self.section else "")
        return f"{header}\n{self.text}"


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float                     # final score (rerank if enabled, RRF otherwise)
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rerank_score: float | None = None


class LLMAnswer(BaseModel):
    """Structured output expected from the LLM (validated with pydantic)."""
    answerable: bool = Field(description="true if the excerpts contain the answer")
    answer: str = Field(description="concise answer in French, or the reason for refusing")
    citations: list[str] = Field(default_factory=list,
                                 description="identifiers of the cited documents, e.g. ['F1234']")


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
