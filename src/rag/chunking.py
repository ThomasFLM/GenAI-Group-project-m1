"""Splitting fact sheets into chunks.

Strategy (justified in docs/architecture.md):
- the grouping unit is the **chapter** (first heading level): a chunk never spans two
  chapters;
- inside a chapter, sub-sections (cases, situations, "À savoir" boxes...) are packed up to
  ``max_words`` words and their heading is **kept in the text** ("### Location vide"): the
  model knows which case each sentence applies to, without producing crumbs of a few words
  (real fact sheets contain dozens of small cases);
- a sub-heading is never separated from its first paragraph; a chunk that starts in the
  middle of a case repeats its heading followed by "(suite)";
- ``overlap_words`` words of overlap between consecutive chunks of the same chapter;
- each chunk is indexed with a contextual header "fact sheet title — chapter".
The markers written into the chunk text ("(suite)", "Résumé") are in French, like the corpus.
"""
from __future__ import annotations

from .schemas import Chunk, Document


def _split_long(par: str, max_words: int) -> list[str]:
    words = par.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def pack_units(units: list[tuple[str, str]], max_words: int, overlap_words: int) -> list[str]:
    """Pack (sub-heading, paragraph) units into chunks of at most ~max_words words.

    A unit's sub-heading is written once, at the start of the block it introduces."""
    # 1. split paragraphs that are too long, the sub-heading stays with the first piece
    pieces: list[tuple[str, str, bool]] = []  # (sub-heading, text, starts_the_case)
    last_sub = None
    for sub, par in units:
        parts = _split_long(par, max_words) if len(par.split()) > max_words else [par]
        for i, p in enumerate(parts):
            pieces.append((sub, p, sub != last_sub and i == 0))
            last_sub = sub

    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sub, text, starts_case in pieces:
        line = f"### {sub}\n{text}" if (sub and starts_case) else text
        n = len(line.split())
        if current and count + n > max_words:
            chunks.append("\n".join(current))
            body = " ".join(line for c in current for line in c.split("\n") if not line.startswith("### "))
            tail = " ".join(body.split()[-overlap_words:]) if overlap_words else ""
            current, count = [], 0
            if sub and not starts_case:  # resuming in the middle of a case: repeat its heading
                current.append(f"### {sub} (suite)")
                count += len(sub.split()) + 1
            if tail:
                current.append(f"[…] {tail}")
                count += overlap_words
        current.append(line)
        count += n
    if current:
        chunks.append("\n".join(current))
    return chunks


def split_section(text: str, max_words: int, overlap_words: int) -> list[str]:
    """Split raw text (without sub-headings) into chunks."""
    units = [("", p.rstrip()) for p in text.split("\n") if p.strip()]
    return pack_units(units, max_words, overlap_words)


def _chapters(doc: Document) -> list[tuple[str, list[tuple[str, str]]]]:
    """Group consecutive sections by chapter (first level of the heading path)."""
    groups: list[tuple[str, list[tuple[str, str]]]] = []
    sections = list(doc.sections)
    if doc.description:
        sections.insert(0, {"heading": "Résumé", "text": doc.description})
    for sec in sections:
        path = [p for p in sec.get("heading", "").split(" > ") if p]
        chapter = path[0] if path else ""
        sub = " > ".join(path[1:])
        units = [(sub, p.rstrip()) for p in sec["text"].split("\n") if p.strip()]
        if groups and groups[-1][0] == chapter:
            groups[-1][1].extend(units)
        else:
            groups.append((chapter, units))
    return groups


def chunk_document(doc: Document, max_words: int = 220, overlap_words: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    for chapter, units in _chapters(doc):
        for piece in pack_units(units, max_words, overlap_words):
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}#{len(chunks)}",
                doc_id=doc.doc_id,
                title=doc.title,
                section=chapter,
                text=piece,
                url=doc.url,
                theme=doc.theme,
                date=doc.date,
            ))
    return chunks


def chunk_documents(docs: list[Document], max_words: int = 220, overlap_words: int = 40) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunk_document(d, max_words, overlap_words))
    return out
