"""Découpage des fiches en passages.

Stratégie (justifiée dans docs/architecture.md) :
- on respecte la structure de la fiche : un passage ne chevauche jamais deux sections
  (les fiches Service-Public sont organisées en cas / situations : mélanger
  « locataire » et « propriétaire » dans un même passage provoque des erreurs) ;
- les sections trop longues sont découpées par paragraphes, regroupés jusqu'à
  ``max_words`` mots, avec un recouvrement de ``overlap_words`` mots ;
- chaque passage est indexé avec un en-tête contextuel « titre — section ».
"""
from __future__ import annotations

from .schemas import Chunk, Document


def _split_long_paragraph(par: str, max_words: int) -> list[str]:
    words = par.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def split_section(text: str, max_words: int, overlap_words: int) -> list[str]:
    paragraphs: list[str] = []
    for p in text.split("\n"):
        p = p.rstrip()
        if not p.strip():
            continue
        paragraphs.extend(_split_long_paragraph(p, max_words) if len(p.split()) > max_words else [p])

    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for p in paragraphs:
        n = len(p.split())
        if current and count + n > max_words:
            chunks.append("\n".join(current))
            # recouvrement : on reprend la fin du passage précédent
            tail_words = " ".join(current).split()[-overlap_words:] if overlap_words else []
            current = [" ".join(tail_words)] if tail_words else []
            count = len(tail_words)
        current.append(p)
        count += n
    if current:
        chunks.append("\n".join(current))
    return chunks


def chunk_document(doc: Document, max_words: int = 220, overlap_words: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    sections = list(doc.sections)
    if doc.description:
        sections.insert(0, {"heading": "Résumé", "text": doc.description})
    for sec in sections:
        for piece in split_section(sec["text"], max_words, overlap_words):
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}#{len(chunks)}",
                doc_id=doc.doc_id,
                title=doc.title,
                section=sec.get("heading", ""),
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
