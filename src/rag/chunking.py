"""Découpage des fiches en passages.

Stratégie (justifiée dans docs/architecture.md) :
- l'unité de regroupement est le **chapitre** (premier niveau de titre) : un passage ne
  chevauche jamais deux chapitres ;
- à l'intérieur d'un chapitre, les sous-sections (cas, situations, encadrés « À savoir »...)
  sont regroupées jusqu'à ``max_words`` mots, et leur titre est **gardé dans le texte**
  (« ### Location vide ») : le modèle sait à quel cas chaque phrase s'applique, sans
  produire des miettes de quelques mots (les vraies fiches ont des dizaines de petits cas) ;
- un sous-titre n'est jamais séparé de son premier paragraphe ; un passage qui commence
  au milieu d'un cas reprend son titre « (suite) » ;
- recouvrement de ``overlap_words`` mots entre passages consécutifs d'un même chapitre ;
- chaque passage est indexé avec un en-tête contextuel « titre de la fiche — chapitre ».
"""
from __future__ import annotations

from .schemas import Chunk, Document


def _split_long(par: str, max_words: int) -> list[str]:
    words = par.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def pack_units(units: list[tuple[str, str]], max_words: int, overlap_words: int) -> list[str]:
    """Regroupe des unités (sous-titre, paragraphe) en passages d'au plus ~max_words mots.

    Le sous-titre d'une unité est écrit une fois, au début du bloc qu'il introduit."""
    # 1. paragraphes trop longs coupés, sous-titre rattaché au premier morceau
    pieces: list[tuple[str, str, bool]] = []  # (sous-titre, texte, est_le_début_du_cas)
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
            if sub and not starts_case:  # on reprend au milieu d'un cas : rappeler son titre
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
    """Découpe un texte brut (sans sous-titres) en passages."""
    units = [("", p.rstrip()) for p in text.split("\n") if p.strip()]
    return pack_units(units, max_words, overlap_words)


def _chapters(doc: Document) -> list[tuple[str, list[tuple[str, str]]]]:
    """Regroupe les sections consécutives par chapitre (premier niveau du chemin de titres)."""
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
