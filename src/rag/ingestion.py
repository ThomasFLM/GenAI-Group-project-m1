"""Ingestion: download, XML parsing of Service-Public fact sheets, cleaning, selection.

Official source: the ``vosdroits-latest.zip`` archive published by DILA (dataset
"Fiches pratiques et ressources de Service-Public.fr Particuliers" on data.gouv.fr).
Each fact sheet is an XML file ``F*.xml`` whose root is ``<Publication ID="F1234">``.
XML element names (Chapitre, Cas, Paragraphe...) are French: they come from the official schema.

Alternative source: the DINUM "vectorised" Parquet export (Hugging Face). Only the text is
reused; chunking and embeddings remain our own.
"""
from __future__ import annotations

import json
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Iterable, Iterator
import xml.etree.ElementTree as ET

from .schemas import Document

# Elements with no value for answering questions (see XSD schema 3.5, Publication.xsd):
# navigation, links, addresses, media, personalisation logic (Condition), glossary.
SKIP_TAGS = {
    "FilDAriane", "Menu", "SousThemePere", "DossierPere", "SousDossier", "SousTheme", "Dossier", "Fiche",
    "FichePrecedente", "FicheSuivante", "RechercheGuidee", "RechercheGuideePere", "Questionnaire",
    "VoirAussi", "FicheConnexe", "Partenaire", "ServiceEnLigne", "PourEnSavoirPlus", "Reference",
    "OuSAdresser", "ListeOuSAdresser", "QuiPeutMAider", "Definition", "Abreviation", "QuestionReponse",
    "CommentFaireSi", "FAQ", "LienExterneCommente", "LienWeb", "Lien", "Theme", "Audience", "Canal",
    "Cible", "SurTitre", "Condition", "TitreAlternatif", "Image", "Video", "Infographie",
    "ImageIllustration", "VideoIllustration", "Colonne", "Affichage",
}
# Blocks that open a (sub-)section: their Titre becomes the chunk's section path
SECTION_TAGS = {"Chapitre", "SousChapitre", "Situation", "Cas", "ASavoir", "ANoter", "Attention",
                "Exemple", "Rappel", "Consigne", "Complement", "Aide", "Introduction", "Avertissement",
                "Conclusion"}
CALLOUT_LABELS = {"ASavoir": "À savoir", "ANoter": "À noter", "Attention": "Attention",
                  "Avertissement": "Avertissement", "Exemple": "Exemple", "Rappel": "Rappel",
                  "Consigne": "Consigne", "Conclusion": "Conclusion"}
# Publication types kept (@type attribute, possibly suffixed with "conditionnée").
# Excluded: "Recherche guidée" (decision-tree questionnaires, up to 850,000 words),
# "Fiche avec liens externes" (links only), "Comment faire si" (aggregates), redirected sheets.
KEEP_PUBLICATION_TYPES = ("Fiche d'information", "Fiche Question-réponse")

_DC = "{http://purl.org/dc/elements/1.1/}"


def local(tag: str) -> str:
    return tag.split("}", 1)[-1]


def clean_text(text: str) -> str:
    """Normalise whitespace and characters (NBSP, thin spaces, repeated newlines)."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ").replace("\u202f", " ").replace("\u2009", " ")
    text = re.sub(r"(?<=\S)[ \t]+", " ", text)
    text = re.sub(r" *\n", "\n", text)
    text = re.sub(r"\n +(?! *- )", "\n", text)  # keep the indentation of nested "- " lists
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _raw_inline(el: ET.Element) -> str:
    """Concatenate the text of an element and of its inline children without stripping anything:
    spaces are sometimes inside the tags ("validité </MiseEnEvidence>de")."""
    parts = [el.text or ""]
    for child in el:
        if local(child.tag) not in SKIP_TAGS:
            parts.append(_raw_inline(child))
        parts.append(child.tail or "")
    return "".join(parts)


def _inline_text(el: ET.Element) -> str:
    """Text of an element including its inline children (links, MiseEnEvidence, Exposant...).
    No space is added between elements: "1<Exposant>er</Exposant>" -> "1er"."""
    return clean_text(_raw_inline(el).replace("\n", " "))


def _title_of(el: ET.Element) -> str:
    for child in el:
        if local(child.tag) == "Titre":
            return _inline_text(child)
    return ""


def _list_lines(liste: ET.Element, depth: int = 0) -> list[str]:
    """Linearise a (possibly nested) Liste into bullet lines."""
    lines = []
    for item in liste:
        if local(item.tag) != "Item":
            continue
        texts, sub = [], []
        for child in item:
            tag = local(child.tag)
            if tag == "Paragraphe":
                texts.append(_inline_text(child))
            elif tag == "Liste":
                sub.extend(_list_lines(child, depth + 1))
        text = " ".join(t for t in texts if t)
        if text:
            lines.append("  " * depth + "- " + text)
        lines.extend(sub)
    return lines


def _cell_text(cell: ET.Element) -> str:
    parts = []
    for child in cell:
        tag = local(child.tag)
        if tag == "Paragraphe":
            parts.append(_inline_text(child))
        elif tag == "Liste":
            parts.extend(line.strip("- ") for line in _list_lines(child))
    if not parts:  # "mixed" cell with direct text
        parts.append(_inline_text(cell))
    return " ; ".join(p for p in parts if p)


def _table_lines(table: ET.Element) -> list[str]:
    """Linearise a Tableau. If a header row exists, each row becomes
    "Header 1 : value ; Header 2 : value", which keeps the amount <-> situation link."""
    lines = []
    for child in table:
        if local(child.tag) in {"TitreRiche", "Resume"}:
            t = _inline_text(child)
            if t:
                lines.append(t)
    headers: list[str] = []
    for row in table:
        if local(row.tag) not in {"Rangée", "Rangee"}:
            continue
        cells = [_cell_text(c) for c in row if local(c.tag) == "Cellule"]
        if row.attrib.get("type") == "header" and not headers:
            headers = cells
            continue
        if headers and len(headers) == len(cells):
            lines.append(" ; ".join(f"{h} : {c}" if h else c for h, c in zip(headers, cells) if c))
        elif any(cells):
            lines.append(" | ".join(cells))
    return lines


def _walk(el: ET.Element, path: list[str], sections: list[dict[str, str]], buf: list[str]) -> None:
    tag = local(el.tag)
    if tag in SKIP_TAGS:
        return
    if tag in SECTION_TAGS:
        _flush(path, sections, buf)
        title = _title_of(el) or CALLOUT_LABELS.get(tag, "")
        new_path = path + [title] if title and tag not in {"Introduction", "Aide"} else path
        for child in el:
            if local(child.tag) in {"Titre", "TitreRiche"}:  # TitreRiche: duplicate of the title inside Cas
                continue
            _walk(child, new_path, sections, buf)
        _flush(new_path, sections, buf)
        return
    if tag == "Liste":
        buf.extend(_list_lines(el))
        return
    if tag == "Tableau":
        buf.extend(_table_lines(el))
        return
    if tag in {"Paragraphe", "TitreFlottant"}:
        txt = _inline_text(el)
        if txt:
            buf.append(txt)
        return
    for child in el:
        _walk(child, path, sections, buf)


def _flush(path: list[str], sections: list[dict[str, str]], buf: list[str]) -> None:
    text = clean_text("\n".join(buf))
    buf.clear()
    if text:
        sections.append({"heading": " > ".join(p for p in path if p), "text": text})


def _parse_date(raw: str) -> str:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", raw or "")
    return m.group(1) if m else ""


def parse_fiche_xml(xml_bytes: bytes) -> Document | None:
    """Parse a Service-Public XML fact sheet. Returns None if it is not a usable fact sheet."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None
    if local(root.tag) != "Publication":
        return None
    doc_id = root.attrib.get("ID", "")
    pub_type = root.attrib.get("type", "")
    if not doc_id or not pub_type.startswith(KEEP_PUBLICATION_TYPES):
        return None

    def dc(name: str) -> str:
        el = root.find(f"{_DC}{name}")
        return clean_text("".join(el.itertext())) if el is not None else ""

    theme = ""
    th = root.find("Theme")
    if th is not None:
        theme = _title_of(th)
    theme = theme or dc("subject")
    subtheme = ""
    stp = root.find("SousThemePere")
    if stp is not None:
        subtheme = clean_text("".join(stp.itertext()))
    audience_el = root.find("Audience")

    sections: list[dict[str, str]] = []
    buf: list[str] = []
    for child in root:
        if local(child.tag) in {"title", "description", "subject", "date", "type", "creator",
                                "publisher", "contributor", "format", "identifier", "source",
                                "language", "relation", "coverage", "rights"}:
            continue
        _walk(child, [], sections, buf)
    _flush([], sections, buf)

    title = dc("title")
    if not title or not sections:
        return None
    return Document(
        doc_id=doc_id,
        title=title,
        url=root.attrib.get("spUrl", f"https://www.service-public.gouv.fr/particuliers/vosdroits/{doc_id}"),
        doc_type=pub_type,
        theme=theme,
        subtheme=subtheme,
        audience=_inline_text(audience_el) if audience_el is not None else "Particuliers",
        date=_parse_date(dc("date")) or _parse_date(root.attrib.get("dateDerniereModificationImportante", "")),
        description=dc("description"),
        sections=sections,
    )


def iter_xml_sources(source: Path) -> Iterator[tuple[str, bytes]]:
    """Iterate over (name, content) of the XML files of a zip archive or a directory."""
    if source.is_dir():
        for p in sorted(source.rglob("*.xml")):
            yield p.name, p.read_bytes()
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as zf:
            for name in sorted(zf.namelist()):
                if name.endswith(".xml"):
                    yield Path(name).name, zf.read(name)
    else:
        raise ValueError(f"Unrecognised source: {source}")


def load_fiches(source: Path, prefix: str = "F") -> list[Document]:
    """Load the fact sheets (F*.xml files) of an archive or a directory."""
    docs = []
    for name, content in iter_xml_sources(source):
        if not re.fullmatch(rf"{prefix}\d+\.xml", name):
            continue
        doc = parse_fiche_xml(content)
        if doc is not None:
            docs.append(doc)
    return docs


# Possible column names in the "vectorised" Parquet exports
_PARQUET_COLS = {
    "doc_id": ["sid", "doc_id", "id", "fiche_id"],
    "title": ["title", "titre"],
    "url": ["url"],
    "theme": ["theme", "surtitre", "thematique"],
    "audience": ["audience"],
    "date": ["date", "date_modification", "modified"],
    "text": ["text", "chunk_text", "content", "texte"],
    "order": ["chunk_index", "chunk_id", "position"],
}


def load_parquet(path: Path) -> list[Document]:
    """Rebuild documents from a Parquet export (one or more chunks per fact sheet)."""
    import pyarrow.parquet as pq

    table = pq.read_table(path).to_pylist()
    if not table:
        return []
    cols = table[0].keys()

    def pick(key: str) -> str | None:
        return next((c for c in _PARQUET_COLS[key] if c in cols), None)

    c = {k: pick(k) for k in _PARQUET_COLS}
    if not c["doc_id"] or not c["text"]:
        raise ValueError(f"Columns not found in {path}: {sorted(cols)}")
    grouped: dict[str, list[dict]] = {}
    for row in table:
        grouped.setdefault(str(row[c["doc_id"]]), []).append(row)
    docs = []
    for doc_id, rows in grouped.items():
        if c["order"]:
            rows.sort(key=lambda r: str(r.get(c["order"]) or ""))
        first = rows[0]
        texts = []
        for r in rows:
            t = clean_text(str(r[c["text"]] or ""))
            if t and t not in texts:  # chunked exports often overlap
                texts.append(t)
        docs.append(Document(
            doc_id=doc_id,
            title=str(first.get(c["title"]) or doc_id) if c["title"] else doc_id,
            url=str(first.get(c["url"]) or "") if c["url"] else "",
            theme=str(first.get(c["theme"]) or "") if c["theme"] else "",
            audience=str(first.get(c["audience"]) or "Particuliers") if c["audience"] else "Particuliers",
            date=_parse_date(str(first.get(c["date"]) or "")) if c["date"] else "",
            sections=[{"heading": "", "text": "\n\n".join(texts)}],
        ))
    return docs


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def select_documents(docs: Iterable[Document], themes: list[str], max_docs: int,
                     must_include: set[str] | None = None) -> list[Document]:
    """Deterministic selection of a coherent sub-corpus.

    1. keep the fact sheets whose theme matches one of the requested themes;
    2. force the inclusion of the golden-set documents (``must_include``);
    3. fill up to ``max_docs`` by spreading evenly across themes, in identifier
       order (reproducible).
    """
    must_include = must_include or set()
    wanted = [_norm(t) for t in themes]
    docs = list(docs)
    by_id = {d.doc_id: d for d in docs}
    selected: dict[str, Document] = {i: by_id[i] for i in sorted(must_include) if i in by_id}

    buckets: dict[str, list[Document]] = {w: [] for w in wanted}
    for d in sorted(docs, key=lambda d: (len(d.doc_id), d.doc_id)):
        nt = _norm(d.theme)
        for w in wanted:
            if w and w in nt:
                buckets[w].append(d)
                break
    # round-robin across themes for a balanced corpus
    iters = [iter(b) for b in buckets.values()]
    while len(selected) < max_docs and iters:
        alive = []
        for it in iters:
            d = next(it, None)
            if d is None:
                continue
            alive.append(it)
            if d.doc_id not in selected:
                selected[d.doc_id] = d
            if len(selected) >= max_docs:
                break
        iters = alive
    return sorted(selected.values(), key=lambda d: (len(d.doc_id), d.doc_id))


def download(url: str, dest: Path) -> Path:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for block in r.iter_content(1 << 16):
                f.write(block)
    return dest


def save_documents(docs: list[Document], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(d.model_dump_json() + "\n")


def load_documents(path: Path) -> list[Document]:
    with open(path, encoding="utf-8") as f:
        return [Document.model_validate_json(line) for line in f if line.strip()]


def golden_doc_ids(golden_path: Path) -> set[str]:
    ids: set[str] = set()
    if golden_path.exists():
        for line in golden_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.update(json.loads(line).get("gold_docs", []))
    return ids

