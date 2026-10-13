"""Ingestion : téléchargement, parsing XML des fiches Service-Public, nettoyage, sélection.

Source officielle : archive ``vosdroits-latest.zip`` publiée par la DILA (jeu
« Fiches pratiques et ressources de Service-Public.fr Particuliers » sur data.gouv.fr).
Chaque fiche est un fichier XML ``F*.xml`` dont la racine est ``<Publication ID="F1234">``.

Source alternative : export Parquet « vectorisé » de la DINUM (Hugging Face). Seul le
texte est réutilisé ; le découpage et les embeddings restent les nôtres.
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

# Éléments de navigation / métadonnées sans valeur pour répondre aux questions
SKIP_TAGS = {
    "FilDAriane", "Menu", "SousThemePere", "DossierPere", "SousDossier", "VoirAussi",
    "Partenaire", "ServiceEnLigne", "PourEnSavoirPlus", "Reference", "OuSAdresser",
    "Montant", "InformationComplementaire", "Definition", "Abreviation", "QuestionReponse",
    "CommentFaireSi", "Glossaire", "Actualite", "RessourceConnexe", "LienWeb",
    "Theme", "Audience", "Canal", "PageWeb", "LienExterneCommente",
}
SECTION_TAGS = {"Chapitre", "SousChapitre", "Situation", "Cas", "ASavoir", "ANoter", "Attention",
                "Introduction", "Avertissement"}
BLOCK_TAGS = {"Paragraphe", "Item", "Titre", "Ligne", "Rangee", "Cellule", "Expression"}
CALLOUT_LABELS = {"ASavoir": "À savoir", "ANoter": "À noter", "Attention": "Attention",
                  "Avertissement": "Avertissement"}

_DC = "{http://purl.org/dc/elements/1.1/}"


def local(tag: str) -> str:
    return tag.split("}", 1)[-1]


def clean_text(text: str) -> str:
    """Normalise espaces et caractères (NBSP, espaces fines, retours ligne multiples)."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _inline_text(el: ET.Element) -> str:
    """Texte d'un élément y compris ses liens/expressions inline."""
    return clean_text(" ".join(t for t in el.itertext()))


def _title_of(el: ET.Element) -> str:
    for child in el:
        if local(child.tag) == "Titre":
            return _inline_text(child)
    return ""


def _walk(el: ET.Element, path: list[str], sections: list[dict[str, str]], buf: list[str]) -> None:
    tag = local(el.tag)
    if tag in SKIP_TAGS:
        return
    if tag in SECTION_TAGS:
        _flush(path, sections, buf)
        title = _title_of(el) or CALLOUT_LABELS.get(tag, "")
        new_path = path + [title] if title and tag not in {"Introduction"} else path
        for child in el:
            if local(child.tag) == "Titre":
                continue
            _walk(child, new_path, sections, buf)
        _flush(new_path, sections, buf)
        return
    if tag == "Item":
        buf.append("- " + _inline_text(el))
        return
    if tag == "Ligne" or tag == "Rangee":
        cells = [_inline_text(c) for c in el if local(c.tag) == "Cellule"]
        if cells:
            buf.append(" | ".join(cells))
        return
    if tag == "Paragraphe":
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
    """Parse une fiche XML Service-Public. Renvoie None si ce n'est pas une fiche exploitable."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None
    if local(root.tag) != "Publication":
        return None
    doc_id = root.attrib.get("ID", "")
    if not doc_id:
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
        url=root.attrib.get("spUrl", f"https://www.service-public.fr/particuliers/vosdroits/{doc_id}"),
        theme=theme,
        subtheme=subtheme,
        audience=_inline_text(audience_el) if audience_el is not None else "Particuliers",
        date=_parse_date(dc("date")),
        description=dc("description"),
        sections=sections,
    )


def iter_xml_sources(source: Path) -> Iterator[tuple[str, bytes]]:
    """Itère sur (nom, contenu) des XML d'une archive zip ou d'un dossier."""
    if source.is_dir():
        for p in sorted(source.rglob("*.xml")):
            yield p.name, p.read_bytes()
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as zf:
            for name in sorted(zf.namelist()):
                if name.endswith(".xml"):
                    yield Path(name).name, zf.read(name)
    else:
        raise ValueError(f"Source non reconnue : {source}")


def load_fiches(source: Path, prefix: str = "F") -> list[Document]:
    """Charge les fiches d'information (fichiers F*.xml) d'une archive ou d'un dossier."""
    docs = []
    for name, content in iter_xml_sources(source):
        if not re.fullmatch(rf"{prefix}\d+\.xml", name):
            continue
        doc = parse_fiche_xml(content)
        if doc is not None:
            docs.append(doc)
    return docs


# Noms de colonnes possibles dans les exports Parquet « vectorisés »
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
    """Reconstruit des documents à partir d'un export Parquet (un ou plusieurs chunks par fiche)."""
    import pyarrow.parquet as pq

    table = pq.read_table(path).to_pylist()
    if not table:
        return []
    cols = table[0].keys()

    def pick(key: str) -> str | None:
        return next((c for c in _PARQUET_COLS[key] if c in cols), None)

    c = {k: pick(k) for k in _PARQUET_COLS}
    if not c["doc_id"] or not c["text"]:
        raise ValueError(f"Colonnes introuvables dans {path} : {sorted(cols)}")
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
            if t and t not in texts:  # les exports chunkés ont souvent du recouvrement
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
    """Sélection déterministe d'un sous-corpus cohérent.

    1. garde les fiches dont le thème correspond à un des thèmes demandés ;
    2. force l'inclusion des documents du golden set (``must_include``) ;
    3. complète jusqu'à ``max_docs`` en répartissant équitablement entre thèmes,
       par ordre d'identifiant (reproductible).
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
    # tourniquet entre thèmes pour un corpus équilibré
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

