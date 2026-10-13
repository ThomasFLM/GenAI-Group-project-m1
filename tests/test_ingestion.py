from rag.chunking import chunk_document, split_section
from rag.ingestion import clean_text, select_documents


def test_parse_fiche(docs):
    by_id = {d.doc_id: d for d in docs}
    d = by_id["F0001"]
    assert d.title == "Dépôt de garantie d'un logement loué"
    assert d.theme == "Logement"
    assert d.date == "2025-01-15"
    assert d.url.endswith("F0001")
    headings = [s["heading"] for s in d.sections]
    assert "Montant > Location vide" in headings
    text = d.text
    assert "1 mois de loyer hors charges" in text
    assert "- 2 mois dans le cas contraire." in text
    assert "Accueil particuliers" not in text  # fil d'Ariane ignoré
    assert "Bail" not in text                 # « Voir aussi » ignoré
    assert by_id["F0002"].url == "https://www.service-public.fr/particuliers/vosdroits/F0002"


def test_clean_text():
    assert clean_text("a  b\n\n\n\nc ") == "a b\n\nc"


def test_select_documents_balanced_and_must_include(docs):
    sel = select_documents(docs, ["Transports - Mobilité"], max_docs=1)
    assert [d.doc_id for d in sel] == ["F0002"]
    sel = select_documents(docs, ["Transports - Mobilité"], max_docs=1, must_include={"F0001"})
    assert {d.doc_id for d in sel} == {"F0001"}
    sel = select_documents(docs, ["logement", "transports"], max_docs=10)
    assert len(sel) == 2


def test_chunks_respect_sections(docs):
    d = next(x for x in docs if x.doc_id == "F0001")
    chunks = chunk_document(d, max_words=50, overlap_words=5)
    assert chunks[0].section == "Résumé"
    for c in chunks:
        assert c.doc_id == "F0001" and c.chunk_id.startswith("F0001#")
        assert not ("Location vide" in c.section and "2 mois" in c.text)
    assert "Dépôt de garantie" in chunks[1].indexed_text


def test_split_section_sizes():
    text = "\n".join(f"phrase {i} " + "mot " * 30 for i in range(10))
    pieces = split_section(text, max_words=80, overlap_words=10)
    assert len(pieces) > 1
    assert all(len(p.split()) <= 80 + 10 + 32 for p in pieces)
    # recouvrement : la fin d'un passage est reprise au début du suivant
    assert pieces[0].split()[-3:] == pieces[1].split()[:3]


def test_schema_35_table_condition_nested_list(docs):
    d = next(x for x in docs if x.doc_id == "F0003")
    text = d.text
    # tableau linéarisé avec les en-têtes de colonnes (élément <Rangée>)
    assert "Demandeur : Majeur ; Prix : 86 €" in text
    assert "Demandeur : Mineur de 15 ans et plus ; Prix : 42 €" in text
    assert "Tarif selon l'âge" in text
    # liste imbriquée conservée, service en ligne, condition et titre alternatif ignorés
    assert "- Achat en ligne\n  - sur le site de l'ANTS" in text
    assert "Acheter un timbre" not in text and "majeur\"" not in text
    assert "Combien ça coûte" not in text
    assert any(s["heading"] == "Prix du timbre fiscal > Exemple" for s in d.sections)
    # date de secours : attribut dateDerniereModificationImportante
    assert d.date == "2025-03-01"


def test_redirected_fiche_skipped(docs):
    assert "F0004" not in {d.doc_id for d in docs}
