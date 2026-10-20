from rag.chunking import chunk_document, pack_units, split_section
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
    assert "Accueil particuliers" not in text  # breadcrumbs ignored
    assert "Bail" not in text                 # "Voir aussi" ignored
    assert by_id["F0002"].url == "https://www.service-public.gouv.fr/particuliers/vosdroits/F0002"


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
    montant = next(c for c in chunks if c.section == "Montant")
    # cases of the same chapter are grouped, each one keeps its heading in the text
    assert "### Location vide\nLe montant du dépôt de garantie est limité à 1 mois" in montant.text
    assert montant.text.index("### Location meublée") < montant.text.index("2 mois")
    assert "Dépôt de garantie" in chunks[1].indexed_text


def test_split_section_sizes():
    text = "\n".join(f"phrase {i} " + "mot " * 30 for i in range(10))
    pieces = split_section(text, max_words=80, overlap_words=10)
    assert len(pieces) > 1
    assert all(len(p.split()) <= 80 + 10 + 32 for p in pieces)
    # overlap: the last 10 words of a chunk are repeated at the start of the next one
    assert pieces[1].split()[0] == "[…]"
    assert pieces[1].split()[1:11] == pieces[0].split()[-10:]


def test_pack_units_keeps_heading_with_text_and_repeats_it():
    units = [("Cas A", "intro " * 5)] + [("Cas A", f"para{i} " + "mot " * 20) for i in range(6)]
    pieces = pack_units(units, max_words=60, overlap_words=0)
    assert pieces[0].startswith("### Cas A\nintro")
    assert all(p.startswith("### Cas A (suite)") for p in pieces[1:])
    assert not any(p.rstrip().endswith("### Cas A") for p in pieces)


def test_schema_35_table_condition_nested_list(docs):
    d = next(x for x in docs if x.doc_id == "F0003")
    text = d.text
    # table linearised with the column headers (<Rangée> element)
    assert "Demandeur : Majeur ; Prix : 86 €" in text
    assert "Demandeur : Mineur de 15 ans et plus ; Prix : 42 €" in text
    assert "Tarif selon l'âge" in text
    # nested list kept; online service, condition and alternative title ignored
    assert "- Achat en ligne\n  - sur le site de l'ANTS" in text
    assert "Acheter un timbre" not in text and "majeur\"" not in text
    assert "Combien ça coûte" not in text
    assert any(s["heading"] == "Prix du timbre fiscal > Exemple" for s in d.sections)
    # fallback date: dateDerniereModificationImportante attribute
    assert d.date == "2025-03-01"


def test_redirected_fiche_skipped(docs):
    assert "F0004" not in {d.doc_id for d in docs}


def test_inline_elements_no_spurious_spaces():
    from rag.ingestion import parse_fiche_xml

    xml = ('<Publication xmlns:dc="http://purl.org/dc/elements/1.1/" ID="F9" type="Fiche d\'information">'
           '<dc:title>T</dc:title><Texte><Paragraphe>Au plus tard le 6<Exposant>e</Exposant> vendredi, '
           'à <MiseEnEvidence>18 ans</MiseEnEvidence>. Voir <LienInterne LienPublication="R1">CNRACL</LienInterne>.'
           '</Paragraphe></Texte></Publication>').encode()
    assert parse_fiche_xml(xml).text == "Au plus tard le 6e vendredi, à 18 ans. Voir CNRACL."


def test_spaces_inside_inline_elements_are_kept():
    from rag.ingestion import parse_fiche_xml

    xml = ('<Publication xmlns:dc="http://purl.org/dc/elements/1.1/" ID="F9" type="Fiche d\'information">'
           '<dc:title>T</dc:title><Texte><Paragraphe>la <MiseEnEvidence>durée de validité </MiseEnEvidence>de la '
           'carte est passée de <MiseEnEvidence>10 </MiseEnEvidence>à <MiseEnEvidence>15<MiseEnEvidence/> ans'
           '</MiseEnEvidence>.</Paragraphe></Texte></Publication>').encode()
    assert parse_fiche_xml(xml).text == "la durée de validité de la carte est passée de 10 à 15 ans."
