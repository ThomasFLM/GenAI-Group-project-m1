from rag.retrieval import rrf_fuse, unique_doc_ids


def test_rrf():
    s = rrf_fuse([[1, 2, 3], [3, 1]], k=60)
    assert max(s, key=s.get) == 1
    assert s[3] > s[2]


def test_hybrid_search(retriever):
    res = retriever.search("délai pour contester une amende forfaitaire")
    assert res[0].chunk.doc_id == "F0002"
    res = retriever.search("montant maximal du dépôt de garantie location meublée")
    assert unique_doc_ids(res)[0] == "F0001"


def test_modes_and_date_filter(retriever):
    for mode in ("bm25", "dense"):
        assert retriever.search("contester amende", mode=mode)[0].chunk.doc_id == "F0002"
    res = retriever.search("contester amende", since="2024-01-01")
    assert all(r.chunk.date >= "2024-01-01" for r in res)
    assert "F0002" not in unique_doc_ids(res)


def test_get_document(retriever):
    assert "45 jours" in retriever.get_document_text("F0002")
    assert retriever.get_document_text("F9999") == ""
