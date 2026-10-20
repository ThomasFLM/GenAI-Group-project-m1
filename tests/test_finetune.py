import json

from rag.finetune import drop_golden_overlap, generate_queries, mine_hard_negatives, select_chunks, split_by_doc
from rag.llm import LLMResult


class FakeLLM:
    model = "fake"

    def __init__(self):
        self.calls = 0

    def chat(self, system, messages, json_schema=None, tools=None):
        self.calls += 1
        return LLMResult(text=json.dumps({"questions": ["Quel est le délai pour agir ?", "Combien ça coûte ?", "x"]}),
                         model=self.model)


def test_generate_queries_and_resume(retriever, tmp_path):
    chunks = select_chunks(retriever.chunks, 3, min_words=5)
    out = tmp_path / "q.jsonl"
    llm = FakeLLM()
    assert generate_queries(chunks, llm, out, per_chunk=2, workers=2) == 6
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {r["chunk_id"] for r in rows} == {c.chunk_id for c in chunks}
    # relance : les passages déjà traités ne sont pas redemandés
    assert generate_queries(chunks, llm, out, per_chunk=2) == 0
    assert llm.calls == 3


def test_mine_hard_negatives_excludes_same_document(retriever):
    pairs = [{"chunk_id": retriever.chunks[0].chunk_id, "doc_id": "F0002",
              "question": "délai pour contester une amende forfaitaire"}]
    [m] = mine_hard_negatives(pairs, retriever, n_candidates=10)
    by_id = {c.chunk_id: c for c in retriever.chunks}
    assert m["candidates"] and m["negatives"]
    assert all(by_id[n].doc_id != "F0002" for n in m["negatives"])
    assert [c for c in m["candidates"] if by_id[c].doc_id != "F0002"] == m["negatives"]  # ordre conservé


def test_drop_golden_overlap():
    pairs = [{"question": "Quel est le délai pour contester une amende forfaitaire ?"},
             {"question": "Combien coûte un passeport pour un adulte ?"}]
    kept = drop_golden_overlap(pairs, ["Quel est le délai pour contester une amende forfaitaire ?"])
    assert [p["question"] for p in kept] == ["Combien coûte un passeport pour un adulte ?"]


def test_split_by_doc_keeps_documents_apart():
    pairs = [{"doc_id": f"F{i % 10}", "chunk_id": f"F{i % 10}#{i}", "question": "q"} for i in range(50)]
    train, val = split_by_doc(pairs, val_frac=0.2)
    assert val and train and len(train) + len(val) == 50
    assert not {p["doc_id"] for p in train} & {p["doc_id"] for p in val}
