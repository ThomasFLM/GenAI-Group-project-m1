import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def docs():
    from rag.ingestion import load_fiches

    return load_fiches(FIXTURES)


@pytest.fixture
def retriever(docs):
    from rag.chunking import chunk_documents
    from rag.config import get_settings
    from rag.retrieval import HashingEmbedder, HybridRetriever

    chunks = chunk_documents(docs, 60, 10)
    emb = HashingEmbedder()
    settings = get_settings(use_reranker=False, top_k=3, candidates_k=10)
    return HybridRetriever(chunks, emb.encode([c.indexed_text for c in chunks]), emb, settings)
