"""Hybrid retrieval: BM25 + dense, RRF fusion, cross-encoder reranking, metadata filters."""
from __future__ import annotations

import json
import time
import zlib
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from .config import CHUNKS_PATH, INDEX_DIR, ROOT, Settings, _env_int
from .schemas import Chunk, RetrievedChunk
from .text import tokenize


# --------------------------------------------------------------------------- embedders
class SentenceTransformerEmbedder:
    """Dense embeddings (sentence-transformers). Handles the E5 model prefixes."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        # Local fine-tuned model (e.g. models/e5-small-servicepublic): path relative to the repo root
        local = ROOT / model_name
        self.model = SentenceTransformer(str(local) if local.is_dir() else model_name)
        self.is_e5 = "e5" in model_name.lower()

    def encode(self, texts: list[str], kind: str = "passage", batch_size: int | None = None) -> np.ndarray:
        # Smaller EMBED_BATCH_SIZE = less RAM during indexing (but slower)
        batch_size = batch_size or _env_int("EMBED_BATCH_SIZE", 32)
        if self.is_e5:
            texts = [f"{'query' if kind == 'query' else 'passage'}: {t}" for t in texts]
        emb = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                show_progress_bar=len(texts) > 100)
        return np.asarray(emb, dtype=np.float32)


class HashingEmbedder:
    """Dependency-free embedder (hashed bag of words). Used by tests and offline mode."""

    model_name = "hashing-512"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts: list[str], kind: str = "passage", batch_size: int = 32) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                out[i, zlib.crc32(tok.encode()) % self.dim] += 1.0  # hash that is stable across processes
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


def make_embedder(model_name: str):
    return HashingEmbedder() if model_name.startswith("hashing") else SentenceTransformerEmbedder(model_name)


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        local = ROOT / model_name  # local fine-tuned reranker (e.g. models/mmarco-reranker-servicepublic)
        self.model = CrossEncoder(str(local) if local.is_dir() else model_name, max_length=512)

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, t) for t in texts]
        return [float(s) for s in self.model.predict(pairs, batch_size=_env_int("RERANK_BATCH_SIZE", 32))]


# --------------------------------------------------------------------------- index
def build_index(chunks: list[Chunk], embedder, index_dir: Path = INDEX_DIR) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    emb = embedder.encode([c.indexed_text for c in chunks], kind="passage")
    np.save(index_dir / "embeddings.npy", emb)
    (index_dir / "meta.json").write_text(json.dumps({
        "embedding_model": embedder.model_name,
        "n_chunks": len(chunks),
        "dim": int(emb.shape[1]),
        "chunk_ids": [c.chunk_id for c in chunks],
    }, ensure_ascii=False), encoding="utf-8")


def load_chunks(path: Path = CHUNKS_PATH) -> list[Chunk]:
    with open(path, encoding="utf-8") as f:
        return [Chunk.model_validate_json(line) for line in f if line.strip()]


def save_chunks(chunks: list[Chunk], path: Path = CHUNKS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = Σ 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray | None, embedder, settings: Settings,
                 reranker=None):
        self.chunks = chunks
        self.settings = settings
        self.embedder = embedder
        self.embeddings = embeddings
        self.reranker = reranker
        self.bm25 = BM25Okapi([tokenize(c.indexed_text) for c in chunks])
        self.last_timings: dict[str, float] = {}

    @classmethod
    def load(cls, settings: Settings, index_dir: Path = INDEX_DIR, chunks_path: Path = CHUNKS_PATH):
        chunks = load_chunks(chunks_path)
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        if meta["chunk_ids"] != [c.chunk_id for c in chunks]:
            raise RuntimeError("Index out of sync with the chunks: rerun `python -m rag.cli index`.")
        if meta["embedding_model"] != settings.embedding_model:
            print(f"[warn] index built with {meta['embedding_model']}, using that model.")
        embedder = make_embedder(meta["embedding_model"])
        embeddings = np.load(index_dir / "embeddings.npy")
        reranker = CrossEncoderReranker(settings.reranker_model) if settings.use_reranker else None
        return cls(chunks, embeddings, embedder, settings, reranker)

    # -- components
    def _allowed(self, since: str | None, theme: str | None) -> np.ndarray:
        mask = np.ones(len(self.chunks), dtype=bool)
        if since:
            mask &= np.array([bool(c.date) and c.date >= since for c in self.chunks])
        if theme:
            t = theme.lower()
            mask &= np.array([t in c.theme.lower() for c in self.chunks])
        return mask

    def bm25_search(self, query: str, n: int, mask: np.ndarray) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:n]
        return [int(i) for i in order if np.isfinite(scores[i]) and scores[i] > 0]

    def dense_search(self, query: str, n: int, mask: np.ndarray) -> list[int]:
        q = self.embedder.encode([query], kind="query")[0]
        scores = self.embeddings @ q
        scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:n]
        return [int(i) for i in order if np.isfinite(scores[i])]

    # -- main API
    def search(self, query: str, k: int | None = None, since: str | None = None,
               theme: str | None = None, mode: str | None = None) -> list[RetrievedChunk]:
        k = k or self.settings.top_k
        mode = mode or self.settings.retrieval_mode
        n = self.settings.candidates_k
        mask = self._allowed(since, theme)
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        bm25_ranking = self.bm25_search(query, n, mask) if mode in {"bm25", "hybrid"} else []
        timings["bm25_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        dense_ranking = self.dense_search(query, n, mask) if mode in {"dense", "hybrid"} else []
        timings["dense_s"] = time.perf_counter() - t0

        fused = rrf_fuse([r for r in (bm25_ranking, dense_ranking) if r], k=self.settings.rrf_k)
        candidates = sorted(fused, key=lambda i: -fused[i])[:n]
        bm25_pos = {idx: r for r, idx in enumerate(bm25_ranking, 1)}
        dense_pos = {idx: r for r, idx in enumerate(dense_ranking, 1)}

        results = [RetrievedChunk(chunk=self.chunks[i], score=fused[i], bm25_rank=bm25_pos.get(i),
                                  dense_rank=dense_pos.get(i)) for i in candidates]
        t0 = time.perf_counter()
        if self.reranker is not None and results:
            rs = self.reranker.score(query, [r.chunk.indexed_text for r in results])
            for r, s in zip(results, rs):
                r.rerank_score = s
                r.score = s
            results.sort(key=lambda r: -r.score)
        timings["rerank_s"] = time.perf_counter() - t0
        self.last_timings = timings
        return results[:k]

    def get_document_text(self, doc_id: str, max_chars: int = 6000) -> str:
        parts = [c for c in self.chunks if c.doc_id == doc_id]
        if not parts:
            return ""
        text = f"[{doc_id}] {parts[0].title}\n" + "\n".join(
            (f"## {c.section}\n" if c.section else "") + c.text for c in parts)
        return text[:max_chars]


def unique_doc_ids(results: list[RetrievedChunk]) -> list[str]:
    seen: list[str] = []
    for r in results:
        if r.chunk.doc_id not in seen:
            seen.append(r.chunk.doc_id)
    return seen
