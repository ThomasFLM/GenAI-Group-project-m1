"""Ligne de commande.

  uv run python -m rag.cli download                      # télécharge l'archive XML officielle
  uv run python -m rag.cli build-corpus --source data/raw/vosdroits-latest.zip
  uv run python -m rag.cli stats                         # chiffres pour la data card + échantillons à lire
  uv run python -m rag.cli index                         # découpage + embeddings
  uv run python -m rag.cli ask "Quel est le délai pour contester une amende ?" [--mode agent]
  uv run python -m rag.cli suggest-gold                  # aide à annoter gold_docs du golden set
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

from .chunking import chunk_documents
from .config import CHUNKS_PATH, DOCS_PATH, INDEX_DIR, RAW_DIR, ROOT, get_settings
from .ingestion import (download, golden_doc_ids, load_documents, load_fiches, load_parquet, save_documents,
                        select_documents)

GOLDEN = ROOT / "eval" / "golden.jsonl"


def cmd_download(args):
    s = get_settings()
    dest = RAW_DIR / Path(s.corpus_url).name
    print(f"Téléchargement de {s.corpus_url} -> {dest}")
    download(s.corpus_url, dest)
    print("OK")


def cmd_build_corpus(args):
    s = get_settings()
    src = Path(args.source)
    docs = load_parquet(src) if src.suffix == ".parquet" else load_fiches(src)
    print(f"{len(docs)} fiches lues dans {src}")
    themes = args.themes or s.themes
    selected = select_documents(docs, themes, args.max_docs or s.max_docs, golden_doc_ids(GOLDEN))
    missing = golden_doc_ids(GOLDEN) - {d.doc_id for d in selected}
    if missing:
        print(f"[warn] documents du golden set absents de la source : {sorted(missing)}")
    save_documents(selected, DOCS_PATH)
    print(f"{len(selected)} fiches retenues -> {DOCS_PATH}")
    print(Counter(d.theme for d in selected).most_common())


def cmd_stats(args):
    docs = load_documents(DOCS_PATH)
    s = get_settings()
    chunks = chunk_documents(docs, s.chunk_max_words, s.chunk_overlap_words)
    words = [len(d.text.split()) for d in docs]
    cw = [len(c.text.split()) for c in chunks]
    dates = sorted(d.date for d in docs if d.date)
    stats = {
        "n_documents": len(docs),
        "n_chunks": len(chunks),
        "themes": Counter(d.theme for d in docs).most_common(),
        "doc_words": {"mean": round(statistics.mean(words)), "median": statistics.median(words),
                      "min": min(words), "max": max(words), "total": sum(words)},
        "chunk_words": {"mean": round(statistics.mean(cw)), "median": statistics.median(cw),
                        "min": min(cw), "max": max(cw)},
        "dates": {"min": dates[0] if dates else "", "max": dates[-1] if dates else "",
                  "since_2024": sum(d >= "2024" for d in dates)},
        "empty_docs": sum(w < 30 for w in words),
    }
    out = ROOT / "docs" / "corpus_stats.json"
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    rng = random.Random(0)
    sample_path = ROOT / "docs" / "samples_to_read.md"
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write("# 20 documents et 20 passages tirés au hasard (seed 0)\n\n## Documents\n\n")
        for d in rng.sample(docs, min(20, len(docs))):
            f.write(f"### {d.doc_id} — {d.title} ({d.theme}, {d.date})\n\n{d.text[:1500]}\n\n")
        f.write("## Passages\n\n")
        for c in rng.sample(chunks, min(20, len(chunks))):
            f.write(f"### {c.chunk_id} — {c.title} / {c.section}\n\n{c.text}\n\n")
    print(f"\nStatistiques -> {out}\nÉchantillons à lire -> {sample_path}")


def cmd_index(args):
    from .retrieval import build_index, make_embedder, save_chunks

    s = get_settings(embedding_model=args.embedding_model)
    docs = load_documents(DOCS_PATH)
    chunks = chunk_documents(docs, args.max_words or s.chunk_max_words,
                             args.overlap if args.overlap is not None else s.chunk_overlap_words)
    save_chunks(chunks, CHUNKS_PATH)
    print(f"{len(chunks)} passages -> {CHUNKS_PATH}")
    build_index(chunks, make_embedder(s.embedding_model), INDEX_DIR)
    print(f"Index ({s.embedding_model}) -> {INDEX_DIR}")


def cmd_ask(args):
    from .pipeline import Assistant

    s = get_settings(llm_backend=args.backend)
    assistant = Assistant.from_disk(s)
    resp = assistant.ask(args.question, mode=args.mode, since=args.since)
    print(f"\n{resp.answer}\n")
    print(f"Sources : {', '.join(resp.cited_sources) or '—'}")
    if args.verbose:
        print(json.dumps(resp.model_dump(exclude={'question', 'answer'}), ensure_ascii=False, indent=2))
    else:
        print(f"[{resp.stop_reason}] {resp.latency_s:.2f}s · tokens {resp.tokens_in}/{resp.tokens_out} · "
              f"outils {[t.name for t in resp.tool_calls]} · récupérés {resp.retrieved_doc_ids}")


def cmd_suggest_gold(args):
    """Propose, pour chaque question sans gold_docs, les fiches les plus probables (à valider à la main)."""
    from .retrieval import HybridRetriever

    s = get_settings(use_reranker=False)
    retriever = HybridRetriever.load(s)
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        if q.get("category") not in {"factual", "multi_doc"} or (q.get("gold_docs") and not args.all):
            continue
        res = retriever.search(q["question"], k=8)
        print(f"\n{q['id']} | {q['question']}\n   réponse attendue : {q['answer']}")
        seen = set()
        for r in res:
            if r.chunk.doc_id in seen:
                continue
            seen.add(r.chunk.doc_id)
            print(f"   {r.chunk.doc_id:8} {r.chunk.title[:70]}  [{r.chunk.section[:40]}]")


def main(argv=None):
    p = argparse.ArgumentParser(prog="rag")
    sub = p.add_subparsers(required=True)

    sp = sub.add_parser("download")
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("build-corpus")
    sp.add_argument("--source", default=str(RAW_DIR / "vosdroits-latest.zip"))
    sp.add_argument("--themes", nargs="*")
    sp.add_argument("--max-docs", type=int)
    sp.set_defaults(func=cmd_build_corpus)

    sp = sub.add_parser("stats")
    sp.set_defaults(func=cmd_stats)

    sp = sub.add_parser("index")
    sp.add_argument("--embedding-model")
    sp.add_argument("--max-words", type=int)
    sp.add_argument("--overlap", type=int)
    sp.set_defaults(func=cmd_index)

    sp = sub.add_parser("ask")
    sp.add_argument("question")
    sp.add_argument("--mode", choices=["rag", "agent"], default="rag")
    sp.add_argument("--backend", choices=["ollama", "anthropic", "extractive"])
    sp.add_argument("--since", help="filtre de date YYYY-MM-DD")
    sp.add_argument("-v", "--verbose", action="store_true")
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("suggest-gold")
    sp.add_argument("--all", action="store_true", help="inclure les questions déjà annotées")
    sp.set_defaults(func=cmd_suggest_gold)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
