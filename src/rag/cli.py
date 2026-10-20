"""Ligne de commande.

  uv run python -m rag.cli download                      # télécharge l'archive XML officielle
  uv run python -m rag.cli build-corpus --source data/raw/vosdroits-latest.zip
  uv run python -m rag.cli stats                         # chiffres pour la data card + échantillons à lire
  uv run python -m rag.cli index                         # découpage + embeddings
  uv run python -m rag.cli ask "Quel est le délai pour contester une amende ?" [--mode agent]
  uv run python -m rag.cli suggest-gold                  # aide à annoter gold_docs du golden set
  uv run python -m rag.cli gen-queries --n-chunks 2500   # questions synthétiques pour l'affinage
  uv run --extra train python -m rag.cli finetune        # affine les embeddings -> models/e5-small-servicepublic
  uv run --extra train python -m rag.cli finetune-reranker   # affine le reranker (négatifs difficiles)
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


def cmd_gen_queries(args):
    from .finetune import generate_queries, select_chunks
    from .llm import OllamaClient
    from .retrieval import load_chunks

    s = get_settings()
    # Contexte court : un passage + la consigne tiennent en ~600 tokens ; moins de VRAM par requête parallèle
    llm = OllamaClient(args.model or s.ollama_model, s.ollama_host, temperature=0.7, num_ctx=2048,
                       max_output_tokens=200)
    chunks = select_chunks(load_chunks(CHUNKS_PATH), args.n_chunks, seed=args.seed)
    print(f"{len(chunks)} passages tirés · modèle {llm.model} · {args.workers} requêtes en parallèle · "
          f"sortie {args.out}")
    n = generate_queries(chunks, llm, Path(args.out), per_chunk=args.per_chunk, workers=args.workers)
    print(f"{n} questions ajoutées -> {args.out}")


def cmd_finetune(args):
    from .finetune import drop_golden_overlap, finetune_embedder
    from .retrieval import load_chunks

    s = get_settings()
    with open(args.queries, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    golden = [json.loads(line)["question"] for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    kept = drop_golden_overlap(pairs, golden)
    print(f"{len(pairs)} paires, {len(pairs) - len(kept)} retirées car trop proches du golden set")
    info = finetune_embedder(kept, load_chunks(CHUNKS_PATH), args.base_model or s.embedding_model, Path(args.out),
                             epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    info["n_dropped_golden_overlap"] = len(pairs) - len(kept)
    (Path(args.out) / "train_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    b, a = info["val_before"], info["val_after"]
    for key in ("val_cosine_accuracy@1", "val_cosine_accuracy@5", "val_cosine_mrr@10"):
        print(f"{key:24} {b.get(key, 0):.3f} -> {a.get(key, 0):.3f}")
    print(f"Modèle -> {args.out}\nÉtape suivante : $env:EMBEDDING_MODEL=\"{args.out}\" puis `rag.cli index`")


def cmd_finetune_reranker(args):
    from .finetune import drop_golden_overlap, finetune_reranker, mine_hard_negatives
    from .retrieval import HybridRetriever, load_chunks

    s = get_settings()
    mined_path = Path(args.mined)
    if mined_path.exists():
        with open(mined_path, encoding="utf-8") as f:
            mined = [json.loads(line) for line in f if line.strip()]
        print(f"{len(mined)} questions avec négatifs déjà extraits ({mined_path})")
    else:
        with open(args.queries, encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f if line.strip()]
        golden = [json.loads(line)["question"] for line in GOLDEN.read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        kept = drop_golden_overlap(pairs, golden)
        print(f"{len(pairs)} paires, {len(pairs) - len(kept)} retirées car trop proches du golden set")
        # Négatifs difficiles = candidats de la recherche hybride telle qu'elle tourne en production (sans reranker)
        retriever = HybridRetriever.load(get_settings(use_reranker=False))
        mined = mine_hard_negatives(kept, retriever, n_candidates=s.candidates_k)
        mined_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mined_path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(m, ensure_ascii=False) + "\n" for m in mined)
        print(f"Négatifs difficiles -> {mined_path}")
    info = finetune_reranker(mined, load_chunks(CHUNKS_PATH), args.base_model or s.reranker_model, Path(args.out),
                             n_negatives=args.negatives, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    b, a = info["val_before"], info["val_after"]
    for key in ("val_base_mrr@10", "val_mrr@10", "val_ndcg@10", "val_map"):
        if key in a:
            print(f"{key:20} {b.get(key, 0):.3f} -> {a.get(key, 0):.3f}")
    print(f"Modèle -> {args.out}\nÉtape suivante : $env:RERANKER_MODEL=\"{args.out}\" (et recalibrer MIN_RERANK_SCORE)")


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

    sp = sub.add_parser("gen-queries")
    sp.add_argument("--n-chunks", type=int, default=2500)
    sp.add_argument("--per-chunk", type=int, default=2)
    sp.add_argument("--model", help="modèle Ollama (défaut : OLLAMA_MODEL)")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--workers", type=int, default=1, help="requêtes parallèles (régler OLLAMA_NUM_PARALLEL)")
    sp.add_argument("--out", default=str(ROOT / "data" / "train" / "synthetic_queries.jsonl"))
    sp.set_defaults(func=cmd_gen_queries)

    sp = sub.add_parser("finetune")
    sp.add_argument("--queries", default=str(ROOT / "data" / "train" / "synthetic_queries.jsonl"))
    sp.add_argument("--base-model", help="défaut : EMBEDDING_MODEL")
    sp.add_argument("--out", default="models/e5-small-servicepublic")
    sp.add_argument("--epochs", type=int, default=1)
    sp.add_argument("--batch-size", type=int, default=32)
    sp.add_argument("--lr", type=float, default=2e-5)
    sp.set_defaults(func=cmd_finetune)

    sp = sub.add_parser("finetune-reranker")
    sp.add_argument("--queries", default=str(ROOT / "data" / "train" / "synthetic_queries.jsonl"))
    sp.add_argument("--mined", default=str(ROOT / "data" / "train" / "hard_negatives.jsonl"),
                    help="cache des négatifs extraits (recalculés s'il est absent)")
    sp.add_argument("--base-model", help="défaut : RERANKER_MODEL")
    sp.add_argument("--out", default="models/mmarco-reranker-servicepublic")
    sp.add_argument("--negatives", type=int, default=4)
    sp.add_argument("--epochs", type=int, default=1)
    sp.add_argument("--batch-size", type=int, default=32)
    sp.add_argument("--lr", type=float, default=2e-5)
    sp.set_defaults(func=cmd_finetune_reranker)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
