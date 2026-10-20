"""Affinage (fine-tuning) du modèle d'embeddings sur le corpus.

1. `generate_queries` : le LLM local écrit, pour des passages tirés au hasard, des questions d'usager
   auxquelles le passage répond -> paires (question, passage) synthétiques.
2. `finetune_embedder` : affine le modèle E5 sur ces paires avec une perte contrastive
   (MultipleNegativesRankingLoss : les autres passages du lot servent d'exemples négatifs).

Le golden set n'est jamais utilisé pour l'entraînement : les questions synthétiques trop proches d'une
question du golden set sont retirées (fuite de données), et la validation se fait sur des fiches tenues
à l'écart de l'entraînement.
"""
from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

from .llm import parse_json_object
from .schemas import Chunk
from .text import tokenize

QUERY_GEN_SYSTEM = (
    "Tu aides à construire un jeu d'entraînement pour un moteur de recherche sur les fiches Service-Public.fr. "
    "On te donne un passage d'une fiche. Écris des questions qu'un usager poserait et auxquelles ce passage "
    "répond. Règles : questions courtes et naturelles, en français, comme tapées par un particulier ; "
    "reformule avec tes propres mots plutôt que de recopier le passage ; une question doit pouvoir se comprendre "
    "seule (nomme la démarche ou le document concerné) ; ne mentionne jamais « le passage », « le texte » ou "
    "« la fiche »."
)

# Questions qui renvoient au passage au lieu de se suffire à elles-mêmes (« selon la fiche… ») : écartées
META_REFERENCE = re.compile(r"\b(selon|d'après|dans) (la fiche|le passage|le texte|ce document|cette fiche)", re.I)

QUERY_JSON_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["questions"],
}


def select_chunks(chunks: list[Chunk], n: int, min_words: int = 40, seed: int = 0) -> list[Chunk]:
    """Tire `n` passages assez longs pour porter une réponse (ordre déterministe)."""
    pool = [c for c in chunks if len(c.text.split()) >= min_words]
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))


def _ask_questions(llm, c: Chunk, per_chunk: int) -> list[str]:
    prompt = (f"Fiche : {c.title}\nSection : {c.section or '—'}\n\nPassage :\n{c.text}\n\n"
              f"Écris exactement {per_chunk} questions différentes.")
    r = llm.chat(QUERY_GEN_SYSTEM, [{"role": "user", "content": prompt}], json_schema=QUERY_JSON_SCHEMA)
    return parse_json_object(r.text).get("questions", [])


def generate_queries(chunks: list[Chunk], llm, out_path: Path, per_chunk: int = 2, workers: int = 1) -> int:
    """Écrit une ligne JSON par question générée. Reprend là où un lancement précédent s'est arrêté.

    `workers` > 1 envoie des requêtes en parallèle (utile si Ollama tourne avec OLLAMA_NUM_PARALLEL > 1)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    done: set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["chunk_id"] for line in f if line.strip()}
    todo = [c for c in chunks if c.chunk_id not in done]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written, t_start = 0, time.perf_counter()
    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_ask_questions, llm, c, per_chunk): c for c in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            c = futures[fut]
            try:
                questions = fut.result()
            except Exception as e:  # une erreur ponctuelle ne doit pas arrêter une génération de plusieurs heures
                print(f"[warn] {c.chunk_id} : {e}")
                continue
            for q in [q.strip() for q in questions if isinstance(q, str)][:per_chunk]:
                if len(q.split()) >= 3 and not META_REFERENCE.search(q):
                    f.write(json.dumps({"chunk_id": c.chunk_id, "doc_id": c.doc_id, "question": q},
                                       ensure_ascii=False) + "\n")
                    n_written += 1
            f.flush()
            if i % 25 == 0 or i == len(todo):
                elapsed = time.perf_counter() - t_start
                eta = elapsed / i * (len(todo) - i)
                print(f"{i}/{len(todo)} passages · {n_written} questions · reste ~{eta / 60:.0f} min", flush=True)
    return n_written


def jaccard(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def drop_golden_overlap(pairs: list[dict], golden_questions: list[str], threshold: float = 0.5) -> list[dict]:
    """Retire les questions synthétiques trop proches d'une question du golden set."""
    return [p for p in pairs if all(jaccard(p["question"], g) < threshold for g in golden_questions)]


def split_by_doc(pairs: list[dict], val_frac: float = 0.1, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Validation sur des fiches entières tenues à l'écart (pas seulement d'autres passages des mêmes fiches)."""
    docs = sorted({p["doc_id"] for p in pairs})
    rng = random.Random(seed)
    val_docs = set(rng.sample(docs, max(1, int(len(docs) * val_frac))))
    return [p for p in pairs if p["doc_id"] not in val_docs], [p for p in pairs if p["doc_id"] in val_docs]


def mine_hard_negatives(pairs: list[dict], retriever, n_candidates: int = 30, seed: int = 0) -> list[dict]:
    """Ajoute à chaque paire les candidats de la recherche hybride (sans reranker), dans l'ordre.

    `negatives` = candidats d'autres fiches (ceux d'une même fiche répondent souvent aussi à la question :
    les traiter comme négatifs apprendrait au reranker des erreurs). `candidates` = liste complète, pour
    évaluer le reranking dans les conditions réelles."""
    out, t_start = [], time.perf_counter()
    for i, p in enumerate(pairs, 1):
        res = retriever.search(p["question"], k=n_candidates)
        ids = [r.chunk.chunk_id for r in res]
        negs = [r.chunk.chunk_id for r in res if r.chunk.doc_id != p["doc_id"]]
        out.append({**p, "candidates": ids, "negatives": negs})
        if i % 500 == 0 or i == len(pairs):
            print(f"négatifs : {i}/{len(pairs)} questions ({time.perf_counter() - t_start:.0f} s)", flush=True)
    return out


def finetune_reranker(mined: list[dict], chunks: list[Chunk], base_model: str, out_dir: Path,
                      n_negatives: int = 4, epochs: int = 1, batch_size: int = 32, lr: float = 2e-5,
                      max_length: int = 512, val_frac: float = 0.1, seed: int = 0) -> dict:
    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import (CrossEncoder, CrossEncoderTrainer,
                                                     CrossEncoderTrainingArguments)
    from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    by_id = {c.chunk_id: c for c in chunks}
    text = {cid: c.indexed_text for cid, c in by_id.items()}  # même texte que le reranker voit en production
    mined = [p for p in mined if p["chunk_id"] in by_id]
    train, val = split_by_doc(mined, val_frac, seed)  # mêmes fiches de validation que pour les embeddings

    # Négatifs difficiles pris parmi les mieux classés, avec un peu de hasard pour ne pas toujours voir les mêmes
    rng = random.Random(seed)
    queries, passages, labels = [], [], []
    for p in train:
        queries.append(p["question"]); passages.append(text[p["chunk_id"]]); labels.append(1.0)
        top = p["negatives"][:2 * n_negatives]
        for nid in rng.sample(top, min(n_negatives, len(top))):
            queries.append(p["question"]); passages.append(text[nid]); labels.append(0.0)
    train_ds = Dataset.from_dict({"query": queries, "passage": passages, "label": labels})

    # Validation réaliste : reclasser les 30 candidats hybrides (le positif est ajouté s'il manque)
    samples = [{"query": p["question"], "positive": [text[p["chunk_id"]]],
                "documents": [text[c] for c in p["candidates"]]} for p in val]
    evaluator = CrossEncoderRerankingEvaluator(samples=samples, at_k=10, name="val", batch_size=64,
                                               write_csv=False)

    model = CrossEncoder(base_model, max_length=max_length)
    before = evaluator(model)
    args = CrossEncoderTrainingArguments(
        output_dir=str(out_dir / "_checkpoints"), num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        learning_rate=lr, warmup_ratio=0.1, save_strategy="no", eval_strategy="no", logging_steps=50,
        report_to="none", seed=seed, dataloader_drop_last=True, fp16=torch.cuda.is_available(),
    )
    loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(float(n_negatives)))
    t0 = time.perf_counter()
    CrossEncoderTrainer(model=model, args=args, train_dataset=train_ds, loss=loss).train()
    train_s = time.perf_counter() - t0
    after = evaluator(model)

    model.save(str(out_dir))
    info = {
        "base_model": base_model, "n_queries_train": len(train), "n_pairs_train": len(train_ds),
        "n_queries_val": len(val), "n_negatives": n_negatives, "epochs": epochs, "batch_size": batch_size,
        "lr": lr, "max_length": max_length, "seed": seed,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "train_seconds": round(train_s), "val_before": before, "val_after": after,
    }
    (out_dir / "train_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def finetune_embedder(pairs: list[dict], chunks: list[Chunk], base_model: str, out_dir: Path,
                      epochs: int = 1, batch_size: int = 32, lr: float = 2e-5, max_seq_length: int = 256,
                      val_frac: float = 0.1, n_distractors: int = 1500, seed: int = 0) -> dict:
    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    by_id = {c.chunk_id: c for c in chunks}
    pairs = [p for p in pairs if p["chunk_id"] in by_id]
    train, val = split_by_doc(pairs, val_frac, seed)

    # Préfixes E5 ajoutés à la main, comme SentenceTransformerEmbedder le fait à l'inférence
    train_ds = Dataset.from_dict({
        "anchor": [f"query: {p['question']}" for p in train],
        "positive": [f"passage: {by_id[p['chunk_id']].indexed_text}" for p in train],
    })

    # Validation : questions des fiches tenues à l'écart, cherchées parmi leurs passages + des distracteurs
    val_docs = {p["doc_id"] for p in val}
    rng = random.Random(seed)
    others = [c for c in chunks if c.doc_id not in val_docs]
    corpus_chunks = [c for c in chunks if c.doc_id in val_docs] + rng.sample(others, min(n_distractors, len(others)))
    evaluator = InformationRetrievalEvaluator(
        queries={str(i): f"query: {p['question']}" for i, p in enumerate(val)},
        corpus={c.chunk_id: f"passage: {c.indexed_text}" for c in corpus_chunks},
        relevant_docs={str(i): {p["chunk_id"]} for i, p in enumerate(val)},
        mrr_at_k=[10], ndcg_at_k=[10], accuracy_at_k=[1, 5, 10], precision_recall_at_k=[5],
        map_at_k=[10], show_progress_bar=False, name="val",
    )

    model = SentenceTransformer(base_model)
    model.max_seq_length = max_seq_length
    before = evaluator(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(out_dir / "_checkpoints"), num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        learning_rate=lr, warmup_ratio=0.1, batch_sampler=BatchSamplers.NO_DUPLICATES, save_strategy="no",
        eval_strategy="no", logging_steps=10, report_to="none", seed=seed, dataloader_drop_last=True,
        fp16=torch.cuda.is_available(),  # GPU : demi-précision, plus rapide et moins de VRAM
    )
    t0 = time.perf_counter()
    SentenceTransformerTrainer(model=model, args=args, train_dataset=train_ds,
                               loss=MultipleNegativesRankingLoss(model)).train()
    train_s = time.perf_counter() - t0
    after = evaluator(model)

    model.save(str(out_dir))
    info = {
        "base_model": base_model, "n_pairs_train": len(train), "n_pairs_val": len(val),
        "n_docs_val": len(val_docs), "val_corpus_size": len(corpus_chunks),
        "epochs": epochs, "batch_size": batch_size, "lr": lr, "max_seq_length": max_seq_length, "seed": seed,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "train_seconds": round(train_s), "val_before": before, "val_after": after,
    }
    (out_dir / "train_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info
