"""Evaluation on the golden set.

    uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/

Writes into --out:
- results.jsonl: one line per question (answer, cited sources, retrieved documents,
                 tokens in/out, latency, tool calls, stop reason, ...)
- scores.jsonl : per-question metrics (hit@k, MRR, judge verdict, error category)
- metrics.json : aggregates per split (dev / test / all)
- report.md    : dashboard + error analysis

Useful options for experiments: --mode agent, --backend anthropic, --retrieval bm25|dense|hybrid,
--no-rerank, --top-k, --no-judge, --split test, --tag experiment_name.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import get_settings  # noqa: E402
from rag.evaluation import aggregate, judge_answer, load_jsonl, score_one, write_report  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="eval/golden.jsonl")
    p.add_argument("--out", default="eval/")
    p.add_argument("--mode", choices=["rag", "agent"], default="rag")
    p.add_argument("--backend", choices=["ollama", "anthropic", "extractive"])
    p.add_argument("--model", help="model name (Ollama or API) for this run")
    p.add_argument("--retrieval", choices=["bm25", "dense", "hybrid"])
    p.add_argument("--no-rerank", action="store_true")
    p.add_argument("--top-k", type=int)
    p.add_argument("--min-rerank-score", type=float)
    p.add_argument("--split", choices=["dev", "test", "all"], default="all")
    p.add_argument("--no-judge", action="store_true", help="no LLM judge (lexical scores only)")
    p.add_argument("--rescore", action="store_true", help="recompute the scores from an existing results.jsonl")
    p.add_argument("--tag", default="", help="output sub-directory to compare experiments")
    p.add_argument("--limit", type=int)
    args = p.parse_args(argv)

    overrides = dict(llm_backend=args.backend, retrieval_mode=args.retrieval, top_k=args.top_k,
                     min_rerank_score=args.min_rerank_score)
    if args.no_rerank:
        overrides["use_reranker"] = False
    settings = get_settings(**overrides)
    if args.model:
        if settings.llm_backend == "anthropic":
            settings.anthropic_model = args.model
        else:
            settings.ollama_model = args.model

    out = Path(args.out) / args.tag if args.tag else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    golden = load_jsonl(args.questions)
    if args.split != "all":
        golden = [q for q in golden if q.get("split", "test") == args.split]
    if args.limit:
        golden = golden[: args.limit]
    gold_by_id = {q["id"]: q for q in golden}

    results_path = out / "results.jsonl"
    if args.rescore:
        results = {r["id"]: r for r in load_jsonl(results_path) if r["id"] in gold_by_id}
    else:
        from rag.pipeline import Assistant

        assistant = Assistant.from_disk(settings)
        results = {}
        t_start = time.perf_counter()
        with open(results_path, "w", encoding="utf-8") as f:
            for i, q in enumerate(golden, 1):
                resp = assistant.ask(q["question"], mode=args.mode)
                row = {"id": q["id"], **resp.model_dump(), "tool_calls_count": len(resp.tool_calls)}
                results[q["id"]] = row
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{i}/{len(golden)}] {q['id']} {resp.stop_reason:20} {resp.latency_s:6.2f}s "
                      f"cit={resp.cited_sources} | {resp.answer[:80]!r}")
        print(f"Generation finished in {time.perf_counter() - t_start:.1f}s -> {results_path}")

    judge = None
    if not args.no_judge:
        from rag.llm import make_llm

        try:
            judge = make_llm(settings, backend=settings.judge_backend, model=settings.judge_model)
        except Exception as e:  # the judge is optional: carry on with lexical scores
            print(f"[warn] judge unavailable ({e}): lexical scores only")

    scores = []
    for q in golden:
        r = results.get(q["id"])
        if r is None:
            continue
        verdict = None
        needs_judge = q.get("category") in {"factual", "multi_doc"} or (
            q.get("category") == "injection" and r.get("answerable"))
        if judge is not None and needs_judge:
            try:
                verdict = judge_answer(judge, q["question"], q["answer"], r["answer"])
            except Exception as e:
                print(f"[warn] judge failed on {q['id']}: {e}")
        scores.append(score_one(q, r, settings.top_k, verdict))

    with open(out / "scores.jsonl", "w", encoding="utf-8") as f:
        for s in scores:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    res_list = [results[s["id"]] for s in scores]
    metrics = {"all": aggregate(scores, res_list, settings.top_k)}
    for split in ("dev", "test"):
        sub = [s for s in scores if s["split"] == split]
        if sub and len(sub) != len(scores):
            metrics[split] = aggregate(sub, [results[s["id"]] for s in sub], settings.top_k)
    config = settings.to_dict() | {"mode": args.mode, "model": settings.generation_model,
                                   "judge": None if judge is None else settings.judge_model}
    (out / "metrics.json").write_text(json.dumps({"config": config, "metrics": metrics}, ensure_ascii=False,
                                                 indent=2), encoding="utf-8")
    write_report(out / "report.md", metrics, scores, gold_by_id, results, config)
    m = metrics.get("test", metrics["all"])
    print(json.dumps({k: m[k] for k in ("retrieval", "answers", "abstention", "safety", "cost_latency")},
                     ensure_ascii=False, indent=2))
    print(f"-> {out / 'metrics.json'}, {out / 'report.md'}")


if __name__ == "__main__":
    main()
