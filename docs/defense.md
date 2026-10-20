# Defence — 21 October 2026 (10 min + 5 min Q&A)

Everyone presents one part (≈ 2 min 30 each for 4 people).

| # | Part | Time | Content | Material |
|---|---|---|---|---|
| 1 | Data | 2 min 30 | Problem (administrative procedures assistant), choice of the Service-Public corpus, selection of the ~500 fact sheets, cleaning, statistics, what we saw while reading 20 fact sheets | data card, `corpus_stats.json` |
| 2 | Architecture | 2 min 30 | Chapter-based chunking + contextual header, hybrid BM25 + dense + RRF, reranker, validated JSON output, agent and tools, guardrails | README diagram, architecture.md |
| 3 | Demo | 2 min 30 | 1 simple question with a citation · 1 date question (agent + `add_days`) · 1 out-of-corpus question ("je ne sais pas") · 1 blocked injection — show the trace (tokens, latency, tools) | Streamlit **+ recorded backup video** |
| 4 | Evaluation | 2 min 30 | Golden set, experiment table (E0→E9), quality / cost / latency trade-offs, error analysis (2 concrete examples), residual risks, integrity of the `test` split | report.md, evaluation.md |

Key figures (E4, final configuration, `test` split): hit@5 1.00 · MRR 0.87 · judged accuracy 0.74 ·
out-of-corpus abstention 100 % · injections blocked 100 % · p50 4.8 s · ~1,650 input tokens · $0.

## Likely jury questions

- Why these chunk sizes? → E8: 120 words = MRR 0.81, 220 = 0.86, 400 = 0.89 for retrieval alone;
  400 words sends 1.8 times more text to the LLM and mixes more cases, not tested end to end → we keep 220.
- Is the reranker worth its latency? → E3 vs E4: test hit@1 0.68 → 0.79, hit@5 0.90 → 1.00, for 0.2 to 0.8 s
  more (p50) and a shorter context (2,052 → 1,649 tokens).
- Why not the agent by default? → E5: better retrieval (MRR 0.95) but almost 4 times slower,
  almost 3 times more tokens, and no more accurate on this golden set.
- Did you train anything? → E9: fine-tuning of the embeddings and the reranker on 7,369 synthetic
  questions; better ranking (MRR 0.86 → 0.95) but no better answers → not kept.
- How do you know the judge is reliable? → agreement with human annotation on 10 answers
  ([judge_check.md](judge_check.md)); and the judge moved from 3b to 7b, because the 3b changed 2-3 verdicts from one run to the next.
- What happens if an amount changes? → "outdated" risk, date shown, re-download.
- Is your injection resistance overestimated? → yes, the patterns were written while looking at the questions.
- Did you tune anything on `test`? → the final configuration was chosen on `dev` with a rule fixed in
  advance; but `test` was looked at during the week and 3 `gold_docs` were corrected → somewhat optimistic figures.

## Checklist before 20 October 23:59

- [x] `gold_docs` annotated and reference answers checked in the fact sheets (q005, q014, q015 completed, PR #1)
- [x] runs E0–E6, E8, E9 done, results table filled in (E7 not run: no API key)
- [ ] human agreement of the judge on 10 answers ([judge_check.md](judge_check.md)), then report the figure
- [x] data card completed (figures + observations)
- [x] exact model versions in the README
- [ ] demo video recorded
- [x] merge into `main`, then `git tag v1.0 && git push origin v1.0`
