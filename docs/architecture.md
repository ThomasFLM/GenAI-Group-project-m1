# Architecture and technical choices

Each choice below is a **hypothesis to measure** on the `dev` split of the golden set, then confirmed
once on the `test` split (questions held out during tuning).

## 1. Ingestion and cleaning (`ingestion.py`)

- Source: the official XML archive `vosdroits-latest.zip` (DILA), one fact sheet = one `F*.xml` file.
  Identifier = the `ID` attribute (e.g. `F1234`), reused as-is in `gold_docs` and in citations.
- Kept: title, summary, theme, modification date, and the structured text (chapters, sub-chapters,
  cases/situations, lists, tables, "À savoir / Attention" boxes).
- Removed: breadcrumbs, menus, "Voir aussi" (see also), legal references, online services, "Où s'adresser"
  (where to go), partners — noise for lexical search that never contains an answer.
- Normalisation: Unicode NFC, non-breaking spaces, repeated spaces.
- Selection (~500): themes *Papiers - Citoyenneté - Élections*, *Logement*, *Transports - Mobilité*,
  spread evenly, deterministic order; the golden-set fact sheets are always included.

## 2. Chunking (`chunking.py`)

- **By chapter**: a chunk never spans two chapters (first heading level).
- Inside a chapter, sub-sections (cases such as "location vide" / "location meublée", "majeur" /
  "mineur") are packed together up to **220 words** with **40 words** of overlap, and each case
  **keeps its heading inside the chunk text** ("### Location vide"). The model therefore knows which case
  each sentence applies to. A first version that split strictly by section produced 14,632 crumbs with a
  median of 41 words, because real fact sheets contain dozens of small nested cases; grouping by chapter
  gives about 8,300 chunks with a median of 195 words.
- A sub-heading is never separated from its first paragraph; a chunk that starts in the middle of a case
  repeats its heading followed by "(suite)".
- 220 words ≈ 300 tokens: enough for a rule and its exceptions, short enough for 5 chunks to fit in the
  context of a 7B model.
- **Contextual header** indexed with each chunk: "fact sheet title — chapter". Without it, a chunk
  saying "Le délai est de 2 mois" can be found neither by BM25 nor by the embeddings.
- Experiment E8 compares 120 / 220 / 400 words (see evaluation.md).

## 3. Hybrid retrieval (`retrieval.py`)

| Step | Choice | Why |
|---|---|---|
| Lexical | BM25 (rank-bm25), French tokenisation without accents, stop words, light stemming | questions contain exact terms ("dépôt de garantie", "zone tendue") |
| Dense | `multilingual-e5-small` (`query:` / `passage:` prefixes), cosine | paraphrases ("rendre la caution" ↔ "restitution du dépôt de garantie") |
| Fusion | RRF (k = 60), 30 candidates per engine | robust, no need to normalise heterogeneous scores |
| Reranking | multilingual cross-encoder on the 30 candidates → top 5 | top-5 precision, and an absolute score usable as an abstention threshold |
| Filters | update date (`since`), theme | "since 2024", "latest fact sheets" questions |

In-memory index (numpy): ~8,300 chunks × 384 dimensions, search in a few milliseconds; a vector database
would bring nothing at this scale (bonus idea: Qdrant for metadata filters).

## 4. Generation (`pipeline.py`, `prompts.py`, `llm.py`)

- Local LLM through Ollama, temperature 0, fixed `seed` (reproducibility).
- Prompts are in French, like the corpus and the questions; the evaluation results were measured with
  these exact prompts.
- **Structured output**: JSON schema enforced by Ollama (`format`), then pydantic validation:
  `{"answerable": bool, "answer": str, "citations": [str]}`. Citations outside the retrieved documents are
  removed; an "answerable" answer without a valid citation is invalid → **one** retry with the error
  message, then the standard refusal.
- Two-level **abstention**: (1) reranker score below the threshold → refusal without calling the LLM
  (saves latency and tokens); (2) the LLM sets `answerable: false`.
- **Agent mode**: tool-calling loop (≤ 4 steps) — `search_corpus` (reformulation, multiple queries, date
  filter), `get_document` (read a whole fact sheet), `add_days` (deadline computation: LLMs make mistakes
  in date arithmetic). If the agent does not search at all, a search is forced.
- **Measurement**: tokens in/out (Ollama's `prompt_eval_count` / `eval_count`, the API's `usage`),
  per-step latency (BM25, dense, rerank, LLM), tool calls, cost ($ for the API; $0 locally, hardware cost
  discussed separately), stop reason (`stop`, `length`, `abstained`, `low_retrieval_score`,
  `injection_blocked`, `validation_failed`, `output_blocked_leak`, `max_steps`, `llm_error`).

## 5. Security: prompt injection (`guardrails.py`)

Defence in depth:
1. **Input**: FR/EN patterns ("ignore les instructions", "prompt système", "tu es désormais",
   `<system>` tags…). `block` policy (immediate refusal) or `warn` (let it through, the other layers protect).
2. **Context**: chunks and question wrapped in tags and declared as *data*.
   Indirect injection tested: a test fact sheet contains "Ignore les instructions précédentes".
3. **Output**: secret canary in the system prompt; if it shows up, the answer is blocked. Citations are
   restricted to the documents actually retrieved (no fake source).

Acknowledged limitation: the patterns were written while looking at the golden-set injections → the
measured resistance rate is optimistic. For an honest measurement, add new injections to the `test` split
without touching the patterns (see evaluation.md).
