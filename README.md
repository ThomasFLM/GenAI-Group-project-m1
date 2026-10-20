# RAG & agent assistant over the Service-Public.fr fact sheets

An assistant that answers questions from private individuals (identity papers, housing, transport)
**only** from about 500 official fact sheets published on data.gouv.fr. It **cites its sources**,
**abstains** ("je ne sais pas" — I don't know) when the information is not in the corpus, and **resists
prompt injection**. Everything runs locally (LLM through Ollama). Every answer is instrumented
(tokens, latency, tool calls, cost, stop reason) and the system is evaluated on a golden set of
42 questions.

- Submission: **20 October 2026, 23:59** (git tag `v1.0`) · Defence: **21 October 2026** (10 min + 5 min Q&A)

The corpus, the questions and the assistant's answers are in **French**; code and documentation are in
English. The LLM prompts are kept in French on purpose (see `src/rag/prompts.py`): the reported results
were measured with them.

## Architecture

```
question ─► input guardrail (injection?) ─► hybrid search ───────────────────► structured generation ─► output guardrails ─► answer + sources
                                             BM25 (top 30)  ┐                   local LLM (Ollama)         JSON validated (pydantic)
                                             dense E5 (30)  ┴► RRF ─► rerank     JSON {answerable,          citations ⊂ retrieved documents
                                             filters: date, theme    cross-      answer, citations}         prompt-leak canary
                                                                     encoder     abstention threshold       1 retry if invalid
agent mode: the LLM calls the search_corpus / get_document / add_days tools itself (≤ 4 steps)
```

Design details and rationale: [docs/architecture.md](docs/architecture.md).
Corpus: [docs/data_card.md](docs/data_card.md). Evaluation protocol and results: [docs/evaluation.md](docs/evaluation.md).
Defence plan: [docs/defense.md](docs/defense.md).

## Models used (exact names and versions)

| Role | Default model (final run) | Exact version | Alternative tested |
|---|---|---|---|
| Generation LLM (local) | `qwen2.5:7b-instruct` (Ollama) | digest `845dbda0ea48` | `qwen2.5:3b-instruct` (`357c53fb659c`, E6) |
| Judge LLM (evaluation) | `qwen2.5:7b-instruct` (Ollama) | digest `845dbda0ea48` | — |
| Embeddings | `intfloat/multilingual-e5-small` | HF revision `614241f6` | version fine-tuned on the corpus (E9) |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | HF revision `1427fd65` | version fine-tuned on the corpus (E9) |
| API LLM (comparison, optional) | `claude-opus-5-5` (Anthropic) | — | not evaluated (no API key) |

Final run: Ollama **0.40.4**, PyTorch 2.14.1, sentence-transformers 6.1.0, on a Windows 11 laptop
(16 GB RAM, NVIDIA RTX 3060 Laptop 6 GB). Everything also works without a GPU, more slowly
(`scripts/run_all.ps1 -LowMemory`: about 8 GB of RAM, `qwen2.5:3b-instruct`).

## Installation

```bash
git clone <repo-url> && cd GenAI-Group-project-m1
uv sync                       # install the dependencies (uv.lock)
cp .env.example .env          # then adapt if needed
# local LLM
ollama pull qwen2.5:7b-instruct
```

Options: `uv sync --extra api` (comparison with an API model, needs `ANTHROPIC_API_KEY` in `.env`),
`uv sync --extra ui` (Streamlit interface).

On Windows, `powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1` runs the whole pipeline
(checks, corpus, index, main evaluation, comparison experiments).

## Usage

```bash
# 1. Data: download the official XML archive (or drop it by hand into data/raw/)
uv run python -m rag.cli download
# 2. Select ~500 fact sheets (themes: identity papers, housing, transport) and clean them
uv run python -m rag.cli build-corpus --source data/raw/vosdroits-latest.zip
# 3. Statistics for the data card + 20 documents / 20 chunks drawn at random to read
uv run python -m rag.cli stats
# 4. Chunking + index (embeddings)
uv run python -m rag.cli index
# 5. Ask a question (in French)
uv run python -m rag.cli ask "Quel est le délai pour contester une amende forfaitaire ?"
uv run python -m rag.cli ask "J'ai reçu un avis de contravention le 1er septembre 2026, jusqu'à quand le contester ?" --mode agent -v
# 6. Full evaluation
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/
# Interface (bonus)
uv run --extra ui streamlit run app/streamlit_app.py
```

`eval/run_eval.py` writes `eval/results.jsonl` (per question: answer, cited sources, retrieved documents,
tokens in/out, latency, tool calls, stop reason), `eval/scores.jsonl`, `eval/metrics.json` and
`eval/report.md` (dashboard + error analysis). Experiments: `--mode agent`, `--retrieval bm25|dense|hybrid`,
`--no-rerank`, `--top-k 3`, `--backend anthropic`, `--model llama3.2:3b`, `--split dev|test`, `--tag name`
(output in `eval/<name>/`).

Golden-set annotation: `uv run python -m rag.cli suggest-gold` suggests candidate fact sheets for each
question; check the fact sheet, fill in `gold_docs` and remove `gold_status`, then rerun `build-corpus`
(which forces the inclusion of the golden-set fact sheets) and `index`.

Tests: `uv run pytest` (no network or GPU needed: hashed embedder and simulated LLM).

### NVIDIA GPU (optional)

`uv sync` installs the CPU build of PyTorch. To use an NVIDIA card (embeddings, reranker and fine-tuning
~10× faster): `uv pip install torch==2.14.1 --torch-backend cu130`, then run the commands with `UV_NO_SYNC=1`
(otherwise `uv run` reinstalls the CPU build). Ollama uses the GPU on its own.

### Fine-tuning the retrieval models (experiment E9, not part of the default configuration)

```bash
uv sync --extra train
uv run python -m rag.cli gen-queries --n-chunks 4000 --workers 4   # synthetic questions (already in data/train/)
uv run python -m rag.cli finetune                                   # -> models/e5-small-servicepublic
uv run python -m rag.cli finetune-reranker                          # -> models/mmarco-reranker-servicepublic
```

Synthetic questions too close to the golden set are removed, and validation uses held-out fact sheets.
Results and decision (not kept by default): [docs/evaluation.md](docs/evaluation.md#e9-fine-tuning-the-retrieval-models).

## Layout

```
src/rag/
  config.py       settings (overridable through .env) and prices
  ingestion.py    download, XML parsing, cleaning, sub-corpus selection
  chunking.py     section-aware chunking with overlap
  retrieval.py    BM25, dense, RRF fusion, reranking, metadata filters, index
  llm.py          Ollama / Anthropic clients, tokens·latency·cost·stop-reason measurement
  prompts.py      system prompts, context tagging, judge prompt
  guardrails.py   injection detection, output validation, canary
  pipeline.py     Assistant: RAG mode and agent mode (tools), full trace
  evaluation.py   metrics, error taxonomy, report
  finetune.py     synthetic questions, embedding and reranker fine-tuning (E9)
  cli.py          command line
eval/golden.jsonl, eval/run_eval.py
app/streamlit_app.py
scripts/run_all.ps1
docs/             data card, architecture, evaluation, defence plan
tests/            unit tests + test XML fact sheets
```



## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Never commit an API key: `.env` is ignored by git.
