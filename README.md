# Assistant RAG & agent sur les fiches Service-Public.fr

Assistant qui répond à des questions de particuliers (papiers, logement, transports) **uniquement** à partir
d'environ 500 fiches pratiques officielles publiées sur data.gouv.fr, en **citant ses sources**, en
**s'abstenant** (« je ne sais pas ») quand l'information n'est pas dans le corpus et en **résistant aux
injections de prompt**. Tout tourne en local (LLM via Ollama). Chaque réponse est instrumentée
(tokens, latence, appels d'outils, coût, raison d'arrêt) et le système est évalué sur un golden set de
42 questions.

- Rendu : **20 octobre 2026, 23:59** (tag git `v1.0`) · Soutenance : **21 octobre 2026** (10 min + 5 min de questions)

## Architecture

```
question ─► garde-fou entrée (injection ?) ─► recherche hybride ─────────────► génération structurée ─► garde-fous sortie ─► réponse + sources
                                               BM25 (top 30)  ┐                 LLM local (Ollama)         JSON validé (pydantic)
                                               dense E5 (30)  ┴► RRF ─► rerank   JSON {answerable,          citations ⊂ documents récupérés
                                               filtres : date, thème  cross-     answer, citations}         canari anti-fuite du prompt
                                                                      encoder    seuil d'abstention         1 nouvel essai si invalide
mode agent : le LLM appelle lui-même les outils search_corpus / get_document / add_days (≤ 4 étapes)
```

Détails et justification des choix : [docs/architecture.md](docs/architecture.md).
Corpus : [docs/data_card.md](docs/data_card.md). Protocole d'évaluation : [docs/evaluation.md](docs/evaluation.md).
Soutenance : [docs/soutenance.md](docs/soutenance.md).

## Modèles utilisés (noms et versions exacts)

| Rôle | Modèle par défaut | Alternative testée |
|---|---|---|
| LLM de génération (local) | `qwen2.5:7b-instruct` (Ollama) | `llama3.2:3b` |
| LLM juge (évaluation) | `qwen2.5:7b-instruct` (Ollama) | — |
| Embeddings | `intfloat/multilingual-e5-small` | `BAAI/bge-m3` |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | `BAAI/bge-reranker-v2-m3` |
| LLM API (comparaison, optionnel) | `claude-opus-5-5` (Anthropic) | `claude-haiku-5-5` |

> Notez ici la version exacte d'Ollama (`ollama --version`) et le digest des modèles (`ollama list`) utilisés pour le run final.

## Installation

```bash
git clone <url-du-repo> && cd GenAI-Group-project-m1
uv sync                       # installe les dépendances (uv.lock)
cp .env.example .env          # puis adapter si besoin
# LLM local
ollama pull qwen2.5:7b-instruct
```

Options : `uv sync --extra api` (comparaison avec un modèle API, nécessite `ANTHROPIC_API_KEY` dans `.env`),
`uv sync --extra ui` (interface Streamlit).

## Utilisation

```bash
# 1. Données : télécharger l'archive XML officielle (ou la déposer à la main dans data/raw/)
uv run python -m rag.cli download
# 2. Sélectionner ~500 fiches (thèmes Papiers, Logement, Transports) et les nettoyer
uv run python -m rag.cli build-corpus --source data/raw/vosdroits-latest.zip
# 3. Statistiques pour la data card + 20 documents / 20 passages tirés au hasard à lire
uv run python -m rag.cli stats
# 4. Découpage + index (embeddings)
uv run python -m rag.cli index
# 5. Poser une question
uv run python -m rag.cli ask "Quel est le délai pour contester une amende forfaitaire ?"
uv run python -m rag.cli ask "J'ai reçu un avis de contravention le 1er septembre 2026, jusqu'à quand le contester ?" --mode agent -v
# 6. Évaluation complète
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/
# Interface (bonus)
uv run --extra ui streamlit run app/streamlit_app.py
```

`eval/run_eval.py` écrit `eval/results.jsonl` (par question : réponse, sources citées, documents récupérés,
tokens in/out, latence, appels d'outils, raison d'arrêt), `eval/scores.jsonl`, `eval/metrics.json` et
`eval/report.md` (tableau de bord + analyse d'erreurs). Expériences : `--mode agent`, `--retrieval bm25|dense|hybrid`,
`--no-rerank`, `--top-k 3`, `--backend anthropic`, `--model llama3.2:3b`, `--split dev|test`, `--tag nom`
(sortie dans `eval/<nom>/`).

Annotation du golden set : `uv run python -m rag.cli suggest-gold` propose les fiches candidates pour chaque
question ; on vérifie dans la fiche, on remplit `gold_docs` et on retire `gold_status`, puis on relance
`build-corpus` (qui force l'inclusion des fiches du golden set) et `index`.

Tests : `uv run pytest` (sans réseau ni GPU : embedder haché et LLM simulé).

## Structure

```
src/rag/
  config.py       paramètres (surchargeables via .env) et tarifs
  ingestion.py    téléchargement, parsing XML, nettoyage, sélection du sous-corpus
  chunking.py     découpage par section avec recouvrement
  retrieval.py    BM25, dense, fusion RRF, reranking, filtres de métadonnées, index
  llm.py          clients Ollama / Anthropic, mesure tokens·latence·coût·stop reason
  prompts.py      prompts système, balisage du contexte, prompt du juge
  guardrails.py   détection d'injection, validation de sortie, canari
  pipeline.py     Assistant : mode RAG et mode agent (outils), trace complète
  evaluation.py   métriques, taxonomie d'erreurs, rapport
  cli.py          ligne de commande
eval/golden.jsonl, eval/run_eval.py
app/streamlit_app.py
docs/             data card, architecture, évaluation, soutenance
tests/            tests unitaires + fiches XML de test
```

## Équipe

| Nom | Partie du projet (et de la soutenance) |
|-----|-------------------------|
|     | Données : corpus, nettoyage, data card |
|     | Retrieval : découpage, index hybride, reranking |
|     | Génération & agent : prompts, sortie structurée, garde-fous, démo |
|     | Évaluation : golden set, métriques, analyse d'erreurs, risques |

## Contribuer

Voir [CONTRIBUTING.md](CONTRIBUTING.md). Ne jamais commiter de clé d'API : `.env` est ignoré par git.
