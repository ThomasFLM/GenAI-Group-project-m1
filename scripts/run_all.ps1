# Lance toute la chaîne sur un PC Windows : vérifications, corpus, index, évaluations.
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
# Options : -SkipExperiments (seulement l'évaluation principale), -Model qwen2.5:3b-instruct
param(
    [string]$Model = "qwen2.5:7b-instruct",
    [switch]$SkipExperiments
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

Step "Vérifications"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw "uv introuvable : voir README (installation)." }
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) { throw "Ollama introuvable : installez-le depuis https://ollama.com/download puis rouvrez PowerShell." }
if (-not (Test-Path "data\raw\vosdroits-latest.zip")) { throw "Placez l'archive dans data\raw\vosdroits-latest.zip" }
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }
$env:OLLAMA_MODEL = $Model
$env:JUDGE_MODEL = $Model

Step "Dépendances"
uv sync

Step "Modèle local $Model"
ollama pull $Model

Step "Tests"
uv run pytest -q

Step "Corpus (500 fiches) et statistiques"
uv run python -m rag.cli build-corpus --source data\raw\vosdroits-latest.zip
uv run python -m rag.cli stats | Out-Null

Step "Index (découpage + embeddings ; le 1er lancement télécharge les modèles)"
uv run python -m rag.cli index

Step "Essai rapide"
uv run python -m rag.cli ask "Combien coûte un passeport pour un adulte ?"

Step "Évaluation principale (hybride + reranker + LLM local)"
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/

if (-not $SkipExperiments) {
    Step "Expériences de comparaison (eval\runs\...)"
    uv run python eval/run_eval.py --out eval --tag runs/e0_bm25 --backend extractive --retrieval bm25 --no-rerank --no-judge
    uv run python eval/run_eval.py --out eval --tag runs/e2_dense --backend extractive --retrieval dense --no-rerank --no-judge
    uv run python eval/run_eval.py --out eval --tag runs/e3_hybrid --backend extractive --no-rerank --no-judge
    uv run python eval/run_eval.py --out eval --tag runs/e4_rerank --backend extractive --no-judge
    uv run python eval/run_eval.py --out eval --tag runs/e5_agent --mode agent
}

Step "Terminé"
Write-Host "Résultats : eval\report.md, eval\metrics.json (et eval\runs\*\report.md)"
Write-Host "Envoyez eval\report.md et les fichiers eval\runs\*\metrics.json pour l'analyse."
