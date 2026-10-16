# Lance toute la chaîne sur un PC Windows : vérifications, corpus, index, évaluations.
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
# Options : -SkipExperiments (seulement l'évaluation principale), -Model qwen2.5:3b-instruct
#           -LowMemory : vise environ la moitié de 16 Go de RAM (plus lent). Prend qwen2.5:3b-instruct
#                        sauf si -Model est donné, et relance le serveur Ollama avec des réglages économes.
param(
    [string]$Model = "qwen2.5:7b-instruct",
    [switch]$SkipExperiments,
    [switch]$LowMemory
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

Step "Vérifications"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw "uv introuvable : voir README (installation)." }
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) { throw "Ollama introuvable : installez-le depuis https://ollama.com/download puis rouvrez PowerShell." }
if (-not (Test-Path "data\raw\vosdroits-latest.zip")) { throw "Placez l'archive dans data\raw\vosdroits-latest.zip" }
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }
if ($LowMemory) {
    if (-not $PSBoundParameters.ContainsKey("Model")) { $Model = "qwen2.5:3b-instruct" }
    # Côté Python : lots plus petits pour les embeddings et le reranker, moins de threads
    $env:EMBED_BATCH_SIZE = "8"
    $env:RERANK_BATCH_SIZE = "8"
    $env:OMP_NUM_THREADS = "2"
    $env:MKL_NUM_THREADS = "2"
    $env:TOKENIZERS_PARALLELISM = "false"
    # Côté Ollama : un seul modèle chargé, une seule requête à la fois, cache KV compressé
    $env:OLLAMA_MAX_LOADED_MODELS = "1"
    $env:OLLAMA_NUM_PARALLEL = "1"
    $env:OLLAMA_FLASH_ATTENTION = "1"
    $env:OLLAMA_KV_CACHE_TYPE = "q8_0"
    $env:OLLAMA_KEEP_ALIVE = "2m"
    # Ces variables ne sont lues qu'au démarrage du serveur : on arrête celui de l'icône
    # de la barre des tâches et on en relance un qui hérite de ces réglages.
    # 127.0.0.1 et pas localhost : sous Windows, localhost essaie d'abord ::1 (IPv6), où Ollama
    # n'écoute pas, et chaque tentative refusée coûte ~2 s.
    $env:OLLAMA_HOST = "http://127.0.0.1:11434"
    Get-Process -Name "ollama app", "ollama" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 3
    $ollamaExe = (Get-Command ollama).Source
    $logOut = Join-Path $env:TEMP "ollama_serve_out.log"
    $logErr = Join-Path $env:TEMP "ollama_serve_err.log"
    $proc = Start-Process -FilePath $ollamaExe -ArgumentList "serve" -NoNewWindow -PassThru `
        -RedirectStandardOutput $logOut -RedirectStandardError $logErr
    $ok = $false
    for ($i = 0; $i -lt 90 -and -not $ok -and -not $proc.HasExited; $i++) {
        Start-Sleep -Seconds 1
        try { Invoke-RestMethod "http://127.0.0.1:11434/api/version" -TimeoutSec 5 | Out-Null; $ok = $true } catch {}
    }
    if (-not $ok) {
        Write-Host "--- fin du journal d'Ollama ($logErr) ---" -ForegroundColor Yellow
        if (Test-Path $logErr) { Get-Content $logErr -Tail 20 }
        throw "Le serveur Ollama ne répond pas après redémarrage. Relancez l'application Ollama, ou lancez 'ollama serve' dans un autre terminal."
    }
    Write-Host "Mode mémoire réduite : modèle $Model, serveur Ollama relancé avec réglages économes."
}
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
if ($LowMemory) { ollama ps }
Write-Host "Résultats : eval\report.md, eval\metrics.json (et eval\runs\*\report.md)"
Write-Host "Envoyez eval\report.md et les fichiers eval\runs\*\metrics.json pour l'analyse."
