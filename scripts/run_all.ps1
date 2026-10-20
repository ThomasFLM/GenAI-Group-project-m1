# Runs the whole pipeline on a Windows PC: checks, corpus, index, evaluations.
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
# Options: -SkipExperiments (main evaluation only), -Model qwen2.5:3b-instruct
#          -LowMemory: targets about half of 16 GB of RAM (slower). Uses qwen2.5:3b-instruct
#                      unless -Model is given, and restarts the Ollama server with frugal settings.
param(
    [string]$Model = "qwen2.5:7b-instruct",
    [switch]$SkipExperiments,
    [switch]$LowMemory
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

Step "Checks"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw "uv not found: see README (installation)." }
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) { throw "Ollama not found: install it from https://ollama.com/download then reopen PowerShell." }
if (-not (Test-Path "data\raw\vosdroits-latest.zip")) { throw "Put the archive in data\raw\vosdroits-latest.zip" }
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }
# Hugging Face downloads through "Xet" fail on some networks (CAS Client Error):
# stick to plain HTTP downloads.
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
if ($LowMemory) {
    if (-not $PSBoundParameters.ContainsKey("Model")) { $Model = "qwen2.5:3b-instruct" }
    # Python side: smaller batches for embeddings and reranker, fewer threads
    $env:EMBED_BATCH_SIZE = "8"
    $env:RERANK_BATCH_SIZE = "8"
    $env:OMP_NUM_THREADS = "2"
    $env:MKL_NUM_THREADS = "2"
    $env:TOKENIZERS_PARALLELISM = "false"
    # Ollama side: one loaded model, one request at a time, compressed KV cache
    $env:OLLAMA_MAX_LOADED_MODELS = "1"
    $env:OLLAMA_NUM_PARALLEL = "1"
    $env:OLLAMA_FLASH_ATTENTION = "1"
    $env:OLLAMA_KV_CACHE_TYPE = "q8_0"
    $env:OLLAMA_KEEP_ALIVE = "2m"
    # These variables are only read when the server starts: stop the one started by the
    # taskbar icon and start a new one that inherits these settings.
    # 127.0.0.1 rather than localhost: on Windows, localhost first tries ::1 (IPv6), where Ollama
    # does not listen, and each refused attempt costs ~2 s.
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
        Write-Host "--- end of the Ollama log ($logErr) ---" -ForegroundColor Yellow
        if (Test-Path $logErr) { Get-Content $logErr -Tail 20 }
        throw "The Ollama server does not answer after restarting. Restart the Ollama app, or run 'ollama serve' in another terminal."
    }
    Write-Host "Low-memory mode: model $Model, Ollama server restarted with frugal settings."
}
$env:OLLAMA_MODEL = $Model
$env:JUDGE_MODEL = $Model

Step "Dependencies"
uv sync

Step "Local model $Model"
ollama pull $Model

Step "Tests"
uv run pytest -q

Step "Corpus (500 fact sheets) and statistics"
uv run python -m rag.cli build-corpus --source data\raw\vosdroits-latest.zip
uv run python -m rag.cli stats | Out-Null

Step "Index (chunking + embeddings; the first run downloads the models)"
uv run python -m rag.cli index

Step "Quick check"
uv run python -m rag.cli ask "Combien coûte un passeport pour un adulte ?"

Step "Main evaluation (hybrid + reranker + local LLM)"
uv run python eval/run_eval.py --questions eval/golden.jsonl --out eval/

if (-not $SkipExperiments) {
    Step "Comparison experiments (eval\runs\...)"
    uv run python eval/run_eval.py --out eval --tag runs/e0_bm25 --backend extractive --retrieval bm25 --no-rerank --no-judge
    uv run python eval/run_eval.py --out eval --tag runs/e2_dense --backend extractive --retrieval dense --no-rerank --no-judge
    uv run python eval/run_eval.py --out eval --tag runs/e3_hybrid --backend extractive --no-rerank --no-judge
    uv run python eval/run_eval.py --out eval --tag runs/e4_rerank --backend extractive --no-judge
    uv run python eval/run_eval.py --out eval --tag runs/e5_agent --mode agent
}

Step "Done"
if ($LowMemory) { ollama ps }
Write-Host "Results: eval\report.md, eval\metrics.json (and eval\runs\*\report.md)"
Write-Host "Share eval\report.md and the eval\runs\*\metrics.json files for the analysis."
