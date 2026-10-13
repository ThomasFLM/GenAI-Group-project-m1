"""Configuration centrale : chemins, modèles, hyperparamètres.

Toutes les valeurs peuvent être surchargées par variables d'environnement (fichier .env).
Chaque expérience de l'évaluation = un jeu de valeurs ici, ce qui rend les runs comparables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv absent : on lit seulement l'environnement
    pass

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

DOCS_PATH = PROCESSED_DIR / "documents.jsonl"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0").lower() in {"1", "true", "yes", "oui"}


# Thèmes Service-Public retenus pour le sous-corpus (~500 fiches) : voir docs/data_card.md
DEFAULT_THEMES = [
    "Papiers - Citoyenneté - Élections",
    "Logement",
    "Transports - Mobilité",
]

# Prix en $ par million de tokens (entrée, sortie). Modèles locaux : 0 $ d'API
# (le coût réel est matériel/énergie, estimé à part dans le rapport).
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


@dataclass
class Settings:
    # Corpus
    corpus_url: str = field(default_factory=lambda: _env(
        "CORPUS_URL", "https://lecomarquage.service-public.fr/vdd/3.4/part/zip/vosdroits-latest.zip"))
    themes: list[str] = field(default_factory=lambda: list(DEFAULT_THEMES))
    max_docs: int = field(default_factory=lambda: _env_int("MAX_DOCS", 500))

    # Découpage
    chunk_max_words: int = field(default_factory=lambda: _env_int("CHUNK_MAX_WORDS", 220))
    chunk_overlap_words: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP_WORDS", 40))

    # Retrieval
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"))
    reranker_model: str = field(default_factory=lambda: _env(
        "RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"))
    use_reranker: bool = field(default_factory=lambda: _env_bool("USE_RERANKER", True))
    retrieval_mode: str = field(default_factory=lambda: _env("RETRIEVAL_MODE", "hybrid"))  # bm25 | dense | hybrid
    candidates_k: int = field(default_factory=lambda: _env_int("CANDIDATES_K", 30))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    rrf_k: int = 60
    # Seuil d'abstention sur le score du reranker (logit). À calibrer sur le split dev.
    min_rerank_score: float = field(default_factory=lambda: _env_float("MIN_RERANK_SCORE", -2.0))

    # Génération
    llm_backend: str = field(default_factory=lambda: _env("LLM_BACKEND", "ollama"))
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "qwen2.5:7b-instruct"))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-opus-5-5"))
    temperature: float = field(default_factory=lambda: _env_float("TEMPERATURE", 0.0))
    max_output_tokens: int = field(default_factory=lambda: _env_int("MAX_OUTPUT_TOKENS", 700))
    num_ctx: int = field(default_factory=lambda: _env_int("NUM_CTX", 8192))

    # Agent
    agent_max_steps: int = field(default_factory=lambda: _env_int("AGENT_MAX_STEPS", 4))

    # Garde-fous
    injection_policy: str = field(default_factory=lambda: _env("INJECTION_POLICY", "block"))  # block | warn

    # Juge
    judge_backend: str = field(default_factory=lambda: _env("JUDGE_BACKEND", "ollama"))
    judge_model: str = field(default_factory=lambda: _env("JUDGE_MODEL", "qwen2.5:7b-instruct"))

    @property
    def generation_model(self) -> str:
        if self.llm_backend == "anthropic":
            return self.anthropic_model
        if self.llm_backend == "ollama":
            return self.ollama_model
        return self.llm_backend

    def to_dict(self) -> dict:
        return asdict(self)


def get_settings(**overrides) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        if v is not None:
            setattr(s, k, v)
    return s
