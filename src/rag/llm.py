"""Clients LLM avec mesure systématique : tokens entrée/sortie, latence, raison d'arrêt, coût.

Format de messages neutre (converti pour chaque fournisseur) :
  {"role": "user"|"assistant", "content": str, "tool_calls": [{"id", "name", "arguments"}]}
  {"role": "tool", "tool_call_id": str, "name": str, "content": str}
Outils : liste de {"name", "description", "parameters" (JSON Schema)}.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .config import PRICING_PER_MTOK, Settings


@dataclass
class LLMResult:
    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    stop_reason: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return cost_usd(self.model, self.tokens_in, self.tokens_out)


def cost_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    price_in, price_out = PRICING_PER_MTOK.get(model, (0.0, 0.0))
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


class LLMError(RuntimeError):
    pass


class OllamaClient:
    """LLM local via l'API HTTP d'Ollama (https://github.com/ollama/ollama/blob/main/docs/api.md)."""

    def __init__(self, model: str, host: str, temperature: float = 0.0, num_ctx: int = 8192,
                 max_output_tokens: int = 700, timeout: float = 300.0):
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout

    @staticmethod
    def _convert(system: str, messages: list[dict]) -> list[dict]:
        out = [{"role": "system", "content": system}] if system else []
        for m in messages:
            if m["role"] == "tool":
                out.append({"role": "tool", "content": m["content"], "tool_name": m.get("name", "")})
            elif m["role"] == "assistant" and m.get("tool_calls"):
                out.append({"role": "assistant", "content": m.get("content", ""),
                            "tool_calls": [{"function": {"name": c["name"], "arguments": c["arguments"]}}
                                           for c in m["tool_calls"]]})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def chat(self, system: str, messages: list[dict], json_schema: dict | None = None,
             tools: list[dict] | None = None) -> LLMResult:
        import requests

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._convert(system, messages),
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx,
                        "num_predict": self.max_output_tokens, "seed": 0},
        }
        if json_schema is not None:
            payload["format"] = json_schema
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
            r.raise_for_status()
        except requests.RequestException as e:
            raise LLMError(f"Appel Ollama impossible ({self.host}, modèle {self.model}) : {e}") from e
        latency = time.perf_counter() - t0
        data = r.json()
        msg = data.get("message", {})
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            calls.append({"id": uuid.uuid4().hex[:8], "name": fn.get("name", ""), "arguments": args})
        return LLMResult(
            text=msg.get("content", ""),
            model=self.model,
            tokens_in=int(data.get("prompt_eval_count", 0)),
            tokens_out=int(data.get("eval_count", 0)),
            latency_s=latency,
            stop_reason="tool_use" if calls else data.get("done_reason", ""),
            tool_calls=calls,
        )


class AnthropicClient:
    """Modèle API (comparaison local vs API). Nécessite `uv sync --extra api` et ANTHROPIC_API_KEY."""

    def __init__(self, model: str, max_output_tokens: int = 2000, effort: str = "low"):
        import anthropic

        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.effort = effort

    @staticmethod
    def _convert(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                # tous les résultats d'outils d'un même tour vont dans un seul message user
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) \
                        and out[-1]["content"] and out[-1]["content"][0].get("type") == "tool_result":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif m["role"] == "assistant" and m.get("tool_calls"):
                content: list[dict] = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                content += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["arguments"]}
                            for c in m["tool_calls"]]
                out.append({"role": "assistant", "content": content})
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def chat(self, system: str, messages: list[dict], json_schema: dict | None = None,
             tools: list[dict] | None = None) -> LLMResult:
        output_config: dict[str, Any] = {"effort": self.effort}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_output_tokens,
            system=system,
            messages=self._convert(messages),
            output_config=output_config,
            # repli automatique côté serveur si le modèle refuse (stop_reason "refusal")
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if tools:
            kwargs["tools"] = [{"name": t["name"], "description": t["description"],
                                "input_schema": t["parameters"]} for t in tools]
        t0 = time.perf_counter()
        try:
            resp = self.client.beta.messages.create(**kwargs)
        except self.anthropic.APIStatusError as e:
            raise LLMError(f"Erreur API Anthropic ({e.status_code}) : {e.message}") from e
        except self.anthropic.APIConnectionError as e:
            raise LLMError(f"Connexion API Anthropic impossible : {e}") from e
        latency = time.perf_counter() - t0
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [{"id": b.id, "name": b.name, "arguments": dict(b.input)}
                 for b in resp.content if b.type == "tool_use"]
        return LLMResult(text=text, model=self.model, tokens_in=resp.usage.input_tokens,
                         tokens_out=resp.usage.output_tokens, latency_s=latency,
                         stop_reason=resp.stop_reason or "", tool_calls=calls)


def make_llm(settings: Settings, backend: str | None = None, model: str | None = None):
    backend = backend or settings.llm_backend
    if backend == "ollama":
        return OllamaClient(model or settings.ollama_model, settings.ollama_host, settings.temperature,
                            settings.num_ctx, settings.max_output_tokens)
    if backend == "anthropic":
        return AnthropicClient(model or settings.anthropic_model)
    if backend == "extractive":
        return None
    raise ValueError(f"Backend LLM inconnu : {backend}")


def parse_json_object(text: str) -> dict:
    """Extrait le premier objet JSON d'une sortie de modèle (tolère du texte autour)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    start = text.find("{")
    if start < 0:
        raise ValueError("aucun objet JSON dans la sortie")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("la sortie JSON n'est pas un objet")
    return obj
