import pytest

from rag.llm import AnthropicClient, OllamaClient, cost_usd, parse_json_object

HISTORY = [
    {"role": "user", "content": "q"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "a", "name": "search_corpus", "arguments": {"query": "x"}},
                                                         {"id": "b", "name": "add_days", "arguments": {}}]},
    {"role": "tool", "tool_call_id": "a", "name": "search_corpus", "content": "r1"},
    {"role": "tool", "tool_call_id": "b", "name": "add_days", "content": "r2"},
]


def test_ollama_conversion():
    msgs = OllamaClient._convert("sys", HISTORY)
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "search_corpus"
    assert msgs[3] == {"role": "tool", "content": "r1", "tool_name": "search_corpus"}


def test_anthropic_conversion_groups_tool_results():
    msgs = AnthropicClient._convert(HISTORY)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert [b["tool_use_id"] for b in msgs[2]["content"]] == ["a", "b"]
    assert msgs[1]["content"][0]["type"] == "tool_use"


def test_ollama_chat_parses_usage(monkeypatch):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": '{"answerable": false, "answer": "x", "citations": []}'},
                    "prompt_eval_count": 321, "eval_count": 12, "done_reason": "stop"}

    import requests

    sent = {}
    monkeypatch.setattr(requests, "post", lambda url, json, timeout: sent.update(json) or Resp())
    r = OllamaClient("m", "http://h").chat("s", [{"role": "user", "content": "q"}], json_schema={"type": "object"})
    assert (r.tokens_in, r.tokens_out, r.stop_reason) == (321, 12, "stop")
    assert sent["format"] == {"type": "object"} and sent["options"]["temperature"] == 0.0


def test_parse_json_and_cost():
    assert parse_json_object('Voici : {"a": 1} merci') == {"a": 1}
    with pytest.raises(ValueError):
        parse_json_object("rien")
    assert cost_usd("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.6)
    assert cost_usd("qwen2.5:7b-instruct", 10_000, 10_000) == 0.0
