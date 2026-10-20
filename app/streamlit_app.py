"""Demo interface: answer, sources, trace (tools, tokens, latency).

    uv run --extra ui streamlit run app/streamlit_app.py

The interface is in English; questions and answers are in French, like the corpus.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import get_settings  # noqa: E402
from rag.pipeline import Assistant  # noqa: E402

st.set_page_config(page_title="Service-Public assistant", page_icon="🏛️")
st.title("Administrative procedures assistant — Service-Public.fr fact sheets")


@st.cache_resource
def load_assistant(backend: str) -> Assistant:
    return Assistant.from_disk(get_settings(llm_backend=backend))


with st.sidebar:
    backend = st.selectbox("Model", ["ollama", "anthropic", "extractive"])
    mode = st.radio("Mode", ["rag", "agent"])
    since = st.text_input("Fact sheets updated since (YYYY-MM-DD)", "") or None

question = st.text_input("Your question (in French)", "Quel est le délai pour contester une amende forfaitaire ?")
if st.button("Ask") and question:
    with st.spinner("Searching and writing…"):
        resp = load_assistant(backend).ask(question, mode=mode, since=since)
    st.markdown(f"### Answer\n{resp.answer}")
    for doc_id in resp.cited_sources:
        st.markdown(f"- [{doc_id}](https://www.service-public.gouv.fr/particuliers/vosdroits/{doc_id})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latency", f"{resp.latency_s:.2f} s")
    c2.metric("Tokens in/out", f"{resp.tokens_in}/{resp.tokens_out}")
    c3.metric("Cost", f"{resp.cost_usd:.5f} $")
    c4.metric("Stop reason", resp.stop_reason)
    with st.expander("Agent trace"):
        for t in resp.tool_calls:
            st.write(f"🔧 `{t.name}` {t.arguments} → {t.result_summary} ({t.latency_s:.2f}s)")
        for c in resp.llm_calls:
            st.write(f"🧠 {c.purpose} · {c.model} · {c.tokens_in}/{c.tokens_out} tokens · {c.latency_s:.2f}s · "
                     f"{c.stop_reason}")
        st.json({"timings": resp.timings, "flags": resp.flags, "retrieved": resp.retrieved_chunk_ids})
