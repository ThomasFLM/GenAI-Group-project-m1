"""Interface de démonstration : réponse, sources, trace (outils, tokens, latence).

    uv run --extra ui streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import get_settings  # noqa: E402
from rag.pipeline import Assistant  # noqa: E402

st.set_page_config(page_title="Assistant Service-Public", page_icon="🏛️")
st.title("Assistant démarches — fiches Service-Public.fr")


@st.cache_resource
def load_assistant(backend: str) -> Assistant:
    return Assistant.from_disk(get_settings(llm_backend=backend))


with st.sidebar:
    backend = st.selectbox("Modèle", ["ollama", "anthropic", "extractive"])
    mode = st.radio("Mode", ["rag", "agent"])
    since = st.text_input("Fiches modifiées depuis (YYYY-MM-DD)", "") or None

question = st.text_input("Votre question", "Quel est le délai pour contester une amende forfaitaire ?")
if st.button("Demander") and question:
    with st.spinner("Recherche et rédaction…"):
        resp = load_assistant(backend).ask(question, mode=mode, since=since)
    st.markdown(f"### Réponse\n{resp.answer}")
    for doc_id in resp.cited_sources:
        st.markdown(f"- [{doc_id}](https://www.service-public.fr/particuliers/vosdroits/{doc_id})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latence", f"{resp.latency_s:.2f} s")
    c2.metric("Tokens in/out", f"{resp.tokens_in}/{resp.tokens_out}")
    c3.metric("Coût", f"{resp.cost_usd:.5f} $")
    c4.metric("Arrêt", resp.stop_reason)
    with st.expander("Trace de l'agent"):
        for t in resp.tool_calls:
            st.write(f"🔧 `{t.name}` {t.arguments} → {t.result_summary} ({t.latency_s:.2f}s)")
        for c in resp.llm_calls:
            st.write(f"🧠 {c.purpose} · {c.model} · {c.tokens_in}/{c.tokens_out} tokens · {c.latency_s:.2f}s · "
                     f"{c.stop_reason}")
        st.json({"timings": resp.timings, "flags": resp.flags, "retrieved": resp.retrieved_chunk_ids})
