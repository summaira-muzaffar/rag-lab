import html
import re
from pathlib import Path

import streamlit as st

from config import settings
from rag.pipeline import RAG

st.set_page_config(page_title="Policy Assistant", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 900px; padding-top: 5rem;}
    h1 {font-size: 2rem !important;}
    .src {border: 1px solid rgba(128,128,128,.35); border-radius: 10px;
          padding: .7rem .9rem; margin-bottom: .6rem;}
    .src-title {font-weight: 600; font-size: .9rem; margin-bottom: .3rem;}
    .src-text {font-size: .85rem; opacity: .85; white-space: pre-wrap; line-height: 1.45;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading embedding model (first time only)...")
def get_rag():
    return RAG(with_llm=True)


rag = get_rag()

if "messages" not in st.session_state:
    st.session_state.messages = []


def clean_snippet(text: str, limit: int = 350) -> str:
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)   # remove markdown heading marks
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)         # remove bold marks
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    text = "\n".join(lines)
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0] + " ..."
    return text


def render_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources used ({len(sources)})"):
        for i, h in enumerate(sources, start=1):
            name = html.escape(str(h.get("source", "unknown")))
            snippet = html.escape(clean_snippet(h.get("text") or ""))
            st.markdown(
                f'<div class="src"><div class="src-title">[{i}] {name}</div>'
                f'<div class="src-text">{snippet}</div></div>',
                unsafe_allow_html=True,
            )


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Controls")
    if st.button("Build / rebuild index", use_container_width=True):
        with st.spinner("Indexing documents..."):
            n = rag.ingest()
        st.success(f"Indexed {n} chunks")
    st.caption("Click this once before asking, and again after changing documents.")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("Documents")
    docs_dir = Path(settings.docs_dir)
    files = sorted(p.name for p in docs_dir.glob("*") if p.is_file()) if docs_dir.exists() else []
    for name in files:
        st.write(f"- {name}")
    if not files:
        st.write("No documents found.")

    st.divider()
    st.subheader("Settings")
    st.caption(f"LLM in use: {type(rag.llm).__name__}")
    st.caption(f"Chunk size: {settings.chunk_size} | Overlap: {settings.chunk_overlap}")
    st.caption(f"Top-k chunks: {settings.top_k}")

# ---------- Main page ----------
st.title("Policy Assistant")
st.caption("Answers come only from the documents in data/docs, with citations.")

examples = [
    "How many sick days do I get?",
    "Can I work remotely?",
    "What does the security policy say about passwords?",
]
cols = st.columns(len(examples))
for col, q in zip(cols, examples):
    if col.button(q, use_container_width=True):
        st.session_state.pending = q

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_sources(m.get("sources"))

typed = st.chat_input("Ask a question about the documents")
question = typed or st.session_state.pop("pending", None)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and writing the answer..."):
                result = rag.ask(question)
            st.markdown(result.text)
            render_sources(result.sources)
            st.session_state.messages.append(
                {"role": "assistant", "content": result.text, "sources": result.sources}
            )
        except Exception as e:
            msg = str(e)
            if "429" in msg:
                friendly = "Too many requests to the free Gemini tier. Wait a minute and try again."
            else:
                friendly = f"Something went wrong: {msg[:300]}"
            st.error(friendly)
            st.session_state.messages.append({"role": "assistant", "content": friendly})
