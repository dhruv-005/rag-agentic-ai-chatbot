import sys
import os
import time
import streamlit as st
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def load_secrets():
    try:
        for key, value in dict(st.secrets).items():
            os.environ[key] = str(value)
    except Exception:
        pass


load_secrets()

from config.settings import settings
from graph.rag_graph import run_query
from ingestion.ingest import run_ingestion


st.set_page_config(
    page_title="Agentic AI Chatbot",
    layout="wide",
)


def get_groq_key():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key and key.startswith("gsk_"):
        return key
    try:
        key = str(st.secrets["GROQ_API_KEY"]).strip()
        os.environ["GROQ_API_KEY"] = key
        return key
    except Exception:
        pass
    return ""


def get_count():
    try:
        import chromadb
        from chromadb.utils import embedding_functions
        ef = embedding_functions.DefaultEmbeddingFunction()
        client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )
        col = client.get_collection(
            name=settings.chroma_collection_name,
            embedding_function=ef,
        )
        return col.count()
    except Exception:
        return 0


def force_ingest():
    try:
        run_ingestion(rebuild=False)
        return get_count()
    except Exception as e:
        st.error(f"Ingest error: {e}")
        return 0


st.title("Agentic AI RAG Chatbot")
st.caption(
    "Answers from the Agentic AI eBook "
    "with page citations and confidence scores."
)

groq_key = get_groq_key()
count = get_count()

# always ingest if empty
if count == 0:
    with st.spinner(
        "Setting up knowledge base. Please wait..."
    ):
        count = force_ingest()
    if count > 0:
        st.success(f"Knowledge base ready with {count} chunks")
        st.rerun()
    else:
        st.error(
            "Knowledge base setup failed. "
            "Click Force Reload in sidebar."
        )

with st.sidebar:
    st.header("System Status")

    if groq_key:
        st.success(f"Groq: {groq_key[:10]}...")
    else:
        st.error("Groq key missing")

    if count > 0:
        st.success("Knowledge base ready")
        st.metric("Chunks", count)
    else:
        st.error("Knowledge base empty")

    st.divider()

    if st.button(
        "Force Reload eBook",
        use_container_width=True,
    ):
        with st.spinner("Reloading..."):
            run_ingestion(rebuild=True)
            st.rerun()

    if st.button("Refresh", use_container_width=True):
        st.rerun()

    st.divider()
    st.header("Sample Questions")

    for q in [
        "What is Agentic AI?",
        "How does Agentic AI differ from traditional AI?",
        "What are the core components of an agentic system?",
        "What industries does the eBook mention?",
        "What are the risks of Agentic AI?",
        "What does the eBook say about the future of Agentic AI?",
    ]:
        if st.button(q, use_container_width=True):
            st.session_state["prefill"] = q

    st.caption("Built with LangGraph, Groq, and ChromaDB")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            c = meta.get("confidence", 0)
            t = meta.get("processing_time_ms", 0)
            color = (
                "green" if c >= 0.75
                else "orange" if c >= 0.5
                else "red"
            )
            st.markdown(
                f"Confidence: :{color}[{c:.0%}] | Time: {t}ms"
            )
            chunks = meta.get("chunks", [])
            if chunks:
                with st.expander("View Source Chunks"):
                    for i, ch in enumerate(chunks, 1):
                        st.markdown(
                            f"Chunk {i} Page {ch['page']} "
                            f"Score {ch['score']:.3f}"
                        )
                        st.text(ch["text"][:400] + "...")
                        st.divider()

prefill = st.session_state.pop("prefill", None)
disabled = count == 0 or not groq_key

user_input = st.chat_input(
    "Ask a question about Agentic AI...",
    disabled=disabled,
)
question = prefill or user_input

if question:
    if count == 0:
        st.warning("Knowledge base loading. Please wait.")
        st.stop()
    if not groq_key:
        st.error("Groq API key missing.")
        st.stop()

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            try:
                start = time.time()
                result = run_query(question)
                ms = int((time.time() - start) * 1000)

                answer = result.get("answer", "")
                chunks = result.get("context_chunks", [])
                confidence = result.get("confidence", 0.0)

                st.markdown(answer)

                color = (
                    "green" if confidence >= 0.75
                    else "orange" if confidence >= 0.5
                    else "red"
                )
                st.markdown(
                    f"Confidence: :{color}[{confidence:.0%}]"
                    f" | Time: {ms}ms"
                )

                if chunks:
                    with st.expander("View Source Chunks"):
                        for i, ch in enumerate(chunks, 1):
                            st.markdown(
                                f"Chunk {i} Page {ch['page']} "
                                f"Score {ch['score']:.3f}"
                            )
                            st.text(ch["text"][:400] + "...")
                            st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "meta": {
                        "confidence": confidence,
                        "processing_time_ms": ms,
                        "chunks": chunks,
                    },
                })

            except Exception as e:
                st.error(f"Error: {e}")
