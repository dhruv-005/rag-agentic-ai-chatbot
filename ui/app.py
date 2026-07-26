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


def get_chunk_count():
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


def do_ingest(rebuild=False):
    try:
        run_ingestion(rebuild=rebuild)
        return True, ""
    except Exception as e:
        return False, str(e)


def ask_question(question):
    key = get_groq_key()
    if key:
        os.environ["GROQ_API_KEY"] = key
    try:
        start = time.time()
        result = run_query(question)
        result["processing_time_ms"] = int(
            (time.time() - start) * 1000
        )
        return result, None
    except Exception as e:
        return None, str(e)


# ── page ──────────────────────────────────────────────────────────
st.title("Agentic AI RAG Chatbot")
st.caption(
    "Answers strictly from the Agentic AI eBook "
    "with page citations and confidence scores."
)

groq_key = get_groq_key()
chunk_count = get_chunk_count()

# ── auto ingest ───────────────────────────────────────────────────
if chunk_count == 0:
    st.info(
        "Setting up knowledge base for the first time. "
        "Please wait 3 to 5 minutes..."
    )
    bar = st.progress(0)
    msg = st.empty()

    msg.text("Step 1 of 4 — Downloading PDF...")
    bar.progress(10)

    ok, err = do_ingest(rebuild=False)

    if ok:
        bar.progress(100)
        msg.text("Knowledge base ready!")
        time.sleep(1)
        bar.empty()
        msg.empty()
        chunk_count = get_chunk_count()
        st.rerun()
    else:
        bar.empty()
        msg.empty()
        st.error(f"Auto setup failed: {err}")
        st.warning(
            "Click Force Reload eBook in sidebar to retry"
        )

# ── sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("System Status")

    if groq_key:
        st.success(f"Groq ready: {groq_key[:12]}...")
    else:
        st.error("Groq key missing in secrets")

    if chunk_count > 0:
        st.success(f"Knowledge base ready")
        st.metric("Chunks", chunk_count)
    else:
        st.error("Knowledge base empty")

    st.divider()

    if st.button(
        "Force Reload eBook",
        use_container_width=True,
    ):
        with st.spinner("Reloading eBook..."):
            ok, err = do_ingest(rebuild=True)
            if ok:
                st.success("Done!")
                st.rerun()
            else:
                st.error(f"Failed: {err}")

    if st.button(
        "Refresh",
        use_container_width=True,
    ):
        st.rerun()

    st.divider()
    st.header("Sample Questions")

    samples = [
        "What is Agentic AI?",
        "How does Agentic AI differ from traditional AI?",
        "What are the core components of an agentic system?",
        "What industries does the eBook mention?",
        "What are the risks of Agentic AI?",
        "What does the eBook say about the future of Agentic AI?",
    ]

    for q in samples:
        if st.button(q, use_container_width=True):
            st.session_state["prefill"] = q

    st.caption("Built with LangGraph, Groq, and ChromaDB")


# ── chat ──────────────────────────────────────────────────────────
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
                            f"Chunk {i} "
                            f"Page {ch['page']} "
                            f"Score {ch['score']:.3f}"
                        )
                        st.text(ch["text"][:400] + "...")
                        st.divider()

prefill = st.session_state.pop("prefill", None)
disabled = chunk_count == 0 or not groq_key

user_input = st.chat_input(
    "Ask a question about Agentic AI...",
    disabled=disabled,
)
question = prefill or user_input

if question:
    if chunk_count == 0:
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
            result, error = ask_question(question)

            if error:
                st.error(f"Error: {error}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": error}
                )
            else:
                answer = result.get("answer", "")
                chunks = result.get("context_chunks", [])
                confidence = result.get("confidence", 0.0)
                time_ms = result.get("processing_time_ms", 0)

                st.markdown(answer)

                color = (
                    "green" if confidence >= 0.75
                    else "orange" if confidence >= 0.5
                    else "red"
                )
                st.markdown(
                    f"Confidence: :{color}[{confidence:.0%}]"
                    f" | Time: {time_ms}ms"
                )

                if chunks:
                    with st.expander("View Source Chunks"):
                        for i, ch in enumerate(chunks, 1):
                            st.markdown(
                                f"Chunk {i} "
                                f"Page {ch['page']} "
                                f"Score {ch['score']:.3f}"
                            )
                            st.text(ch["text"][:400] + "...")
                            st.divider()

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "meta": {
                            "confidence": confidence,
                            "processing_time_ms": time_ms,
                            "chunks": chunks,
                        },
                    }
                )
