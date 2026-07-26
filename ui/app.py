import sys
import os
import time
import streamlit as st
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def load_secrets():
    try:
        all_secrets = dict(st.secrets)
        for key, value in all_secrets.items():
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
        if key and key.startswith("gsk_"):
            os.environ["GROQ_API_KEY"] = key
            return key
    except Exception:
        pass
    return ""


def get_collection():
    import chromadb
    from chromadb.utils import embedding_functions
    ef = embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir
    )
    try:
        col = client.get_collection(
            name=settings.chroma_collection_name,
            embedding_function=ef,
        )
        return col, col.count()
    except Exception:
        return None, 0


def check_knowledge_base():
    col, count = get_collection()
    return count > 0, count


def do_ingest(rebuild=False):
    try:
        run_ingestion(rebuild=rebuild)
        return True, ""
    except Exception as e:
        return False, str(e)


def ask_question(question: str):
    key = get_groq_key()
    if key:
        os.environ["GROQ_API_KEY"] = key
    try:
        start = time.time()
        result = run_query(question)
        elapsed = int((time.time() - start) * 1000)
        result["processing_time_ms"] = elapsed
        return result, None
    except Exception as e:
        return None, str(e)


# ── page ──────────────────────────────────────────────────────────
st.title("Agentic AI RAG Chatbot")
st.caption(
    "Ask anything about the Agentic AI eBook. "
    "Answers include page citations and confidence scores."
)

key_ok = bool(get_groq_key())
is_populated, chunk_count = check_knowledge_base()

# ── sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("System Status")

    groq_key = get_groq_key()
    if groq_key:
        st.success(f"Groq key: {groq_key[:12]}...")
    else:
        st.error("Groq key missing")

    st.write(f"Chunks in DB: {chunk_count}")

    if chunk_count == 0:
        st.error("Database is empty")
    else:
        st.success(f"Database ready: {chunk_count} chunks")

    st.divider()

    # manual ingest button
    if st.button(
        "Load eBook Now",
        use_container_width=True,
        type="primary",
    ):
        with st.spinner("Running ingestion..."):
            ok, err = do_ingest(rebuild=False)
            if ok:
                st.success("Done!")
                st.rerun()
            else:
                st.error(f"Failed: {err}")

    if st.button(
        "Force Reload eBook",
        use_container_width=True,
    ):
        with st.spinner("Rebuilding..."):
            ok, err = do_ingest(rebuild=True)
            if ok:
                st.success("Done!")
                st.rerun()
            else:
                st.error(f"Failed: {err}")

    st.divider()

    # debug section to see what is really happening
    if st.button(
        "Show Debug Info",
        use_container_width=True,
    ):
        st.write("ChromaDB path:")
        st.code(settings.chroma_persist_dir)
        st.write("Collection name:")
        st.code(settings.chroma_collection_name)
        st.write("PDF path:")
        st.code(settings.pdf_local_path)
        st.write("PDF URL:")
        st.code(settings.pdf_url)
        st.write("Threshold:")
        st.code(settings.relevance_threshold)
        st.write("Top K:")
        st.code(settings.top_k_results)

        pdf_exists = Path(settings.pdf_local_path).exists()
        st.write(f"PDF file exists: {pdf_exists}")

        db_path = Path(settings.chroma_persist_dir)
        st.write(f"DB folder exists: {db_path.exists()}")

        if db_path.exists():
            files = list(db_path.rglob("*"))
            st.write(f"DB files count: {len(files)}")

    if st.button(
        "Test Query Direct",
        use_container_width=True,
    ):
        col, count = get_collection()
        if col and count > 0:
            try:
                results = col.query(
                    query_texts=["What is Agentic AI"],
                    n_results=3,
                )
                docs = results["documents"][0]
                dists = results["distances"][0]
                st.write(f"Query returned {len(docs)} chunks")
                for i, (doc, dist) in enumerate(
                    zip(docs, dists)
                ):
                    score = round(1 - dist, 4)
                    st.write(f"Chunk {i+1} score: {score}")
                    st.text(doc[:200])
            except Exception as e:
                st.error(f"Query failed: {e}")
        else:
            st.error("Collection empty or missing")

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


# ── auto ingest if empty ──────────────────────────────────────────
if chunk_count == 0:
    st.warning(
        "Knowledge base is empty. Loading eBook automatically..."
    )
    progress = st.progress(0)
    status = st.empty()

    try:
        status.text("Downloading PDF from konverge.ai...")
        progress.progress(10)

        status.text("Extracting and chunking text...")
        progress.progress(40)

        status.text("Generating embeddings and storing...")
        progress.progress(70)

        run_ingestion(rebuild=False)

        progress.progress(100)
        status.text("Knowledge base ready!")
        time.sleep(2)
        progress.empty()
        status.empty()
        st.rerun()

    except Exception as e:
        progress.empty()
        status.empty()
        st.error(f"Auto ingestion failed: {str(e)}")
        st.info(
            "Click Load eBook Now button in the sidebar"
        )


# ── chat history ──────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            confidence = meta.get("confidence", 0)
            time_ms = meta.get("processing_time_ms", 0)
            chunks = meta.get("chunks", [])

            if confidence >= 0.75:
                color = "green"
            elif confidence >= 0.5:
                color = "orange"
            else:
                color = "red"

            st.markdown(
                f"Confidence: :{color}[{confidence:.0%}]"
                f" | Time: {time_ms}ms"
            )

            if chunks:
                with st.expander("View Source Chunks"):
                    for i, chunk in enumerate(chunks, 1):
                        st.markdown(
                            f"Chunk {i} "
                            f"Page {chunk['page']} "
                            f"Score {chunk['score']:.3f}"
                        )
                        st.text(chunk["text"][:400] + "...")
                        st.divider()


# ── chat input ────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", None)
chat_disabled = chunk_count == 0 or not key_ok

user_input = st.chat_input(
    "Ask a question about Agentic AI...",
    disabled=chat_disabled,
)
question = prefill or user_input

if question:
    if chunk_count == 0:
        st.warning(
            "Knowledge base is still loading. Please wait."
        )
        st.stop()

    if not key_ok:
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
                    {
                        "role": "assistant",
                        "content": f"Error: {error}",
                    }
                )
            else:
                answer = result.get("answer", "")
                chunks = result.get("context_chunks", [])
                confidence = result.get("confidence", 0.0)
                time_ms = result.get(
                    "processing_time_ms", 0
                )

                st.markdown(answer)

                if confidence >= 0.75:
                    color = "green"
                elif confidence >= 0.5:
                    color = "orange"
                else:
                    color = "red"

                st.markdown(
                    f"Confidence: :{color}[{confidence:.0%}]"
                    f" | Time: {time_ms}ms"
                )

                if chunks:
                    with st.expander("View Source Chunks"):
                        for i, chunk in enumerate(chunks, 1):
                            st.markdown(
                                f"Chunk {i} "
                                f"Page {chunk['page']} "
                                f"Score {chunk['score']:.3f}"
                            )
                            st.text(
                                chunk["text"][:400] + "..."
                            )
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
