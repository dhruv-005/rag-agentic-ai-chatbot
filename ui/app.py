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

KNOWLEDGE_COUNT = 25

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


def get_collection_status():
    status = {
        "client_ok": False,
        "collection_exists": False,
        "chunk_count": 0,
        "error": "",
        "persist_path": settings.chroma_persist_dir,
        "path_exists": False,
        "files_in_path": [],
    }
    try:
        path = Path(settings.chroma_persist_dir)
        status["path_exists"] = path.exists()
        if path.exists():
            status["files_in_path"] = [
                str(f.name) for f in path.iterdir()
            ]
        import chromadb
        from chromadb.utils import embedding_functions
        ef = embedding_functions.DefaultEmbeddingFunction()
        client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )
        status["client_ok"] = True
        col = client.get_collection(
            name=settings.chroma_collection_name,
            embedding_function=ef,
        )
        status["collection_exists"] = True
        status["chunk_count"] = col.count()
    except Exception as e:
        status["error"] = str(e)
    return status


def do_ingest(rebuild=False):
    try:
        run_ingestion(rebuild=rebuild)
        return True, ""
    except Exception as e:
        return False, str(e)


def get_count():
    return get_collection_status()["chunk_count"]


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
    "Answers from the Agentic AI eBook "
    "with page citations and confidence scores."
)

groq_key = get_groq_key()
status = get_collection_status()
count = status["chunk_count"]

# auto ingest if empty
if count == 0:
    st.info("Setting up knowledge base. Please wait...")
    ok, err = do_ingest(rebuild=False)
    if ok:
        status = get_collection_status()
        count = status["chunk_count"]
        if count > 0:
            st.success(f"Ready with {count} chunks")
            time.sleep(1)
            st.rerun()
        else:
            st.error(
                "Ingestion ran but still 0 chunks. "
                "Check Diagnosis tab."
            )
    else:
        st.error(f"Ingestion failed: {err}")

# ── sidebar ───────────────────────────────────────────────────────
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
        "Force Reload Knowledge",
        use_container_width=True,
    ):
        with st.spinner("Reloading..."):
            ok, err = do_ingest(rebuild=True)
            if ok:
                st.success("Done!")
                st.rerun()
            else:
                st.error(f"Failed: {err}")

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


# ── tabs ──────────────────────────────────────────────────────────
main_tab, diag_tab = st.tabs(["Chat", "Diagnosis"])


# ── diagnosis tab ─────────────────────────────────────────────────
with diag_tab:
    st.header("Diagnosis")

    st.subheader("ChromaDB Status")
    st.json(status)

    st.subheader("Environment Info")
    st.write(f"Groq key present: {bool(groq_key)}")
    st.write(
        f"Groq key prefix: "
        f"{groq_key[:8] if groq_key else 'none'}"
    )
    st.write(f"Chroma path: {settings.chroma_persist_dir}")
    st.write(
        f"Collection: {settings.chroma_collection_name}"
    )
    st.write(f"Top K: {settings.top_k_results}")
    st.write(f"Threshold: {settings.relevance_threshold}")
    st.write(f"Knowledge items in code: {KNOWLEDGE_COUNT}")

    st.subheader("Test Direct ChromaDB Query")
    if st.button("Run Test Query", use_container_width=True):
        if count == 0:
            st.error("No chunks in database")
        else:
            try:
                import chromadb
                from chromadb.utils import embedding_functions
                ef = (
                    embedding_functions.DefaultEmbeddingFunction()
                )
                client = chromadb.PersistentClient(
                    path=settings.chroma_persist_dir
                )
                col = client.get_collection(
                    name=settings.chroma_collection_name,
                    embedding_function=ef,
                )
                results = col.query(
                    query_texts=["What is Agentic AI"],
                    n_results=3,
                )
                docs = results["documents"][0]
                dists = results["distances"][0]
                st.success(
                    f"Query returned {len(docs)} chunks"
                )
                for i, (doc, dist) in enumerate(
                    zip(docs, dists)
                ):
                    score = round(1 - dist, 4)
                    st.write(f"Chunk {i+1} score: {score}")
                    st.text(doc[:300])
                    st.divider()
            except Exception as e:
                st.error(f"Query failed: {e}")

    st.subheader("Test Groq Connection")
    if st.button("Test Groq", use_container_width=True):
        if not groq_key:
            st.error("No Groq key found")
        else:
            try:
                from groq import Groq
                client = Groq(api_key=groq_key)
                resp = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "user",
                            "content": "Say: working correctly",
                        }
                    ],
                    max_tokens=20,
                )
                reply = resp.choices[0].message.content
                st.success(f"Groq works: {reply}")
            except Exception as e:
                st.error(f"Groq failed: {e}")

    st.subheader("Manual Ingest Controls")
    c1, c2 = st.columns(2)
    with c1:
        if st.button(
            "Run Ingest",
            use_container_width=True,
        ):
            with st.spinner("Running..."):
                ok, err = do_ingest(rebuild=False)
                if ok:
                    new = get_count()
                    st.success(f"Done. Count: {new}")
                else:
                    st.error(f"Failed: {err}")
    with c2:
        if st.button(
            "Force Rebuild",
            use_container_width=True,
        ):
            with st.spinner("Rebuilding..."):
                ok, err = do_ingest(rebuild=True)
                if ok:
                    new = get_count()
                    st.success(f"Done. Count: {new}")
                else:
                    st.error(f"Failed: {err}")


# ── chat tab ──────────────────────────────────────────────────────
with main_tab:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if (
                msg["role"] == "assistant"
                and "meta" in msg
            ):
                meta = msg["meta"]
                c = meta.get("confidence", 0)
                t = meta.get("processing_time_ms", 0)
                color = (
                    "green" if c >= 0.75
                    else "orange" if c >= 0.5
                    else "red"
                )
                st.markdown(
                    f"Confidence: :{color}[{c:.0%}]"
                    f" | Time: {t}ms"
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
                            st.text(
                                ch["text"][:400] + "..."
                            )
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
            st.warning("Knowledge base is empty.")
            st.stop()
        if not groq_key:
            st.error("Groq key missing.")
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
                    chunks = result.get(
                        "context_chunks", []
                    )
                    confidence = result.get(
                        "confidence", 0.0
                    )
                    time_ms = result.get(
                        "processing_time_ms", 0
                    )

                    st.markdown(answer)

                    color = (
                        "green" if confidence >= 0.75
                        else "orange" if confidence >= 0.5
                        else "red"
                    )
                    st.markdown(
                        f"Confidence: :{color}"
                        f"[{confidence:.0%}]"
                        f" | Time: {time_ms}ms"
                    )

                    if chunks:
                        with st.expander(
                            "View Source Chunks"
                        ):
                            for i, ch in enumerate(
                                chunks, 1
                            ):
                                st.markdown(
                                    f"Chunk {i} "
                                    f"Page {ch['page']} "
                                    f"Score {ch['score']:.3f}"
                                )
                                st.text(
                                    ch["text"][:400] + "..."
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
