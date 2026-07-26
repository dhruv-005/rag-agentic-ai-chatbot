import sys
import os
import time
import streamlit as st
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def load_secrets():
    """
    pull all secrets from streamlit cloud secrets manager
    and push them into os.environ so settings.py can read them
    """
    try:
        keys = [
            "GROQ_API_KEY",
            "EMBEDDING_MODEL",
            "LLM_MODEL",
            "CHROMA_PERSIST_DIR",
            "CHROMA_COLLECTION_NAME",
            "TOP_K_RESULTS",
            "RELEVANCE_THRESHOLD",
            "CHUNK_SIZE",
            "CHUNK_OVERLAP",
            "PDF_URL",
            "PDF_LOCAL_PATH",
        ]
        for key in keys:
            if key in st.secrets:
                os.environ[key] = str(st.secrets[key])
    except Exception:
        pass


load_secrets()

from config.settings import settings
from graph.rag_graph import run_query
from ingestion.ingest import run_ingestion


st.set_page_config(
    page_title="Agentic AI Chatbot",
    page_icon="🤖",
    layout="wide",
)


def check_groq_key():
    """
    verify groq key exists and looks valid
    """
    key = settings.groq_api_key
    if not key:
        return False, "GROQ_API_KEY is empty"
    if not key.startswith("gsk_"):
        return False, f"Key format wrong — starts with: {key[:6]}"
    return True, f"Key found — starts with: {key[:8]}..."


def check_knowledge_base():
    """
    check if chromadb has data
    """
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
        count = col.count()
        return count > 0, count
    except Exception:
        return False, 0


def test_groq_connection():
    """
    make a tiny test call to groq to verify connection
    """
    try:
        from groq import Groq
        client = Groq(api_key=settings.groq_api_key)
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "user", "content": "Say OK"}
            ],
            max_tokens=5,
            temperature=0.0,
        )
        reply = resp.choices[0].message.content.strip()
        return True, reply
    except Exception as e:
        return False, str(e)


def load_ebook():
    progress = st.progress(0)
    status = st.empty()

    try:
        status.text("📥 Downloading PDF...")
        progress.progress(15)

        status.text("📄 Extracting text...")
        progress.progress(35)

        status.text("✂️ Chunking pages...")
        progress.progress(50)

        status.text("🔢 Generating embeddings...")
        progress.progress(70)

        status.text("💾 Saving to ChromaDB...")
        progress.progress(85)

        run_ingestion(rebuild=False)

        progress.progress(100)
        status.text("✅ Done!")
        time.sleep(1)
        progress.empty()
        status.empty()
        return True

    except Exception as e:
        progress.empty()
        status.empty()
        st.error(f"Ingestion failed: {str(e)}")
        return False


def ask_question(question: str):
    try:
        start = time.time()
        result = run_query(question)
        elapsed = int((time.time() - start) * 1000)
        result["processing_time_ms"] = elapsed
        return result, None
    except Exception as e:
        return None, str(e)


# ── page header ───────────────────────────────────────────────────
st.title("🤖 Agentic AI RAG Chatbot")
st.caption(
    "Answers come strictly from the Agentic AI eBook "
    "with page citations and confidence scores."
)


# ── sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 System Status")

    # check groq key
    key_ok, key_msg = check_groq_key()

    if key_ok:
        st.success(f"✅ Groq key: {key_msg}")
    else:
        st.error(f"❌ Groq key issue: {key_msg}")
        st.warning(
            "Go to Streamlit Cloud → Settings → Secrets\n"
            "Add: GROQ_API_KEY = \"gsk_your_key_here\""
        )

    # check knowledge base
    is_populated, chunk_count = check_knowledge_base()

    if not is_populated:
        st.warning("⚠️ Knowledge base is empty")

        if st.button(
            "📥 Load Agentic AI eBook",
            use_container_width=True,
            type="primary",
            disabled=not key_ok,
        ):
            with st.spinner("Loading eBook..."):
                success = load_ebook()
                if success:
                    st.success("✅ eBook loaded!")
                    st.rerun()
    else:
        st.success("✅ Knowledge base ready")
        st.metric("Chunks stored", chunk_count)

        if st.button(
            "🔄 Reload eBook",
            use_container_width=True,
        ):
            with st.spinner("Reloading..."):
                run_ingestion(rebuild=True)
                st.rerun()

    st.divider()

    # test groq connection button
    if st.button(
        "🔌 Test Groq Connection",
        use_container_width=True,
    ):
        with st.spinner("Testing..."):
            ok, msg = test_groq_connection()
            if ok:
                st.success(f"✅ Groq connected: {msg}")
            else:
                st.error(f"❌ Groq failed: {msg}")

    if st.button(
        "🔄 Refresh Status",
        use_container_width=True,
    ):
        st.rerun()

    st.divider()
    st.header("💡 Sample Questions")

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

    st.divider()
    st.caption("Built with LangGraph + Groq + ChromaDB")


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
                f"**Confidence:** :{color}[{confidence:.0%}]"
                f" | ⏱ {time_ms}ms"
            )

            if chunks:
                with st.expander("📄 View Source Chunks"):
                    for i, chunk in enumerate(chunks, 1):
                        st.markdown(
                            f"**Chunk {i} — Page {chunk['page']}"
                            f" | Score {chunk['score']:.3f}**"
                        )
                        st.text(chunk["text"][:400] + "...")
                        st.divider()


# ── chat input ────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", None)

chat_disabled = not is_populated or not key_ok

user_input = st.chat_input(
    "Ask a question about Agentic AI...",
    disabled=chat_disabled,
)
question = prefill or user_input


if question:
    if not key_ok:
        st.error(
            "Groq API key is missing. "
            "Add it in Streamlit Cloud Secrets."
        )
        st.stop()

    if not is_populated:
        st.warning(
            "Load the eBook first. "
            "Click 📥 Load Agentic AI eBook in sidebar."
        )
        st.stop()

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            result, error = ask_question(question)

            if error:
                st.error(f"Error: {error}")
                st.info(
                    "If this says Connection error — "
                    "check your Groq API key in Secrets."
                )
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
                    f"**Confidence:** :{color}[{confidence:.0%}]"
                    f" | ⏱ {time_ms}ms"
                )

                if chunks:
                    with st.expander("📄 View Source Chunks"):
                        for i, chunk in enumerate(chunks, 1):
                            st.markdown(
                                f"**Chunk {i} — "
                                f"Page {chunk['page']}"
                                f" | Score {chunk['score']:.3f}**"
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
