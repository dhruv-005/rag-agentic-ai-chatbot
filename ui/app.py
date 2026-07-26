import sys
import os
import time
import streamlit as st
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


# ── must run before anything else ────────────────────────────────
def load_secrets():
    try:
        all_secrets = dict(st.secrets)
        for key, value in all_secrets.items():
            os.environ[key] = str(value)
    except Exception:
        pass


load_secrets()

# ── now safe to import settings ───────────────────────────────────
from config.settings import settings
from graph.rag_graph import run_query
from ingestion.ingest import run_ingestion


st.set_page_config(
    page_title="Agentic AI Chatbot",
    page_icon="🤖",
    layout="wide",
)


def get_groq_key():
    """
    try every possible way to get the groq key
    """
    # way 1 - from os environ
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key and key.startswith("gsk_"):
        return key

    # way 2 - directly from st.secrets
    try:
        key = str(st.secrets["GROQ_API_KEY"]).strip()
        if key and key.startswith("gsk_"):
            os.environ["GROQ_API_KEY"] = key
            return key
    except Exception:
        pass

    # way 3 - from st.secrets as dict
    try:
        key = str(st.secrets.get("GROQ_API_KEY", "")).strip()
        if key and key.startswith("gsk_"):
            os.environ["GROQ_API_KEY"] = key
            return key
    except Exception:
        pass

    return ""


def check_groq_key():
    key = get_groq_key()
    if not key:
        return False, "GROQ_API_KEY is empty"
    if not key.startswith("gsk_"):
        return False, f"Key format wrong starts with {key[:6]}"
    return True, f"Key found starts with {key[:8]}..."


def check_knowledge_base():
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
    key = get_groq_key()
    if not key:
        return False, "No API key found"
    try:
        from groq import Groq
        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
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
        progress.progress(55)
        status.text("🔢 Generating embeddings...")
        progress.progress(75)
        status.text("💾 Saving to ChromaDB...")
        progress.progress(90)
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
    # make sure key is set before calling pipeline
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


# ── page header ───────────────────────────────────────────────────
st.title("🤖 Agentic AI RAG Chatbot")
st.caption(
    "Answers come strictly from the Agentic AI eBook "
    "with page citations and confidence scores."
)

# ── sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("📊 System Status")

    key_ok, key_msg = check_groq_key()
    is_populated, chunk_count = check_knowledge_base()

    if key_ok:
        st.success(f"✅ Groq key: {key_msg}")
    else:
        st.error(f"❌ Groq key issue: {key_msg}")
        st.warning(
            "Go to Streamlit Cloud\n"
            "→ Settings → Secrets\n"
            "Add your GROQ_API_KEY"
        )

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

    if st.button(
        "🔌 Test Groq Connection",
        use_container_width=True,
    ):
        with st.spinner("Testing Groq..."):
            ok, msg = test_groq_connection()
            if ok:
                st.success(f"✅ Groq works: {msg}")
            else:
                st.error(f"❌ Groq error: {msg}")

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
                time_ms = result.get("processing_time_ms", 0)

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
                            st.text(chunk["text"][:400] + "...")
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
