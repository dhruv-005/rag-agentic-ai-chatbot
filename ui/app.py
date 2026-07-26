import sys
import requests
import streamlit as st
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Agentic AI Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agentic AI RAG Chatbot")
st.caption(
    "Ask anything about Agentic AI. "
    "Answers come strictly from the eBook with page citations."
)

# sidebar shows health info
with st.sidebar:
    st.header("System Status")

    try:
        health = requests.get(f"{API_BASE}/health", timeout=5).json()
        if health.get("index_populated"):
            st.success("✅ Knowledge base ready")
            st.write(f"Chunks stored: {health.get('total_chunks', 0)}")
        else:
            st.error("❌ Knowledge base empty — run ingestion first")
    except Exception:
        st.error("❌ API not reachable — start FastAPI first")

    st.divider()
    st.header("Sample Questions")
    sample_questions = [
        "What is Agentic AI?",
        "How does Agentic AI differ from traditional AI?",
        "What are the core components of an agentic system?",
        "What industries does the eBook mention?",
        "What are the risks of Agentic AI?",
        "What does the eBook say about the future of Agentic AI?",
    ]

    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state["prefill_question"] = q

# init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "chunks" in msg:
            confidence = msg.get("confidence", 0)
            color = (
                "green" if confidence >= 0.75
                else "orange" if confidence >= 0.5
                else "red"
            )
            st.markdown(
                f"**Confidence:** :{color}[{confidence:.0%}]"
            )

            with st.expander("📄 Source Chunks from eBook"):
                for i, chunk in enumerate(msg["chunks"], 1):
                    st.markdown(
                        f"**Chunk {i} — Page {chunk['page']} "
                        f"(Score: {chunk['score']:.3f})**"
                    )
                    st.text(chunk["text"][:400] + "...")
                    st.divider()

# handle prefilled question from sidebar buttons
prefill = st.session_state.pop("prefill_question", None)

user_input = st.chat_input(
    "Ask a question about Agentic AI...",
)

# use prefill if button was clicked otherwise use typed input
question = prefill or user_input

if question:
    # show user message
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    # call the api and show response
    with st.chat_message("assistant"):
        with st.spinner("Searching the eBook and generating answer..."):
            try:
                response = requests.post(
                    f"{API_BASE}/chat",
                    json={"question": question},
                    timeout=60,
                )
                data = response.json()

                answer = data.get("answer", "No answer returned.")
                chunks = data.get("context_chunks", [])
                confidence = data.get("confidence", 0.0)
                time_ms = data.get("processing_time_ms", 0)

                st.markdown(answer)

                color = (
                    "green" if confidence >= 0.75
                    else "orange" if confidence >= 0.5
                    else "red"
                )
                st.markdown(
                    f"**Confidence:** :{color}[{confidence:.0%}] "
                    f"| ⏱ {time_ms}ms"
                )

                with st.expander("📄 Source Chunks from eBook"):
                    for i, chunk in enumerate(chunks, 1):
                        st.markdown(
                            f"**Chunk {i} — Page {chunk['page']} "
                            f"(Score: {chunk['score']:.3f})**"
                        )
                        st.text(chunk["text"][:400] + "...")
                        st.divider()

                # save to history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "chunks": chunks,
                        "confidence": confidence,
                    }
                )

            except Exception as e:
                err_msg = (
                    f"Could not reach the API. "
                    f"Make sure FastAPI is running. Error: {e}"
                )
                st.error(err_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err_msg}
                )
