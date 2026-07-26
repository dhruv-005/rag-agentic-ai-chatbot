import sys
import numpy as np
from pathlib import Path
from groq import Groq
import chromadb
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from graph.state import RAGState, RetrievedChunk
from graph.prompts import (
    ANSWER_GENERATION_PROMPT,
    SELF_CHECK_PROMPT,
    NO_ANSWER_RESPONSE,
)


# ── cached loaders so models load only once ───────────────────────
@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """
    loads once and stays in memory
    st.cache_resource persists across reruns
    so we never reload the model twice
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(settings.embedding_model)
    return model


@st.cache_resource(show_spinner=False)
def get_chroma_collection():
    """
    opens chromadb connection once and reuses it
    """
    client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir
    )
    collection = client.get_collection(
        settings.chroma_collection_name
    )
    return collection


@st.cache_resource(show_spinner=False)
def get_groq_client():
    """
    creates groq client once and reuses it
    """
    return Groq(api_key=settings.groq_api_key)


# ── graph nodes ───────────────────────────────────────────────────
def retrieve_node(state: RAGState) -> RAGState:
    """
    embed the question and search chromadb
    for the most relevant chunks
    """
    question = state["question"]

    model = get_embedding_model()
    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
    ).tolist()[0]

    try:
        collection = get_chroma_collection()
    except Exception:
        return {
            **state,
            "retrieved_chunks": [],
            "best_score": 0.0,
        }

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=settings.top_k_results,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    best_score = 0.0

    if results and results["documents"] and results["documents"][0]:
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, distances):
            score = float(1 - dist)
            score = max(0.0, min(1.0, score))

            chunk: RetrievedChunk = {
                "text": doc,
                "page": meta.get("page_number", 0),
                "score": round(score, 4),
                "chunk_id": meta.get("chunk_id", ""),
            }
            chunks.append(chunk)

            if score > best_score:
                best_score = score

    return {
        **state,
        "retrieved_chunks": chunks,
        "best_score": round(best_score, 4),
    }


def grade_relevance_node(state: RAGState) -> RAGState:
    """
    check if best score clears the threshold
    set route field for the conditional edge
    """
    best_score = state.get("best_score", 0.0)

    route = (
        "generate"
        if best_score >= settings.relevance_threshold
        else "no_answer"
    )

    return {**state, "route": route}


def generate_node(state: RAGState) -> RAGState:
    """
    build grounded prompt from chunks and call groq
    then run a quick self check for hallucination
    """
    question = state["question"]
    chunks = state["retrieved_chunks"]

    context_parts = [
        f"[Page {c['page']}]: {c['text']}"
        for c in chunks
    ]
    context_text = "\n\n".join(context_parts)

    prompt = ANSWER_GENERATION_PROMPT.format(
        context=context_text,
        question=question,
    )

    client = get_groq_client()

    answer_resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise assistant that only answers "
                    "from provided context. Never guess or hallucinate."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=800,
    )

    answer_text = answer_resp.choices[0].message.content.strip()

    check_prompt = SELF_CHECK_PROMPT.format(
        question=question,
        context=context_text,
        answer=answer_text,
    )

    check_resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "user", "content": check_prompt}
        ],
        temperature=0.0,
        max_tokens=5,
    )

    check_text = (
        check_resp.choices[0].message.content.strip().lower()
    )
    self_check_passed = check_text.startswith("yes")

    avg_score = float(
        np.mean([c["score"] for c in chunks]) if chunks else 0.0
    )
    multiplier = 1.0 if self_check_passed else 0.6
    confidence = round(avg_score * multiplier, 2)

    return {
        **state,
        "answer": answer_text,
        "confidence": confidence,
        "self_check_passed": self_check_passed,
    }


def no_answer_node(state: RAGState) -> RAGState:
    """
    fallback when nothing relevant was retrieved
    """
    return {
        **state,
        "answer": NO_ANSWER_RESPONSE,
        "confidence": 0.0,
        "self_check_passed": False,
    }


def route_after_grading(state: RAGState) -> str:
    """
    used as conditional edge function in langgraph
    """
    return state.get("route", "no_answer")
