import sys
import numpy as np
from pathlib import Path
from groq import Groq
import chromadb
import streamlit as st
from chromadb.utils import embedding_functions

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from graph.state import RAGState, RetrievedChunk
from graph.prompts import (
    ANSWER_GENERATION_PROMPT,
    SELF_CHECK_PROMPT,
    NO_ANSWER_RESPONSE,
)


@st.cache_resource(show_spinner=False)
def get_embedding_function():
    """
    chromadb default embedding uses onnxruntime
    no torch no torchvision required
    loads once and stays cached
    """
    return embedding_functions.DefaultEmbeddingFunction()


@st.cache_resource(show_spinner=False)
def get_chroma_collection():
    """
    open chromadb connection once and reuse it
    """
    ef = get_embedding_function()
    client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir
    )
    collection = client.get_collection(
        name=settings.chroma_collection_name,
        embedding_function=ef,
    )
    return collection


@st.cache_resource(show_spinner=False)
def get_groq_client():
    """
    create groq client once and reuse across queries
    """
    return Groq(api_key=settings.groq_api_key)


def retrieve_node(state: RAGState) -> RAGState:
    """
    embed question using chromadb built in function
    then search for top k most relevant chunks
    """
    question = state["question"]

    try:
        collection = get_chroma_collection()
    except Exception:
        return {
            **state,
            "retrieved_chunks": [],
            "best_score": 0.0,
        }

    # chromadb handles embedding the query internally
    results = collection.query(
        query_texts=[question],
        n_results=settings.top_k_results,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    best_score = 0.0

    if (
        results
        and results["documents"]
        and results["documents"][0]
    ):
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
    check best score against threshold
    set route for conditional edge
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
    build grounded prompt and call groq llm
    then self check the answer for hallucinations
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
                    "You are a precise assistant. "
                    "Only answer from the provided context. "
                    "Never guess or make up information."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=800,
    )

    answer_text = (
        answer_resp.choices[0].message.content.strip()
    )

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
        np.mean([c["score"] for c in chunks])
        if chunks
        else 0.0
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
    fallback when retrieval finds nothing relevant
    """
    return {
        **state,
        "answer": NO_ANSWER_RESPONSE,
        "confidence": 0.0,
        "self_check_passed": False,
    }


def route_after_grading(state: RAGState) -> str:
    """
    conditional edge function for langgraph
    """
    return state.get("route", "no_answer")
