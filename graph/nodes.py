import sys
import os
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


def get_ef():
    return embedding_functions.DefaultEmbeddingFunction()


def get_chroma_collection():
    try:
        ef = get_ef()
        client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )
        col = client.get_collection(
            name=settings.chroma_collection_name,
            embedding_function=ef,
        )
        count = col.count()
        print(f"Collection has {count} chunks")
        return col
    except Exception as e:
        print(f"ChromaDB error: {e}")
        return None


def get_groq_client():
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise ValueError("GROQ_API_KEY not set")
    return Groq(api_key=key)


def retrieve_node(state: RAGState) -> RAGState:
    question = state["question"]
    print(f"Retrieving for: {question}")

    col = get_chroma_collection()

    if col is None:
        print("No collection found")
        return {
            **state,
            "retrieved_chunks": [],
            "best_score": 0.0,
        }

    count = col.count()
    print(f"Collection count: {count}")

    if count == 0:
        print("Collection is empty")
        return {
            **state,
            "retrieved_chunks": [],
            "best_score": 0.0,
        }

    try:
        n = min(settings.top_k_results, count)
        results = col.query(
            query_texts=[question],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        print(f"Query error: {e}")
        return {
            **state,
            "retrieved_chunks": [],
            "best_score": 0.0,
        }

    chunks = []
    best_score = 0.0

    if results and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = float(max(0.0, min(1.0, 1 - dist)))
            chunk: RetrievedChunk = {
                "text": doc,
                "page": meta.get("page_number", 0),
                "score": round(score, 4),
                "chunk_id": meta.get("chunk_id", ""),
            }
            chunks.append(chunk)
            if score > best_score:
                best_score = score

    print(f"Retrieved {len(chunks)} chunks")
    print(f"Best score: {best_score}")

    return {
        **state,
        "retrieved_chunks": chunks,
        "best_score": round(best_score, 4),
    }


def grade_relevance_node(state: RAGState) -> RAGState:
    chunks = state.get("retrieved_chunks", [])
    best_score = state.get("best_score", 0.0)

    print(f"Grading: chunks={len(chunks)} score={best_score}")

    # if we have any chunks send to generate
    if len(chunks) > 0:
        route = "generate"
    else:
        route = "no_answer"

    print(f"Route: {route}")
    return {**state, "route": route}


def generate_node(state: RAGState) -> RAGState:
    question = state["question"]
    chunks = state["retrieved_chunks"]

    print(f"Generating answer for: {question}")
    print(f"Using {len(chunks)} chunks")

    context_parts = [
        f"[Page {c['page']}]: {c['text']}"
        for c in chunks
    ]
    context_text = "\n\n".join(context_parts)

    prompt = ANSWER_GENERATION_PROMPT.format(
        context=context_text,
        question=question,
    )

    try:
        client = get_groq_client()
    except ValueError as e:
        return {
            **state,
            "answer": str(e),
            "confidence": 0.0,
            "self_check_passed": False,
        }

    try:
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert assistant on "
                        "Agentic AI. Use the provided context "
                        "to give detailed accurate answers. "
                        "Always mention page numbers."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
            max_tokens=1000,
        )
        answer = resp.choices[0].message.content.strip()
        print(f"Generated answer length: {len(answer)}")
    except Exception as e:
        print(f"Groq error: {e}")
        return {
            **state,
            "answer": f"Error calling LLM: {str(e)}",
            "confidence": 0.0,
            "self_check_passed": False,
        }

    avg_score = float(
        np.mean([c["score"] for c in chunks])
        if chunks else 0.0
    )
    confidence = round(avg_score, 2)

    return {
        **state,
        "answer": answer,
        "confidence": confidence,
        "self_check_passed": True,
    }


def no_answer_node(state: RAGState) -> RAGState:
    print("No relevant chunks found going to no_answer")
    return {
        **state,
        "answer": NO_ANSWER_RESPONSE,
        "confidence": 0.0,
        "self_check_passed": False,
    }


def route_after_grading(state: RAGState) -> str:
    route = state.get("route", "no_answer")
    print(f"Routing to: {route}")
    return route
