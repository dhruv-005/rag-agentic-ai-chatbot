import sys
import os
import numpy as np
from pathlib import Path
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from graph.state import RAGState, RetrievedChunk
from graph.prompts import (
    ANSWER_GENERATION_PROMPT,
    NO_ANSWER_RESPONSE,
)


def get_collection():
    try:
        ef = embedding_functions.DefaultEmbeddingFunction()
        client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )
        col = client.get_collection(
            name=settings.chroma_collection_name,
            embedding_function=ef,
        )
        count = col.count()
        print(f"Collection count: {count}")
        return col, count
    except Exception as e:
        print(f"Collection error: {e}")
        return None, 0


def retrieve_node(state: RAGState) -> RAGState:
    question = state["question"]
    print(f"Retrieving for: {question}")

    col, count = get_collection()

    if col is None or count == 0:
        print("No collection or empty")

        # try to auto ingest
        try:
            from ingestion.ingest import run_ingestion
            print("Auto ingesting now")
            run_ingestion(rebuild=False)
            col, count = get_collection()
            print(f"After ingest count: {count}")
        except Exception as e:
            print(f"Auto ingest failed: {e}")

        if col is None or count == 0:
            return {
                **state,
                "retrieved_chunks": [],
                "best_score": 0.0,
            }

    try:
        n = min(8, count)
        results = col.query(
            query_texts=[question],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        print(f"Query results: {len(results['documents'][0])}")
    except Exception as e:
        print(f"Query error: {e}")
        return {
            **state,
            "retrieved_chunks": [],
            "best_score": 0.0,
        }

    chunks = []
    best_score = 0.0

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = float(max(0.0, min(1.0, 1 - dist)))
        chunks.append({
            "text": doc,
            "page": meta.get("page_number", 0),
            "score": round(score, 4),
            "chunk_id": meta.get("chunk_id", ""),
        })
        if score > best_score:
            best_score = score

    print(f"Got {len(chunks)} chunks best score {best_score}")

    return {
        **state,
        "retrieved_chunks": chunks,
        "best_score": round(best_score, 4),
    }


def generate_node(state: RAGState) -> RAGState:
    question = state["question"]
    chunks = state["retrieved_chunks"]

    print(f"Generating answer using {len(chunks)} chunks")

    context = "\n\n".join([
        f"[Page {c['page']}]: {c['text']}"
        for c in chunks
    ])

    prompt = ANSWER_GENERATION_PROMPT.format(
        context=context,
        question=question,
    )

    key = os.environ.get("GROQ_API_KEY", "").strip()
    print(f"Key available: {bool(key)} starts: {key[:8] if key else 'none'}")

    if not key:
        return {
            **state,
            "answer": "Groq API key is missing from secrets.",
            "confidence": 0.0,
            "self_check_passed": False,
        }

    try:
        client = Groq(api_key=key)
        print("Calling Groq API")

        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert on Agentic AI. "
                        "Give detailed helpful answers using "
                        "the context provided. Always mention "
                        "page numbers in your answer."
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
        print(f"Answer length: {len(answer)}")

        avg = float(
            np.mean([c["score"] for c in chunks])
            if chunks else 0.5
        )
        confidence = round(max(avg, 0.5), 2)

        return {
            **state,
            "answer": answer,
            "confidence": confidence,
            "self_check_passed": True,
        }

    except Exception as e:
        print(f"Groq error: {e}")
        return {
            **state,
            "answer": f"LLM error: {str(e)}",
            "confidence": 0.0,
            "self_check_passed": False,
        }


def no_answer_node(state: RAGState) -> RAGState:
    print("No chunks found going to no answer")
    return {
        **state,
        "answer": NO_ANSWER_RESPONSE,
        "confidence": 0.0,
        "self_check_passed": False,
    }


def route_after_grading(state: RAGState) -> str:
    return state.get("route", "no_answer")
