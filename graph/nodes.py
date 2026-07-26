import sys
import numpy as np
from pathlib import Path
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from graph.state import RAGState, RetrievedChunk
from graph.prompts import (
    ANSWER_GENERATION_PROMPT,
    SELF_CHECK_PROMPT,
    NO_ANSWER_RESPONSE,
)

# load these once when the module is imported
# so we dont reload them on every single query
_embedding_model = None
_chroma_collection = None
_groq_client = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(settings.embedding_model)
    return _embedding_model


def get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )
        _chroma_collection = client.get_collection(
            settings.chroma_collection_name
        )
    return _chroma_collection


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


def retrieve_node(state: RAGState) -> RAGState:
    """
    embed the user question then search chromadb for the
    most similar chunks and return them with their scores
    """
    question = state["question"]

    # embed the question using the same local model used during ingestion
    model = get_embedding_model()
    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
    ).tolist()[0]

    # search chromadb
    collection = get_chroma_collection()
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
            # chromadb returns cosine distance so convert to similarity
            similarity_score = float(1 - dist)
            similarity_score = max(0.0, min(1.0, similarity_score))

            chunk: RetrievedChunk = {
                "text": doc,
                "page": meta.get("page_number", 0),
                "score": round(similarity_score, 4),
                "chunk_id": meta.get("chunk_id", ""),
            }
            chunks.append(chunk)

            if similarity_score > best_score:
                best_score = similarity_score

    return {
        **state,
        "retrieved_chunks": chunks,
        "best_score": round(best_score, 4),
    }


def grade_relevance_node(state: RAGState) -> RAGState:
    """
    check if the best score is above the threshold
    just sets the route field so the conditional edge can use it
    """
    best_score = state.get("best_score", 0.0)

    if best_score >= settings.relevance_threshold:
        route = "generate"
    else:
        route = "no_answer"

    return {**state, "route": route}


def generate_node(state: RAGState) -> RAGState:
    """
    build a grounded prompt from the retrieved chunks and
    call groq llama to generate the final answer
    then run a self check to verify grounding
    """
    question = state["question"]
    chunks = state["retrieved_chunks"]

    # build the context block with page references
    context_parts = []
    for chunk in chunks:
        context_parts.append(
            f"[Page {chunk['page']}]: {chunk['text']}"
        )
    context_text = "\n\n".join(context_parts)

    # build the final prompt
    prompt = ANSWER_GENERATION_PROMPT.format(
        context=context_text,
        question=question,
    )

    groq_client = get_groq_client()

    # generate the answer
    answer_response = groq_client.chat.completions.create(
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

    answer_text = answer_response.choices[0].message.content.strip()

    # run self check to see if answer is grounded
    self_check_prompt = SELF_CHECK_PROMPT.format(
        question=question,
        context=context_text,
        answer=answer_text,
    )

    check_response = groq_client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "user", "content": self_check_prompt}
        ],
        temperature=0.0,
        max_tokens=5,
    )

    check_text = check_response.choices[0].message.content.strip().lower()
    self_check_passed = check_text.startswith("yes")

    # calculate confidence score
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
    fallback when no relevant chunks were found
    returns a polite message with zero confidence
    """
    return {
        **state,
        "answer": NO_ANSWER_RESPONSE,
        "confidence": 0.0,
        "self_check_passed": False,
    }


def route_after_grading(state: RAGState) -> str:
    """
    this function is used as the conditional edge in langgraph
    returns the name of the next node to go to
    """
    return state.get("route", "no_answer")
