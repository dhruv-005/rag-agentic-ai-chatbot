from typing import TypedDict, List, Dict, Any


class RetrievedChunk(TypedDict):
    text: str
    page: int
    score: float
    chunk_id: str


class RAGState(TypedDict):
    # the question the user asked
    question: str

    # chunks pulled back from chromadb
    retrieved_chunks: List[RetrievedChunk]

    # best similarity score from retrieval
    best_score: float

    # the final answer text
    answer: str

    # numeric confidence 0.0 to 1.0
    confidence: float

    # did the self check pass
    self_check_passed: bool

    # which path the graph took
    route: str
