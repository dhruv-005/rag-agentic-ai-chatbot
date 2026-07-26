import sys
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings
from graph.rag_graph import run_query

app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description=(
        "Ask questions about Agentic AI and get answers "
        "strictly from the eBook with page citations."
    ),
    version="1.0.0",
)


# request and response models
class ChatRequest(BaseModel):
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is Agentic AI?"
            }
        }


class ContextChunk(BaseModel):
    text: str
    page: int
    score: float
    chunk_id: str


class ChatResponse(BaseModel):
    answer: str
    context_chunks: list[ContextChunk]
    confidence: float
    processing_time_ms: int


class HealthResponse(BaseModel):
    status: str
    index_populated: bool
    total_chunks: int


@app.get("/health", response_model=HealthResponse)
def health_check():
    try:
        client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir
        )
        collection = client.get_collection(
            settings.chroma_collection_name
        )
        count = collection.count()
        populated = count > 0
    except Exception:
        populated = False
        count = 0

    return HealthResponse(
        status="healthy",
        index_populated=populated,
        total_chunks=count,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    if len(question) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Question is too long. Keep it under 1000 characters.",
        )

    start_time = time.time()

    try:
        result = run_query(question)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}",
        )

    elapsed_ms = int((time.time() - start_time) * 1000)

    # build context chunk list for the response
    context_chunks = [
        ContextChunk(
            text=chunk["text"],
            page=chunk["page"],
            score=chunk["score"],
            chunk_id=chunk["chunk_id"],
        )
        for chunk in result["context_chunks"]
    ]

    return ChatResponse(
        answer=result["answer"],
        context_chunks=context_chunks,
        confidence=result["confidence"],
        processing_time_ms=elapsed_ms,
    )


@app.get("/")
def root():
    return {
        "message": "Agentic AI RAG Chatbot is running",
        "docs": "/docs",
        "health": "/health",
        "chat": "POST /chat",
    }
