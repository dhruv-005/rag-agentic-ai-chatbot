import sys
import json
import time
from pathlib import Path

import httpx
import pytest

sys.path.append(str(Path(__file__).parent.parent))

API_BASE = "http://localhost:8000"

# load sample queries
queries_path = Path(__file__).parent / "sample_queries.json"
with open(queries_path) as f:
    SAMPLE_QUERIES = json.load(f)


def call_chat(question: str) -> dict:
    with httpx.Client(timeout=60) as client:
        response = client.post(
            f"{API_BASE}/chat",
            json={"question": question},
        )
        response.raise_for_status()
        return response.json()


def test_health_endpoint():
    with httpx.Client(timeout=10) as client:
        response = client.get(f"{API_BASE}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["index_populated"] is True
    print(f"\nHealth check passed — {data['total_chunks']} chunks in DB")


def test_chat_returns_required_fields():
    result = call_chat("What is Agentic AI?")
    assert "answer" in result
    assert "context_chunks" in result
    assert "confidence" in result
    assert "processing_time_ms" in result
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 10
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0


def test_context_chunks_have_required_fields():
    result = call_chat("What is Agentic AI?")
    chunks = result["context_chunks"]
    assert len(chunks) > 0
    for chunk in chunks:
        assert "text" in chunk
        assert "page" in chunk
        assert "score" in chunk
        assert isinstance(chunk["score"], float)
        assert isinstance(chunk["page"], int)


def test_empty_question_returns_400():
    with httpx.Client(timeout=10) as client:
        response = client.post(
            f"{API_BASE}/chat",
            json={"question": ""},
        )
    assert response.status_code == 400


def test_all_sample_queries():
    results_path = Path(__file__).parent / "eval_results.md"
    lines = [
        "# Evaluation Results\n\n",
        f"Run at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
    ]

    for item in SAMPLE_QUERIES:
        question = item["question"]
        print(f"\nTesting: {question}")

        start = time.time()
        result = call_chat(question)
        elapsed = int((time.time() - start) * 1000)

        answer = result["answer"]
        confidence = result["confidence"]
        chunks = result["context_chunks"]

        assert isinstance(answer, str)
        assert len(answer) > 10

        lines.append(f"## Query {item['id']}\n\n")
        lines.append(f"**Question:** {question}\n\n")
        lines.append(f"**Answer:** {answer}\n\n")
        lines.append(f"**Confidence:** {confidence:.2f}\n\n")
        lines.append(f"**Time:** {elapsed}ms\n\n")
        lines.append(f"**Chunks retrieved:** {len(chunks)}\n\n")

        if chunks:
            lines.append("**Top chunk:**\n\n")
            top = chunks[0]
            lines.append(
                f"- Page {top['page']} | Score {top['score']:.3f}\n"
            )
            lines.append(f"- {top['text'][:200]}...\n\n")

        lines.append("---\n\n")
        print(f"  Confidence: {confidence:.2f} | Time: {elapsed}ms")

    with open(results_path, "w") as f:
        f.writelines(lines)

    print(f"\nResults saved to {results_path}")
