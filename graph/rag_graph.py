import sys
from pathlib import Path
from langgraph.graph import StateGraph, END

sys.path.append(str(Path(__file__).parent.parent))

from graph.state import RAGState
from graph.nodes import (
    retrieve_node,
    generate_node,
    no_answer_node,
)


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("no_answer", no_answer_node)

    graph.set_entry_point("retrieve")

    # always go to generate after retrieve
    # no grading step blocking the flow
    graph.add_conditional_edges(
        "retrieve",
        lambda state: (
            "generate"
            if len(state.get("retrieved_chunks", [])) > 0
            else "no_answer"
        ),
        {
            "generate": "generate",
            "no_answer": "no_answer",
        },
    )

    graph.add_edge("generate", END)
    graph.add_edge("no_answer", END)

    return graph.compile()


pipeline = build_graph()


def run_query(question: str) -> dict:
    initial: RAGState = {
        "question": question,
        "retrieved_chunks": [],
        "best_score": 0.0,
        "answer": "",
        "confidence": 0.0,
        "self_check_passed": False,
        "route": "",
    }

    result = pipeline.invoke(initial)

    return {
        "answer": result["answer"],
        "context_chunks": result["retrieved_chunks"],
        "confidence": result["confidence"],
        "best_score": result["best_score"],
    }
