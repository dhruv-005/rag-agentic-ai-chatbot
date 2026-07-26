import sys
from pathlib import Path
from langgraph.graph import StateGraph, END

sys.path.append(str(Path(__file__).parent.parent))

from graph.state import RAGState
from graph.nodes import (
    retrieve_node,
    grade_relevance_node,
    generate_node,
    no_answer_node,
    route_after_grading,
)


def build_rag_graph():
    """
    assemble the langgraph state machine

    flow:
      retrieve → grade_relevance → generate   (if score is good)
                                 → no_answer  (if score is too low)
    """
    graph = StateGraph(RAGState)

    # register all nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_relevance", grade_relevance_node)
    graph.add_node("generate", generate_node)
    graph.add_node("no_answer", no_answer_node)

    # set the entry point
    graph.set_entry_point("retrieve")

    # fixed edge from retrieve to grading
    graph.add_edge("retrieve", "grade_relevance")

    # conditional edge based on score threshold
    graph.add_conditional_edges(
        "grade_relevance",
        route_after_grading,
        {
            "generate": "generate",
            "no_answer": "no_answer",
        },
    )

    # both terminal nodes go to END
    graph.add_edge("generate", END)
    graph.add_edge("no_answer", END)

    return graph.compile()


# build once and reuse
rag_pipeline = build_rag_graph()


def run_query(question: str) -> dict:
    """
    run a question through the full rag pipeline
    and return everything the api or ui needs
    """
    initial_state: RAGState = {
        "question": question,
        "retrieved_chunks": [],
        "best_score": 0.0,
        "answer": "",
        "confidence": 0.0,
        "self_check_passed": False,
        "route": "",
    }

    final_state = rag_pipeline.invoke(initial_state)

    return {
        "answer": final_state["answer"],
        "context_chunks": final_state["retrieved_chunks"],
        "confidence": final_state["confidence"],
        "self_check_passed": final_state["self_check_passed"],
        "best_score": final_state["best_score"],
        "route_taken": final_state["route"],
    }
