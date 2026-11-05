import pytest
from src.tools import search_launcher_specs
from src.graph import app

def test_retriever_returns_relevant_context():
    """Verifies that the ChromaDB retrieval extracts the correct passages."""
    context = search_launcher_specs.invoke("Vulcain engine")
    assert "Vulcain" in context, "Vulcain engine should be present in the context."

def test_graph_nominal_execution():
    """Verifies that the complete workflow executes and validates the answer."""
    initial_state = {
        "question": "Which engine is used for the upper stage of Ariane 6 ?",
        "iteration": 0
    }
    result = app.invoke(initial_state)

    assert "final_answer" in result
    assert result["score"] >= 7, "The score of the critic must be at least 7 for a nominal question."
    assert "Vinci" in result["final_answer"], "The validated answer must mention the Vinci engine."