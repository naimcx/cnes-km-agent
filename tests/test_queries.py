import pytest
from src.tools import search_docs_hybrid
from src.graph import app

def test_retriever_returns_relevant_context():
    """Verifies that the ChromaDB retrieval extracts the correct passages."""
    context = search_docs_hybrid.invoke({"query": "Vulcain engine"})
    assert "Vulcain" in context or "cryogenic" in context

def test_graph_nominal_execution():
    """Verifies that the complete workflow executes and validates the answer."""
    initial_state = {
        "question": "Which engine is used for the upper stage of Ariane 6 ?",
    }
    result = app.invoke(initial_state)

    assert "final_answer" in result
    assert result.get("is_valid", False) is True
    assert "Vinci" in result["final_answer"]