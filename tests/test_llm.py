from src.config import llm

def test_active_llm_connection():
    """Test the connection to the active LLM by sending a simple prompt and checking for a valid response."""
    response = llm.invoke("Hello, confirm that you are working correctly.")
    assert response.content != ""
    