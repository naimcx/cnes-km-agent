from langchain_groq import ChatGroq
from src.config import (
    GROQ_API_KEY,
    GROQ_TEXT_MODEL,
)

def test_llm_groq_connection():

    # Instantiate the LLM
    # temperature=0 ensures that the model's responses are consistent and reproducible.
    llm = ChatGroq(
        model_name=GROQ_TEXT_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0
    )
    
    response = llm.invoke("Hello, confirm that you are working correctly.")
    assert response.content != ""
    