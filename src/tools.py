from langchain_core.tools import tool
from src.rag_corpus import load_vectorstore

vectorstore = load_vectorstore()

@tool
def search_docs(query: str) -> str:
    """Searches the vector memory for relevant documents based on the query."""
    results = vectorstore.similarity_search(query, k=2)
    return "\n".join([doc.page_content for doc in results])