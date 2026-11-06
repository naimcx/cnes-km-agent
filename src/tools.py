from langchain_core.tools import tool
from src.rag_corpus import load_vectorstore
from langchain_community.utilities import SearxSearchWrapper
from wikipediaapi import Wikipedia

vectorstore = load_vectorstore()
search = SearxSearchWrapper(searx_host="http://localhost:8080")

@tool
def search_docs(query: str) -> str:
    """Searches the vector memory for relevant documents based on the query."""
    results = vectorstore.similarity_search(query, k=2)
    return "\n".join([doc.page_content for doc in results])

@tool
def search_web(query: str) -> str:
    """Searches the web using SearXNG."""
    try:
        return search.run(query)
    except Exception as e:
        return f"Error during web search: {e}"

@tool
def fetch_wikipedia(title: str) -> str:
    """Wikipedia search, returns summary of the article (≤ 1500 chars)."""
    wiki = Wikipedia(
        user_agent='CNES-KMAgent/0.0 (https://github.com/naimcx/cnes-km-agent)', # mandatory user_agent to avoid 403 error
        language='en'
    )
    page = wiki.page(title)
    return page.summary[:1500] if page.exists() else 'Article not found.'