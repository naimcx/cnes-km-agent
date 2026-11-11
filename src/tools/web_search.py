import httpx
from langchain_core.tools import tool
from src.config import SEARX_HOST
from wikipediaapi import Wikipedia

@tool
def search_web(query: str, max_results: int = 3) -> str:
    """Search the web using the local SearXNG instance."""
    try:
        url = f"{SEARX_HOST.rstrip('/')}/search"
        response = httpx.get(
            url,
            params={"q": query, "format": "json"},
            timeout=8.0
        )
        response.raise_for_status()
        results = response.json().get("results", [])[:max_results]
        
        if not results:
            return "No relevant web results found."
            
        formatted = []
        for r in results:
            formatted.append(f"Source: {r.get('title')}\nURL: {r.get('url')}\nContent: {r.get('content')}")
        return "\n\n---\n\n".join(formatted)
    except Exception as e:
        return f"Error during web search: {e}"

@tool
def fetch_wikipedia(title: str) -> str:
    """
    EXTERNAL TOOL. Returns a concise summary of a Wikipedia article for the given title.
    Use this tool for foundational knowledge, physics formulas, or mission context.
    """
    try:
        wiki = Wikipedia(
            user_agent='CNES-KMAgent/0.0 (https://github.com/naimcx/cnes-km-agent)', # mandatory user_agent to avoid 403 error
            language='en'
        )
        page = wiki.page(title)
        return page.summary[:1500] if page.exists() else 'Article not found.'
    except Exception as e:
        return f"Error during Wikipedia fetch: {e}"
