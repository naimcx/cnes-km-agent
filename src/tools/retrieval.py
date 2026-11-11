from langchain_core.tools import tool
from src.rag_llamaindex import get_qdrant_retriever
from src.rag_local import get_local_retriever
from src.config import RAG_MODE, RAG_DB

# ==============================================
# EXTERNAL SEARCH ENGINES
# ==============================================

def _format_docs(results) -> str:
    """Formats retrieved documents into a readable string with source, page, and score information."""
    if not results:
        return "No relevant documentation found."
    
    formatted = []
    for i, doc in enumerate(results):
        text = getattr(doc, "text", getattr(doc, "page_content", str(doc))).strip()
        metadata = getattr(doc, "metadata", {})
        source = metadata.get("file_name", metadata.get("file_path", "Manual"))
        page = metadata.get("page_label", metadata.get("page", "N/A"))
        score = f"{doc.score:.3f}" if hasattr(doc, "score") and doc.score is not None else "N/A"
        img = metadata.get("image_path", "")

        entry = f"[Source {i} - Page: {page} - Score: {score} - File: {source}]\n{text}"
        if img:
            entry += f"\n[Associated Page Image]: {img}"
        formatted.append(entry)

    return "\n\n---\n\n".join(formatted)


# ==============================================
# RETRIEVAL TOOLS
# ==============================================

@tool
def search_docs_hybrid(query: str) -> str:
    """
    PRIMARY TOOL. Search the knowledge base for relevant documents.
    """
    try:
        if RAG_DB == "qdrant":
            retriever = get_qdrant_retriever(similarity_top_k=5, mode=RAG_MODE, alpha=0.5)
            nodes = retriever.retrieve(query)
            return _format_docs(nodes)

        elif RAG_DB == "chroma":
            retriever = get_local_retriever(source_type="pdf", mode=RAG_MODE, backend="chroma", k=3)
            docs = retriever.invoke(query)
            return _format_docs(docs)

        elif RAG_DB == "faiss":
            retriever = get_local_retriever(source_type="text", mode=RAG_MODE, backend="faiss", k=2)
            docs = retriever.invoke(query)
            return _format_docs(docs)

    except Exception as e:
        return f"Error during document retrieval ({RAG_DB}/{RAG_MODE}): {e}"