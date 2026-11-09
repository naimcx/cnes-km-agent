import glob
import base64
import pymupdf as fitz
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_community.utilities import SearxSearchWrapper
from wikipediaapi import Wikipedia

from src.rag_local import get_local_retriever
from src.rag_llamaindex import get_qdrant_retriever
from src.config import (
    vision_llm,
    SEARX_HOST,
    RAG_DB,
    RAG_MODE,
    DATA_PATH,
    IMG_PATH
)

# ==============================================
# EXTERNAL SEARCH ENGINES
# ==============================================

# Initialize SearXNG search wrapper for web search
search = SearxSearchWrapper(searx_host=SEARX_HOST)


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
    PRIMARY TOOL. Searches the CNES technical documentation.
    """
    try:
        if RAG_DB == "qdrant":
            retriever = get_qdrant_retriever(similarity_top_k=3, mode=RAG_MODE, alpha=0.7)
            nodes = retriever.retrieve(query)
            return _format_docs(nodes)

        elif RAG_DB == "chroma":
            retriever = get_local_retriever(source_type="pdf", search_mode=RAG_MODE, backend="chroma", k=3)
            docs = retriever.invoke(query)
            return _format_docs(docs)

        elif RAG_DB == "faiss":
            retriever = get_local_retriever(source_type="text", search_mode=RAG_MODE, backend="faiss", k=2)
            docs = retriever.invoke(query)
            return _format_docs(docs)

    except Exception as e:
        return f"Error during document retrieval ({RAG_DB}/{RAG_MODE}): {e}"

@tool
def search_web(query: str) -> str:
    """
    EXTERNAL TOOL. Searches the web using SearXNG and returns top results.
    Use this tool for general knowledge, physics, or competitor launchers.
    """
    try:
        return search.run(query)
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


# ==============================================
# MULTIMODAL VISION TOOL
# ==============================================

@tool
def inspect_pdf_page_image(page_number: int, question: str) -> str:
    """
    Inspects a PDF page technical diagram or drawing using a vision model.
    Use this tool whenever dimensions, diameter symbols (Ø), or curves are inside a diagram.
    """
    try:
        page_num = int(page_number)
        img_bytes = None

        # 1. Look for an existing PNG image of the page in the static directory
        if IMG_PATH.exists():
            matches = list(IMG_PATH.glob(f"*_p{page_num}.png"))
            if matches:
                img_bytes = matches[0].read_bytes()

        # 2. Fallback: Render the page from the first PDF in the data directory if no image is found
        if not img_bytes:
            pdf_files = list(DATA_PATH.rglob("*.pdf")) if DATA_PATH.exists() else []
            if not pdf_files:
                pdf_files = [Path(p) for p in glob.glob("data/**/*.pdf", recursive=True)]
            if not pdf_files:
                return f"Error: No PDF files found in {DATA_PATH} or data/."

            # Open primary document and extract page pixmap at 200 DPI
            doc = fitz.open(str(pdf_files[0]))
            page_idx = max(0, min(page_num - 1, len(doc) - 1))
            page = doc.load_page(page_idx)
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            doc.close()

        # Encode the image to base64 for embedding in the prompt
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        prompt = (
            f"You are an aerospace engineer examining an official launcher drawing on page {page_num}.\n"
            f"Question: {question}\n"
            "SYSTEMATIC VISUAL EXTRACTION RULES:\n"
            "1. Identify the type of visual: CAD dimensional drawing, exploded hardware view, curve plot, or photo.\n"
            "2. Extract all visible callouts, labels, legends, and captions verbatim.\n"
            "3. For dimensional drawings, carefully transcribe all diameter symbols (Ø), radius, and vertical/horizontal dimensions (in mm or m).\n"
            "4. For hardware schematics, list every annotated part, quantity (e.g., 'x32', '2 doors'), and functional name.\n"
            "5. For plots/graphs, extract axis titles, units, limits, and curve levels.\n"
            "6. Zero-hallucination guarantee: Only report what is clearly visible on the image. If a dimension, value, or formula is not visibly printed, explicitly state: 'Not shown on the visual'.\n"
        )

        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
        ])

        response = vision_llm.invoke([message])
        return f"[Visual Inspection Page {page_num}]:\n{response.content}"

    except Exception as e:
        return f"Error during visual inspection: {e}"
