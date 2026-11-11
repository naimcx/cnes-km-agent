from .aerospace_calculator import calculate_aerospace_math
from .retrieval import search_docs_hybrid
from .visual_inspector import inspect_pdf_page_image
from .web_search import search_web, fetch_wikipedia

INTERNAL_TOOLS = [search_docs_hybrid, inspect_pdf_page_image, calculate_aerospace_math]
EXTERNAL_TOOLS = [search_web, fetch_wikipedia]
ALL_TOOLS = [
    search_docs_hybrid,
    inspect_pdf_page_image,
    search_web,
    fetch_wikipedia,
    calculate_aerospace_math,
]
