import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

# ======================================================
# ENVIRONMENT LOADING & MODELS CONFIGURATION
# =====================================================

# Load environment variables from .env file
load_dotenv()

# Provider selection
TEXT_PROVIDER = os.getenv("TEXT_PROVIDER", "groq").lower()
VISION_PROVIDER = os.getenv("VISION_PROVIDER", "groq").lower()

# Model selection
GROQ_TEXT_MODEL = os.getenv("GROQ_TEXT_MODEL", "openai/gpt-oss-120b")
GROQ_VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "llama-3.2-11b-vision-preview")
LOCAL_TEXT_MODEL = os.getenv("LOCAL_TEXT_MODEL", "llama3.1:8b")
LOCAL_VISION_MODEL = os.getenv("LOCAL_VISION_MODEL", "minicpm-v")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Temperature and retry settings
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0))
VISION_TEMPERATURE = float(os.getenv("VISION_TEMPERATURE", 0))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", 3))

# Embedding model selection
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ======================================================
# LLM AND VISION MODEL INITIALIZATION
# =====================================================

# 1. Text Model
if TEXT_PROVIDER == "groq":
    llm = ChatGroq(
        model_name=os.getenv("GROQ_TEXT_MODEL", "openai/gpt-oss-120b"),
        temperature=float(os.getenv("LLM_TEMPERATURE", 0)),
        api_key=GROQ_API_KEY,
        max_retries=int(os.getenv("LLM_MAX_RETRIES", 3))
    )
else:
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=os.getenv("LOCAL_TEXT_MODEL", "llama3.1:8b"),
        temperature=float(os.getenv("LLM_TEMPERATURE", 0))
    )

# 2. Vision Model
if VISION_PROVIDER == "groq":
    vision_llm = ChatGroq(
        model_name=os.getenv("GROQ_VISION_MODEL", "llama-3.2-11b-vision-preview"),
        temperature=float(os.getenv("VISION_TEMPERATURE", 0)),
        api_key=GROQ_API_KEY
    )
else:
    vision_llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=os.getenv("LOCAL_VISION_MODEL", "minicpm-v"),
        temperature=float(os.getenv("VISION_TEMPERATURE", 0))
    )


# ======================================================
# RAG, DATABASE & STORAGE CONFIGURATION
# =====================================================

# Vector db selection and retrieval mode
RAG_DB = os.getenv("RAG_DB", "qdrant") # "qdrant" | "chroma_db" | "faiss"
RAG_MODE = os.getenv("RAG_MODE", "hybrid") # "hybrid" | "vector" | "keyword"

# Qdrant configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "cnes_km_agent")

# Web search
SEARX_HOST = os.getenv("SEARX_HOST", "http://localhost:8080")

# Path configurations for data, Chroma DB, and images
ROOT_DIR = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT_DIR / "prompts"
SKILLS_DIR = ROOT_DIR / "skills"

DATA_PATH = ROOT_DIR / os.getenv("DATA_PATH", "data")
CHROMA_PATH = ROOT_DIR / os.getenv("CHROMA_PATH", "chroma_db")
IMG_PATH = ROOT_DIR / os.getenv("IMG_PATH", "static/pdf_images")

# Ensure the image path exists
IMG_PATH.mkdir(parents=True, exist_ok=True)

# Default PDF path for testing
TEST_PDF_NAME = os.getenv("TEST_PDF_NAME", "Mua-6_Issue-2_Revision-0_March-2021.pdf")
DEFAULT_PDF_PATH = DATA_PATH / TEST_PDF_NAME

PDF_PARSER_TYPE = os.getenv("PDF_PARSER_TYPE", "docling") # "docling" | "pymupdf" | "unstructured"
UNSTRUCTURED_STRATEGY = os.getenv("UNSTRUCTURED_STRATEGY", "fast") # "fast" | "hi_res"
PRE_RENDER_IMAGES = os.getenv("PRE_RENDER_IMAGES", True) # True | False
