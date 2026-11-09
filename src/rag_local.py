import os
from typing import Literal, List
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from src.config import (
    EMBEDDING_MODEL,
    CHROMA_PATH,
    DEFAULT_PDF_PATH,
)

# 1. Textual mock corpus for testing purposes
DEFAULT_TEXT_CORPUS = [
    Document(
        page_content="Ariane 6 uses the Vulcain 2.1 engine for its cryogenic main stage (liquid hydrogen and oxygen).",
        metadata={"source": "vulcain_datasheet", "page": "1"}
    ),
    Document(
        page_content="The upper stage of Ariane 6 carries the reusable Vinci engine, optimized for multiple orbital burns.",
        metadata={"source": "vinci_datasheet", "page": "2"}
    ),
    Document(
        page_content="Ariane 6 fairing usable diameter is 4570 mm at the base and 2000 mm at the upper cone tip.",
        metadata={"source": "fairing_manual", "page": "2-3"}
    )
]

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def load_documents_from_source(source_type: Literal["text", "pdf"] = "pdf", pdf_path: str = None) -> List[Document]:
    """
    Loads documents from the specified source type.
    If 'text', returns a predefined text corpus. 
    If 'pdf', loads and splits the specified PDF document into chunks.
    """
    if source_type == "text":
        return DEFAULT_TEXT_CORPUS

    target_pdf = Path(pdf_path) if pdf_path else Path(DEFAULT_PDF_PATH)

    if target_pdf.is_dir():
        pdf_files = list(target_pdf.glob("*.pdf"))
        target_pdf = pdf_files[0] if pdf_files else None

    if not target_pdf or not target_pdf.exists():
        print(f"[Warning] PDF '{target_pdf}' not found. Falling back to default text corpus.")
        return DEFAULT_TEXT_CORPUS

    loader = PyPDFLoader(str(target_pdf))
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    return splitter.split_documents(loader.load())

def get_local_retriever(
    source_type: Literal["text", "pdf"] = "pdf",
    mode: Literal["vector", "keyword", "hybrid"] = "hybrid",
    backend: Literal["chroma", "faiss"] = "chroma",
    k: int = 3,
    alpha: float = 0.5
):
    """
    Constructs a configurable local retriever.
    - mode: 'vector' (pure semantic), 'keyword' (BM25), or 'hybrid' (combination).
    - backend: 'faiss' (pure RAM, ultra fast) or 'chroma' (local persistent).
    """
    docs = load_documents_from_source(source_type)
    embeddings = get_embeddings()

    # 1. Keyword search mode using BM25
    if mode == "keyword":
        return BM25Retriever.from_documents(docs, k=k)

    # 2. Construct vectorstore based on the selected backend
    if backend == "faiss":
        vectorstore = FAISS.from_documents(docs, embeddings)
    else:
        vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=str(CHROMA_PATH))
    
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 3. Vector search mode
    if mode == "vector":
        return dense_retriever

    # 4. Hybrid search mode: combine dense and sparse retrievers
    sparse_retriever = BM25Retriever.from_documents(docs, k=k)
    return EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[alpha, 1.0 - alpha]
    )