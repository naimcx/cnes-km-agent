from src.rag_llamaindex import get_qdrant_retriever
from src.rag_local import get_local_retriever

def test_qdrant_retriever():
    """Test the Qdrant retriever"""
    retriever = get_qdrant_retriever(similarity_top_k=2, mode="vector")
    nodes = retriever.retrieve("Vinci engine")
    assert len(nodes) > 0

def test_chroma_retriever():
    """Test the Chroma retriever"""
    retriever = get_local_retriever(source_type="pdf", mode="vector", backend="chroma", k=2)
    nodes = retriever.invoke("fairing")
    assert len(nodes) > 0

def test_faiss_retriever():
    """Test the Faiss retriever"""
    retriever = get_local_retriever(source_type="text", mode="vector", backend="faiss", k=2)
    nodes = retriever.invoke("Ariane 6")
    assert len(nodes) > 0
