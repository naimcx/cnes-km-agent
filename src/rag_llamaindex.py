from qdrant_client import QdrantClient
from llama_index.core import (
    VectorStoreIndex,
    Settings,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL

Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
Settings.llm = None # LLM managed by LangChain, not LlamaIndex


def get_vector_store():
    """Init Qdrant connection"""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25"
    )


def get_qdrant_retriever(similarity_top_k: int = 3, mode: str = "hybrid", alpha: float = 0.7):
    """
    Returns a retriever for the CNES knowledge base using LlamaIndex and Qdrant.
    - similarity_top_k: number of top documents to retrieve.
    - mode: 'hybrid', 'vector', or 'keyword' retrieval mode.
    - alpha: weight for hybrid retrieval (0.0 = pure keyword, 1.0 = pure vector).
    """
    vector_store = get_vector_store()
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=Settings.embed_model,
    )

    if mode == "vector":
        query_mode = "default"
        using_alpha = 1.0
    elif mode == "keyword":
        query_mode = "sparse"
        using_alpha = 0.0
    else:
        query_mode = "hybrid"
        using_alpha = alpha


    return index.as_retriever(
        vector_store_query_mode=query_mode,
        similarity_top_k=similarity_top_k,
        alpha=using_alpha,
    )
