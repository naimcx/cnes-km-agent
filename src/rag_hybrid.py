from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

def get_hybrid_retriever_prototype(docs: list[Document], k: int = 2):
    """
    Prototype local hybrid retriever using Chroma for vector search and BM25 (in memory) for keyword search.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 1. Create a Chroma vectorstore from the provided documents
    vectorstore = Chroma.from_documents(docs, embedding_model)
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # 2. Create a BM25 retriever from the provided documents
    bm25_retriever = BM25Retriever.from_documents(docs, k=k)

    # 3. Combine the two retrievers into an ensemble retriever with 50% weight for each
    hybrid_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    return hybrid_retriever

def get_hybrid_retriever_production(docs: list[Document], k: int = 2):
    """
    For industrial use, using Qdrant
    """
    # in a prod environment
    # from langchain_qdrant import Qdrant
    # return Qdrant.from_documents(
    # docs, 
    # embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")).as_retriever(search_kwargs={"k": k})

    print("Production retriever not implemented yet. Using prototype retriever instead.")
    return get_hybrid_retriever_prototype(docs, k=k)


# ============================================================================== 
# Using OpenSearch for semantic search and keyword search (not implemented yet)
# ==============================================================================