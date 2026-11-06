from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Simulate a small corpus of technical documents (CNES / Ariane 6)
corpus = [
    Document(
        page_content="Ariane 6 uses the Vulcain 2.1 engine for its cryogenic main stage (liquid hydrogen and oxygen).",
        metadata={"source": "vulcain_datasheet"}
    ),
    Document(
        page_content="The upper stage of Ariane 6 carries the reusable engine Vinci, optimized for multiple orbits.",
        metadata={"source": "vinci_datasheet"}
    ),
    Document(
        page_content="The P120C solid rocket boosters are used by Ariane 6 (A62 with 2 boosters, A64 with 4 boosters) and Vega-C.",
        metadata={"source": "p120c_datasheet"}
    )
]

# 2. Embedding model: transforms text into mathematical vectors (runs locally on CPU)
def get_embedding_model():
    """Returns the embedding model used for vectorization."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Indexing in FAISS: creates the vector space
def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(corpus, embedding_model)