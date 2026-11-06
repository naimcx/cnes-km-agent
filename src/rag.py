from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PATH = "./chroma_db_cnes"

def get_embedding_model():
    """Returns the embedding model used for vectorization."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_vectorstore():
    """Builds a vectorstore from the CNES PDF documents."""
    # Load documents from PDF files
    pdf_files = [
        "data/Mua-6_Issue-2_Revision-0_March-2021.pdf"
    ]
    docs = []
    
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs.extend(loader.load())
    
    # Split documents into smaller chunks for better embedding and retrieval
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    # Get the embedding model
    embedding_model = get_embedding_model()
    
    # Create a Chroma vectorstore from the document chunks
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )
    
    print(f"Vectorstore built and persisted at {CHROMA_PATH}.")
    return vectorstore

def load_vectorstore():
    """Loads the vectorstore from disk, or builds it if it doesn't exist."""
    try:
        vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_model())
        print(f"Vectorstore loaded from {CHROMA_PATH}.")
    except Exception as e:
        print(f"Failed to load vectorstore: {e}. Building a new one.")
        vectorstore = build_vectorstore()
    
    return vectorstore

if __name__ == "__main__":
    # Build the vectorstore when this script is run directly
    build_vectorstore()