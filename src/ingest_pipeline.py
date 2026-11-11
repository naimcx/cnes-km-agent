import argparse
import hashlib
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict

# Add the parent directory to sys.path to allow imports from the src package
sys.path.append(str(Path(__file__).resolve().parent.parent))

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config import DATA_PATH, EMBEDDING_MODEL, PDF_PARSER_TYPE
from src.rag_llamaindex import get_vector_store
from src.services.pdf_engine import get_pdf_parser
from src.services.qdrant_service import (
    get_qdrant_client,
    init_qdrant_schema,
    get_indexed_files_map,
    delete_file_points,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IngestPipeline")

# Selection of the engine via environment variable
parser_engine = get_pdf_parser(PDF_PARSER_TYPE)
logger.info(f"Active PDF Parser initialized: {type(parser_engine).__name__} (PDF_PARSER_TYPE={PDF_PARSER_TYPE})")


# ===========================================
# Helper Functions
# ===========================================

def calculate_sha256(file_path: Path) -> str:
    """Calculates the SHA-256 hash of a file in 64KB chunks."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


# ==========================================
# Worker Function for PDF Extraction
#==========================================

def worker_extract_pdf(task: Tuple[Path, str, str]) -> List[Document]:
    """Worker function to extract text from a PDF file using the configured parser engine."""
    pdf_path, pdf_str, file_hash = task
    try:
        return parser_engine.extract(pdf_path, file_hash)
    except Exception as e:
        logger.error(f"Extraction error on {pdf_path.name} with {PDF_PARSER_TYPE} : {e}")
        return []


#==========================================
# Ingestion Pipeline
#==========================================

def run_ingestion_pipeline(max_workers: int = 4, batch_size: int = 5, force: bool = False):
    data_dir = Path(DATA_PATH)
    if not data_dir.exists():
        logger.error(f"The source folder {DATA_PATH} does not exist.")
        return

    all_pdfs = list(data_dir.rglob("*.pdf"))
    if not all_pdfs:
        logger.info(f"No PDF files found in {DATA_PATH}.")
        return

    client = get_qdrant_client()
    init_qdrant_schema(client)
    indexed_map = get_indexed_files_map(client)

    # Clean up documents that have been removed from the disk
    current_disk_paths = {str(p.resolve()) for p in all_pdfs}
    deleted_paths = set(indexed_map.keys()) - current_disk_paths
    for d_path in deleted_paths:
        logger.info(f"Document removed from disk, deleting from Qdrant: {d_path}")
        delete_file_points(client, d_path)

    # Detect new or modified files by comparing their SHA-256 hashes with the stored values in Qdrant
    to_process = []
    for pdf in all_pdfs:
        pdf_str = str(pdf.resolve())
        current_hash = calculate_sha256(pdf)
        stored_hash = indexed_map.get(pdf_str)

        if force or stored_hash != current_hash:
            is_update = stored_hash is not None
            to_process.append((pdf, pdf_str, current_hash, is_update))

    if not to_process:
        logger.info("No new or modified PDF files detected for processing.")
        return

    logger.info(f"{len(to_process)} documents to process.")

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.llm = None

    vector_store = get_vector_store()
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=Settings.embed_model
    )
    parser = MarkdownNodeParser()

    total_lots = (len(to_process) + batch_size - 1) // batch_size
    for i in range(0, len(to_process), batch_size):
        batch = to_process[i:i + batch_size]
        current_lot = (i // batch_size) + 1

        for _, pdf_str, _, is_update in batch:
            if is_update:
                delete_file_points(client, pdf_str)

        tasks = [(item[0], item[1], item[2]) for item in batch]
        flat_docs = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker_extract_pdf, t) for t in tasks]
            for future in as_completed(futures):
                flat_docs.extend(future.result())

        if flat_docs:
            nodes = parser.get_nodes_from_documents(flat_docs)
            index.insert_nodes(nodes)
            logger.info(f"Lot {current_lot}/{total_lots} ingestion completed and indexed ({len(nodes)} chunks).")

    logger.info("Ingestion pipeline completed successfully.")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="PDF Ingestion Pipeline")
    arg_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction and image generation even if PDF hash is unchanged in Qdrant"
    )
    args = arg_parser.parse_args()
    run_ingestion_pipeline(force=args.force)
