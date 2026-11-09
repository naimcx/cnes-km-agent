import hashlib
import logging
import os
import re
import shutil
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict

# Add the parent directory to sys.path to allow imports from the src package
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pymupdf
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from src.config import DATA_PATH, EMBEDDING_MODEL, QDRANT_HOST, COLLECTION_NAME, IMG_PATH
from src.rag_llamaindex import get_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IngestPipeline")

HAS_TESSERACT = shutil.which("tesseract") is not None

# regex to detect figure captions in text, e.g., "Figure 1.2a" or "Fig. 3"
FIGURE_CAPTION_REGEX = re.compile(r"\b(figure|fig\.)\s+\d+(\.\d+)*[a-z]?\b", re.IGNORECASE)


def calculate_sha256(file_path: Path) -> str:
    """Calculates the SHA-256 hash of a file in 64KB chunks."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def get_qdrant_client() -> QdrantClient:
    port = int(os.getenv("QDRANT_PORT", 6333))
    return QdrantClient(host=QDRANT_HOST, port=port)


def init_qdrant_schema(client: QdrantClient):
    """Creates a payload index on file_path for instant filtering."""
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="file_path",
            field_schema=rest.PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass


def get_indexed_files_map(client: QdrantClient) -> Dict[str, str]:
    """Returns a mapping of file paths to their corresponding SHA-256 hashes from the Qdrant collection."""
    files_map = {}
    offset = None
    try:
        while True:
            records, offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,
                offset=offset,
                with_payload=["file_path", "file_hash"],
                with_vectors=False,
            )
            for point in records:
                path = point.payload.get("file_path")
                f_hash = point.payload.get("file_hash")
                if path and f_hash:
                    files_map[path] = f_hash
            if offset is None:
                break
    except Exception as e:
        logger.debug(f"Collection scroll error: {e}")
    return files_map


def delete_file_points(client: QdrantClient, file_path_str: str):
    """Deletes all points in the Qdrant collection associated with a specific file path."""
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=rest.FilterSelector(
                filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="file_path",
                            match=rest.MatchValue(value=file_path_str),
                        )
                    ]
                )
            ),
        )
        logger.info(f"Delete all old vectors : {Path(file_path_str).name}")
    except Exception as e:
        logger.warning(f"Error delete for {file_path_str} : {e}")


def is_actual_diagram_page(page, page_text: str) -> bool:
    """Detects if a PDF page contains a technical diagram or visual based on image size, drawing count, and specific keywords."""
    page_rect = page.rect
    
    # 1. Verification of raster images (photos, 3D renderings) outside headers/footers
    for img_info in page.get_images(full=True):
        try:
            xref = img_info[0]
            base_img = page.parent.extract_image(xref)
            # Exclude small images that are likely logos or decorative elements
            if base_img["width"] >= 400 and base_img["height"] >= 300:
                return True
        except Exception:
            continue

    # 2. Verification of vector drawings (CAD plans, fairing dimensions)
    drawings = page.get_drawings()
    if drawings:
        # Filter drawings to exclude those in the header/footer areas (e.g., title blocks, page numbers)
        body_drawings = [
            d for d in drawings 
            if d["rect"].y0 > 80 and d["rect"].y1 < (page_rect.height - 80)
        ]
        # If there are a significant number of drawings, it's likely a technical diagram page
        if len(body_drawings) >= 30:
            return True
        
        # If the page contains a figure caption and at least 15 drawings, consider it a diagram page
        if len(body_drawings) >= 15 and FIGURE_CAPTION_REGEX.search(page_text):
            return True

    return False


def worker_extract_pdf(task: Tuple[Path, str, str]) -> List[Document]:
    """Extracts text and generates full-page captures for each diagram."""
    pdf_path, pdf_str, file_hash = task
    extracted_docs = []
    pdf_slug = pdf_path.stem.replace(" ", "_")

    try:
        doc = pymupdf.open(str(pdf_path))
        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1

            # Extract text from the page
            page_text = page.get_text("text")

            # Fallback to OCR if the page is mostly empty and Tesseract is available
            if len(page_text.strip()) < 20 and HAS_TESSERACT:
                try:
                    tp = page.get_textpage_ocr(language="eng+fra", dpi=150)
                    page_text = tp.extractText()
                except Exception:
                    pass

            if len(page_text.strip()) < 10:
                continue

            has_schema = is_actual_diagram_page(page, page_text)
            img_rel_path = ""

            # If the page contains a diagram, capture the page at 200 DPI (optimal clarity for dimensions)
            if has_schema:
                try:
                    pix = page.get_pixmap(dpi=200)
                    img_name = f"{pdf_slug}_p{page_num}.png"
                    img_path = IMG_PATH / img_name
                    pix.save(str(img_path))
                    img_rel_path = f"/static/pdf_images/{img_name}"
                except Exception as img_err:
                    logger.warning(f"Échec rendu image p.{page_num} : {img_err}")

            extracted_docs.append(
                Document(
                    text=page_text,
                    metadata={
                        "file_path": pdf_str,
                        "file_name": pdf_path.name,
                        "page_label": str(page_num),
                        "file_hash": file_hash,
                        "image_path": img_rel_path,
                        "has_drawings": has_schema,
                    }
                )
            )
        doc.close()
    except Exception as e:
        logger.error(f"Extraction error on {pdf_path.name} :\n{traceback.format_exc()}")

    return extracted_docs


def run_ingestion_pipeline(max_workers: int = 4, batch_size: int = 5):
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

        if stored_hash != current_hash:
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
    run_ingestion_pipeline()
