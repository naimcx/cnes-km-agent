import abc
import re
import logging
from pathlib import Path
from typing import List, Optional

import pymupdf as fitz
from llama_index.core import Document

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling.datamodel.base_models import InputFormat

from unstructured.partition.pdf import partition_pdf

from src.config import (
    ROOT_DIR,
    DATA_PATH,
    IMG_PATH,
    PDF_PARSER_TYPE,
    UNSTRUCTURED_STRATEGY,
    PRE_RENDER_IMAGES
)

logger = logging.getLogger(__name__)

# Regular expression to detect figure/picture/table captions in text, e.g., "Figure 5.3.1a" or "Picture 5.3.4.a"
FIGURE_CAPTION_REGEX = re.compile(
    r"\b(figure|fig\.|picture|pic\.|diagram)\s+\d+(\.\d+)*[a-z]?\b",
    re.IGNORECASE
)


# =====================================================================
# 1. DETECTION OF TECHNICAL FIGURES (VECTOR DRAWINGS OR BITMAP IMAGES)
# =====================================================================

def detect_visual_bounding_boxes(page, page_text: str = "") -> List[List[float]]:
    """
    Detects technical figures (dense vector drawings or bitmap images)
    while excluding header (Arianespace/CNES logos) and footer areas.
    """
    page_rect = page.rect
    header_margin = 75.0
    footer_margin = page_rect.height - 75.0
    side_margin = 30.0

    boxes: List[fitz.Rect] = []

    # 1. Bitmap images (photos, 3D renderings) outside the margins
    for img_info in page.get_images(full=True):
        try:
            xref = img_info[0]
            base_img = page.parent.extract_image(xref)
            # Exclude low-resolution bullets and logos
            if base_img["width"] >= 250 and base_img["height"] >= 200:
                for r in page.get_image_rects(xref):
                    if r.y0 > header_margin and r.y1 < footer_margin:
                        boxes.append(r)
        except Exception:
            continue

    # 2. Vector drawings (CAD plans, technical diagrams) outside the margins
    drawings = page.get_drawings()
    if drawings:
        body_drawings = [
            d["rect"] for d in drawings
            if d["rect"].y0 > header_margin
            and d["rect"].y1 < footer_margin
            and d["rect"].x0 > side_margin
            and d["rect"].x1 < (page_rect.width - side_margin)
            and d["rect"].width > 30
            and d["rect"].height > 30
        ]

        has_caption = FIGURE_CAPTION_REGEX.search(page_text) is not None
        if len(body_drawings) >= 20 or (len(body_drawings) >= 8 and has_caption):
            cluster_rect = fitz.Rect()
            for r in body_drawings:
                cluster_rect.include_rect(r)

            if cluster_rect.width > 120 and cluster_rect.height > 120:
                boxes.append(cluster_rect)

    # 3. Formatting in coordinates [x0, y0, x1, y1] with a 10-pixel margin
    serialized = []
    for b in boxes:
        expanded = fitz.Rect(
            max(0, b.x0 - 10),
            max(0, b.y0 - 10),
            min(page_rect.width, b.x1 + 10),
            min(page_rect.height, b.y1 + 10),
        )
        serialized.append([round(v, 2) for v in [expanded.x0, expanded.y0, expanded.x1, expanded.y1]])

    return serialized


# =====================================================================
# 2. RENDERING OF FIGURES (FULL PAGE OR CROPPED)
# ====================================================================

def render_lazy_visual(
    page_number: int,
    pdf_path: Optional[Path] = None,
    bbox: Optional[List[float]] = None,
    dpi: int = 200,
) -> Path:
    """
    Renders and saves a PNG snapshot of a figure (full page or cropped bounding box).
    """
    target_pdf = pdf_path or next(DATA_PATH.glob("*.pdf"), None)
    if not target_pdf or not target_pdf.exists():
        raise FileNotFoundError(f"No PDF file found in {DATA_PATH}.")

    IMG_PATH.mkdir(parents=True, exist_ok=True)
    pdf_slug = target_pdf.stem.replace(" ", "_")
    suffix = f"_crop_{int(bbox[0])}_{int(bbox[1])}" if bbox else ""
    img_filename = f"{pdf_slug}_p{page_number}{suffix}.png"

    # 1. If parser subfolders exist from testing/benchmarking, reuse the image from there!
    if not bbox:
        parser_subfolder_img = IMG_PATH / PDF_PARSER_TYPE / img_filename
        if parser_subfolder_img.exists():
            return parser_subfolder_img
        # Also check any other subfolder in IMG_PATH
        existing_matches = list(IMG_PATH.glob(f"*/{img_filename}"))
        if existing_matches:
            return existing_matches[0]

    # 2. Standard production path in static/pdf_images/
    output_path = IMG_PATH / img_filename
    if output_path.exists():
        return output_path

    # 3. Lazy render only if the image does not exist anywhere yet
    doc = fitz.open(str(target_pdf))
    page_idx = max(0, min(page_number - 1, len(doc) - 1))
    page = doc.load_page(page_idx)

    if bbox:
        crop_rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
        pix = page.get_pixmap(dpi=dpi, clip=crop_rect)
    else:
        pix = page.get_pixmap(dpi=dpi)

    pix.save(str(output_path))
    doc.close()
    return output_path


# =====================================================================
# 3. PROCESSING DOCUMENTS WITH VISUALS (Bounding Boxes + Optional Pre-rendering)
# ====================================================================

def process_document_visuals(
    docs: List[Document],
    pdf_path: Path,
    pre_render: bool = PRE_RENDER_IMAGES,
    output_dir: Optional[Path] = None
) -> List[Document]:
    """
    Analyzes the pages of any extractor, computes bounding boxes,
    and immediately generates the PNG file in static/ if pre_render=True.
    """
    target_img_dir = output_dir if output_dir is not None else IMG_PATH
    target_img_dir.mkdir(parents=True, exist_ok=True)

    pdf_slug = pdf_path.stem.replace(" ", "_")
    doc_fitz = None

    try:
        doc_fitz = fitz.open(str(pdf_path))
        for d in docs:
            page_label = d.metadata.get("page_label", "1")
            try:
                page_num = int(page_label)
            except ValueError:
                page_num = 1

            page_idx = max(0, min(page_num - 1, len(doc_fitz) - 1))
            fitz_page = doc_fitz.load_page(page_idx)

            boxes = detect_visual_bounding_boxes(fitz_page, d.text)
            has_visual = len(boxes) > 0

            img_rel_path = ""
            if has_visual and pre_render:
                img_name = f"{pdf_slug}_p{page_num}.png"
                img_dest = target_img_dir / img_name

                if pre_render and not img_dest.exists():
                    pix = fitz_page.get_pixmap(dpi=200)
                    pix.save(str(img_dest))

                try:
                    rel_to_root = img_dest.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
                    img_rel_path = f"/{rel_to_root}"
                except ValueError:
                    img_rel_path = f"/static/pdf_images/{img_name}"

            d.metadata["has_drawings"] = has_visual
            d.metadata["has_visual"] = has_visual
            d.metadata["visual_bboxes"] = boxes
            d.metadata["image_path"] = img_rel_path
            d.metadata["pdf_file_name"] = pdf_path.name

    finally:
        if doc_fitz:
            doc_fitz.close()

    return docs


# =====================================================================
# 4. ABSTRACT BASE CLASS FOR PDF PARSERS
# =====================================================================

class BasePDFParser(abc.ABC):
    """Abstract base class for PDF parsers."""

    @abc.abstractmethod
    def extract(self, pdf_path: Path, file_hash: str) -> List[Document]:
        """Extracts the document into a list of LlamaIndex Document objects with metadata."""
        pass


# =====================================================================
# 5. PYMUPDF (FITZ) EXTRACTION: Markdown or Plain Text with Figure Detection
# =====================================================================

class PyMuPDFParser(BasePDFParser):
    def __init__(self, mode: str = "markdown"):
        self.mode = mode

    def extract(self, pdf_path: Path, file_hash: str) -> List[Document]:
        doc = fitz.open(str(pdf_path))
        extracted_docs = []

        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1

            if self.mode == "markdown" and hasattr(page, "to_markdown"):
                page_content = page.to_markdown()
            else:
                page_content = page.get_text("text")

            if len(page_content.strip()) < 10:
                continue

            enriched_text = f"[Document Page {page_num}]\n\n{page_content}"

            extracted_docs.append(
                Document(
                    text=enriched_text,
                    metadata={
                        "file_path": str(pdf_path.resolve()),
                        "file_name": pdf_path.name,
                        "page_label": str(page_num),
                        "file_hash": file_hash,
                        "parser": f"pymupdf_{self.mode}",
                    }
                )
            )
        doc.close()
        return process_document_visuals(extracted_docs, pdf_path)


# =====================================================================
# 6. DOCLING (IBM) EXTRACTION: Nested tables, LaTeX and Markdown formulas.
# =====================================================================

class DoclingParser(BasePDFParser):
    def __init__(self):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        pipeline_options.do_ocr = False
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=AcceleratorDevice.AUTO
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def extract(self, pdf_path: Path, file_hash: str) -> List[Document]:
        result = self.converter.convert(pdf_path)
        extracted_docs = []
        fitz_doc = fitz.open(str(pdf_path))

        try:
            if hasattr(result.document, "pages") and result.document.pages:
                for page_no, page in result.document.pages.items():
                    page_md = result.document.export_to_markdown(page_no=page_no)

                    # Only on full-page diagrams where Docling outputs almost no text (< 250 chars, e.g. Page 84),
                    # supplement with PyMuPDF text so vector dimensions (Ø 575, Ø 4600, Ø 5400) are indexed in Qdrant
                    clean_md = page_md.replace("<!-- image -->", "").strip()
                    if len(clean_md) < 250 and 1 <= page_no <= len(fitz_doc):
                        raw_text = fitz_doc[page_no - 1].get_text("text").strip()
                        if len(raw_text) > len(clean_md):
                            page_md = f"{page_md}\n\n{raw_text}"

                    if len(page_md.strip()) < 10:
                        continue

                    enriched_text = f"[Document Page: {page_no}]\n{page_md}"

                    extracted_docs.append(
                        Document(
                            text=enriched_text,
                            metadata={
                                "file_path": str(pdf_path.resolve()),
                                "file_name": pdf_path.name,
                                "page_label": str(page_no),
                                "file_hash": file_hash,
                                "parser": "docling",
                            }
                        )
                    )
        finally:
            fitz_doc.close()

        return process_document_visuals(extracted_docs, pdf_path)


# =====================================================================
# 7. UNSTRUCTURED (Partitioning by typed elements)
# =====================================================================

class UnstructuredParser(BasePDFParser):
    def __init__(self, strategy: str = "fast"):
        self.strategy = strategy

    def extract(self, pdf_path: Path, file_hash: str) -> List[Document]:

        elements = partition_pdf(
            filename=str(pdf_path),
            strategy=self.strategy,
            infer_table_structure=(self.strategy == "hi_res"),
        )

        pages_content = {}
        for el in elements:
            p_num = getattr(el.metadata, "page_number", 1)
            pages_content.setdefault(p_num, []).append(str(el))

        extracted_docs = []
        for p_num, chunks in pages_content.items():
            text = "\n\n".join(chunks)
            if len(text.strip()) < 10:
                continue

            extracted_docs.append(
                Document(
                    text=text,
                    metadata={
                        "file_path": str(pdf_path.resolve()),
                        "file_name": pdf_path.name,
                        "page_label": str(p_num),
                        "file_hash": file_hash,
                        "parser": f"unstructured_{self.strategy}",
                    }
                )
            )
        return process_document_visuals(extracted_docs, pdf_path)


# =====================================================================
# 8. FACTORY : EXTRACTION ENGINE SELECTION
# =====================================================================

def get_pdf_parser(parser_type: Optional[str] = None) -> BasePDFParser:
    """Instantiates the requested engine for benchmarking or production."""
    selected_parser = (parser_type or PDF_PARSER_TYPE).strip().lower()

    if selected_parser == "pymupdf":
        return PyMuPDFParser(mode="markdown")
    elif selected_parser == "pymupdf_text":
        return PyMuPDFParser(mode="text")
    elif selected_parser == "docling":
        return DoclingParser()
    elif selected_parser == "unstructured":
        return UnstructuredParser(strategy=UNSTRUCTURED_STRATEGY)
    else:
        raise ValueError(
            f"Parser {parser_type} not recognized. Valid options: ['docling', 'pymupdf', 'pymupdf_text', 'unstructured']"
        )
