import base64
from typing import Optional, List
import pymupdf as fitz
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from src.config import vision_llm, ROOT_DIR, DEFAULT_PDF_PATH, IMG_PATH
from src.services.pdf_engine import render_lazy_visual, detect_visual_bounding_boxes
from src.services.prompt_loader import get_skill_prompt


def _has_rendered_visual(pdf_slug: str, page_num: int) -> bool:
    """Checks if a given page has a detected visual diagram in static/pdf_images."""
    expected_img = IMG_PATH / f"{pdf_slug}_p{page_num}.png"
    sub_matches = list(IMG_PATH.glob(f"*/{pdf_slug}_p{page_num}.png"))
    return expected_img.exists() or len(sub_matches) > 0


@tool
def inspect_pdf_page_image(page_number: int, question: str, bbox: Optional[List[float]] = None) -> str:
    """
    Inspect a technical drawing, a launcher cap plan, or a CAD layout on a given page of the manual.
    """
    try:
        requested_page = int(page_number)
        target_page = requested_page
        pdf_slug = DEFAULT_PDF_PATH.stem.replace(" ", "_")

        # If the LLM requests page N (where a figure/picture is announced in the text),
        # but page N is text-only while page N+1 has the actual full-page diagram,
        # automatically shift to page N+1
        redirect_note = ""
        if not _has_rendered_visual(pdf_slug, target_page):
            if _has_rendered_visual(pdf_slug, target_page + 1):
                target_page = target_page + 1
                redirect_note = (
                    f"(Note: Page {requested_page} is text-only; automatically inspecting "
                    f"the adjacent technical diagram on Page {target_page})\n"
                )
            else:
                return (
                    f"[Visual Inspection Page {requested_page}]: Page {requested_page} (and Page {requested_page + 1}) "
                    f"contain only text and NO technical diagram. Do not inspect this page again."
                )

        # 1. Render the confirmed diagram page
        img_path = render_lazy_visual(page_number=target_page, bbox=bbox, dpi=200)
        img_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

        try:
            rel_img_url = "/" + img_path.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
        except ValueError:
            rel_img_url = str(img_path)

        # 2. Delegation of intelligence to the skill
        skill_prompt = get_skill_prompt("cad-inspector")

        prompt = (
            f"{skill_prompt}\n\n"
            f"--- TECHNICAL DRAWING ANALYSIS ---\n"
            f"Page Number: {target_page}\n"
            f"Question : {question}\n"
            f"Transcribe all visible callouts, equipment names, quantities (e.g. x32, 2 doors), "
            f"diameters (Ø in mm), and vertical heights in a clear Markdown list or table."
        )

        # 3. Call the vision LLM on verified diagram pages
        response = vision_llm.invoke([
            HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ])
        ])

        return (
            f"[Visual Inspection Page {target_page}]:\n{redirect_note}{response.content}\n"
            f"[Associated Page Image]: {rel_img_url}"
        )

    except Exception as e:
        return f"Error during visual inspection: {str(e)}"
