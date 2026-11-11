import pytest
from src.services.prompt_loader import get_node_prompt, get_skill_prompt
from src.services.pdf_engine import render_lazy_visual
from src.tools import (
    calculate_aerospace_math,
    fetch_wikipedia,
    search_web,
)
from src.config import DEFAULT_PDF_PATH


def test_all_node_prompts_load():
    """Verifies that all 6 LangGraph agent prompts exist and are non-empty."""
    nodes = ["orchestrator", "researcher", "analyst", "critic", "self_reflection", "synthesizer"]
    for node_name in nodes:
        content = get_node_prompt(node_name)
        assert len(content.strip()) > 20, f"Prompt for {node_name} is empty or missing."


def test_all_skill_prompts_load():
    """Verifies that all domain skills exist and load properly."""
    skills = ["cad-inspector", "aerospace-calc"]
    for skill_name in skills:
        content = get_skill_prompt(skill_name)
        assert len(content.strip()) > 20, f"Skill {skill_name} is empty or missing."


def test_lazy_visual_rendering():
    """Verifies that PyMuPDF renders a page from the manual into static/pdf_images/."""
    if not DEFAULT_PDF_PATH.exists():
        pytest.skip(f"Sample PDF {DEFAULT_PDF_PATH} not found.")
    img_path = render_lazy_visual(page_number=1, pdf_path=DEFAULT_PDF_PATH, dpi=150)
    assert img_path.exists()
    assert img_path.suffix == ".png"
    assert img_path.stat().st_size > 1000


def test_aerospace_calculator_docker():
    """Verifies that the Dockerized aerospace calculator executes math and constants."""
    code = "r = 2.7\nresult = round(math.pi * (r ** 2) * constants.g, 2)"
    output = calculate_aerospace_math.invoke({"code": code})
    assert "Error" not in output, f"Calculator failed with: {output}"
    assert float(output) == pytest.approx(224.6, rel=1e-2)


def test_wikipedia_tool():
    """Verifies that the Wikipedia tool fetches article summaries."""
    res = fetch_wikipedia.invoke({"title": "Ariane 6"})
    assert "Ariane 6" in res or "launch system" in res


def test_searxng_web_search():
    """Verifies that the local SearXNG container responds to queries."""
    res = search_web.invoke({"query": "Ariane 6 launcher CNES"})
    assert "Error" not in res, f"SearXNG container might not be running: {res}"
