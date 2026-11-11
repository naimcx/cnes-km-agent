import shutil
import time
from pathlib import Path
import src.services.pdf_engine as pdf_engine
from src.services.pdf_engine import get_pdf_parser
from src.config import IMG_PATH, DEFAULT_PDF_PATH, ROOT_DIR

ENGINES = ["pymupdf", "pymupdf_text", "docling", "unstructured"]

# True : Creates static/pdf_images/<engine_name>/ for each parser to compare them
# False : Cleans static/pdf_images/ before each parser run
SEPARATE_FOLDERS = True

def clear_png_files(directory: Path):
    """Deletes all .png files in the specified directory."""
    if directory.exists():
        for png_file in directory.glob("*.png"):
            png_file.unlink()

def test_pdf_parser_factory():
    for engine_name in ENGINES:
        parser = get_pdf_parser(engine_name)
        assert parser is not None

def run_benchmark():
    if not DEFAULT_PDF_PATH.exists():
        print(f"Target file not found : {DEFAULT_PDF_PATH}")
        return

    print(f"=== COMPARATIVE BENCHMARK OF PARSERS ({DEFAULT_PDF_PATH.name}) ===\n")
    print(f"Mode: {'Separate subfolder per parser' if SEPARATE_FOLDERS else 'Single folder cleaned between runs'}\n")

    clear_png_files(IMG_PATH)
    original_img_path = pdf_engine.IMG_PATH

    try:
        for engine_name in ENGINES:
            try:
                if SEPARATE_FOLDERS:
                    engine_img_dir = original_img_path / engine_name
                    if engine_img_dir.exists():
                        shutil.rmtree(engine_img_dir)
                    engine_img_dir.mkdir(parents=True, exist_ok=True)
                    pdf_engine.IMG_PATH = engine_img_dir
                else:
                    engine_img_dir = original_img_path
                    clear_png_files(engine_img_dir)

                parser = get_pdf_parser(engine_name)

                t0 = time.time()
                docs = parser.extract(DEFAULT_PDF_PATH, file_hash="test_benchmark_hash")
                duration = time.time() - t0

                total_chars = sum(len(d.text) for d in docs)
                nb_tables = sum(1 for d in docs if "|" in d.text)
                nb_visuals = sum(1 for d in docs if d.metadata.get("has_visual", False))
                nb_images_created = len(list(engine_img_dir.glob("*.png")))
                rel_folder = engine_img_dir.relative_to(ROOT_DIR).as_posix()

                print(f"[{engine_name.upper()}]")
                print(f"• Total time            : {duration:.2f} s ({len(docs)} extracted pages)")
                print(f"• Extract volume        : {total_chars:,} chars")
                print(f"• Tabular pages         : {nb_tables} detected")
                print(f"• Visual diagrams       : {nb_visuals} pages with visuals")
                print(f"• PNG generated on disk : {nb_images_created} images in {rel_folder}/\n")

            except Exception as err:
                print(f"[{engine_name.upper()}] Test Failed: {err}\n")
    finally:
        pdf_engine.IMG_PATH = original_img_path

if __name__ == "__main__":
    run_benchmark()
