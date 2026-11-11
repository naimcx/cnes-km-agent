import argparse
import os
from dotenv import load_dotenv

load_dotenv()


def parse_cli_args():
    """
    Parse command-line arguments for the CNES Knowledge Management Multi-Agent System.
    """
    parser = argparse.ArgumentParser(description="CNES Knowledge Management MAS CLI")
    parser.add_argument("--db", choices=["qdrant", "chroma", "faiss"], default=None, help="Vector DB")
    parser.add_argument("--mode", choices=["hybrid", "vector", "keyword"], default=None, help="Search mode")
    parser.add_argument("--provider", choices=["groq", "ollama"], default=None, help="LLM Provider")
    parser.add_argument("--text-model", type=str, default=None, help="Text model to use")
    parser.add_argument("--vision-model", type=str, default=None, help="Vision model to use")
    parser.add_argument("--embedding", type=str, default=None, help="Embedding model to use")
    parser.add_argument("--question", type=str, default="", help="Question to ask")
    return parser.parse_args()

args = parse_cli_args()

if args.db:
    os.environ["RAG_DB"] = args.db
if args.mode:
    os.environ["RAG_MODE"] = args.mode
if args.provider:
    os.environ["TEXT_PROVIDER"] = args.provider
    os.environ["VISION_PROVIDER"] = args.provider
if args.text_model:
    if os.getenv("TEXT_PROVIDER", "ollama").lower() == "groq":
        os.environ["GROQ_TEXT_MODEL"] = args.text_model
    else:
        os.environ["LOCAL_TEXT_MODEL"] = args.text_model
if args.vision_model:
    if os.getenv("VISION_PROVIDER", "ollama").lower() == "groq":
        os.environ["GROQ_VISION_MODEL"] = args.vision_model
    else:
        os.environ["LOCAL_VISION_MODEL"] = args.vision_model
if args.embedding:
    os.environ["EMBEDDING_MODEL"] = args.embedding

from src.config import (
    TEXT_PROVIDER,
    GROQ_TEXT_MODEL,
    GROQ_VISION_MODEL,
    LOCAL_TEXT_MODEL,
    LOCAL_VISION_MODEL,
    EMBEDDING_MODEL,
    RAG_DB,
    RAG_MODE,
)
from src.graph import app

# ===========================================
# CLI EXECUTION
# ===========================================

def main():
    """
    Command-line interface for the CNES Knowledge Management Multi-Agent System.
    """
    active_text_model = GROQ_TEXT_MODEL if TEXT_PROVIDER == "groq" else LOCAL_TEXT_MODEL
    active_vision_model = GROQ_VISION_MODEL if TEXT_PROVIDER == "groq" else LOCAL_VISION_MODEL

    print("============ CNES Knowledge Management Test - Multi-Agent Demo ============")
    print(
        f"Active Configuration: \n"
        f"DB = {RAG_DB}, Mode = {RAG_MODE}, " 
        f"Provider = {TEXT_PROVIDER}, "
        f"Text LLM = [{active_text_model}], "
        f"Vision Model = [{active_vision_model}], "
        f"Embedding Model = [{EMBEDDING_MODEL}]"

    )

    # Prompt for a question if not provided via CLI
    question = args.question.strip()
    if not question:
        question = input("Enter your question about Ariane 6: ")
        if not question.strip():
            question = (
                "What is the usable volume within the A62 for a fairing in a single launch? I would like the diameter at the very top and at the base."
            )
            # What equipment, access doors, and separation systems are present on the Ariane 6 fairing? I need the dimensions of the doors.

    print(f"\nExecuting graph for: {question}...")

    # Execute workflow until graph reaches END
    res = app.invoke({"question": question, "reflection_cycle": 0})

    # Display cumulative context retrievec during tool interactions
    print("\n============ SOURCE AND CONTEXT RETRIEVED ============")
    for i, ctx in enumerate(res.get("context", []), start=1):
        print(f"\n--- Chunk {i} ---\n{ctx}")

    # Display final results and any critic feedback
    print("\n============ VALIDATED RESULT ============")
    print(f"Reflection-cycle : {res.get('reflection_cycle', 0)}")
    print(f"Is the answer valid? : {res.get('is_valid', False)}")
    print(f"Control report :\n{res.get('critic', 'N/A')}")
    print(f"\nFinal answer :\n{res.get('final_answer', 'NO ANSWER')}")


if __name__ == "__main__":
    main()
