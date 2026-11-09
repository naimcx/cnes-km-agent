import argparse
import os
from src.graph import app
from src.config import (
    TEXT_PROVIDER,
    GROQ_TEXT_MODEL,
    LOCAL_TEXT_MODEL,
    EMBEDDING_MODEL,
)


# ===========================================
# CLI EXECUTION
# ===========================================

def main():
    """
    Command-line interface for the CNES Knowledge Management Multi-Agent System.
    """
    parser = argparse.ArgumentParser(description="CNES Knowledge Management MAS CLI")
    parser.add_argument("--db", choices=["qdrant", "chroma", "faiss"], default="qdrant", help="Vector DB")
    parser.add_argument("--mode", choices=["hybrid", "vector", "keyword"], default="hybrid", help="Search mode")
    parser.add_argument("--question", type=str, default="", help="Question to ask")
    parser.add_argument("--model", type=str, default=LOCAL_TEXT_MODEL, help="Text model to use")
    parser.add_argument("--embedding", type=str, default=EMBEDDING_MODEL, help="Embedding model to use")
    args = parser.parse_args()

    # Set environment variables based on CLI arguments
    os.environ["RAG_DB"] = args.db
    os.environ["RAG_MODE"] = args.mode

    active_text_model = GROQ_TEXT_MODEL if TEXT_PROVIDER == "groq" else LOCAL_TEXT_MODEL

    print("============ CNES Knowledge Management Test - Multi-Agent Demo ============")
    print(
        f"Active Configuration: DB={args.db}, Mode={args.mode}, " 
        f"Provider={TEXT_PROVIDER}, Model={active_text_model}, Embedding Model={args.embedding}"
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
    res = app.invoke({"question": question, "iteration": 0})

    # Display cumulative context retrievec during tool interactions
    print("\n============ SOURCE AND CONTEXT RETRIEVED ============")
    for i, ctx in enumerate(res.get("context", []), start=1):
        print(f"\n--- Chunk {i} ---\n{ctx}")

    # Display final results and any critic feedback
    print("\n============ VALIDATED RESULT ============")
    print(f"Iterations : {res.get('iteration', 0)}")
    print(f"Control report :\n{res.get('critic', 'N/A')}")
    print(f"\nFinal answer :\n{res.get('final_answer', 'NO ANSWER')}")


if __name__ == "__main__":
    main()
