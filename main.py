from src.graph import app

def main():
    print("============ CNES Knowledge Management Test - Multi-Agent Demo ============")
    question = input("Enter your question about Ariane 6: ")
    
    if not question.strip():
        question = "What are the engines of the main stage and upper stage of Ariane 6?"

    print(f"\nExecuting graph for: {question}\n...")
    res = app.invoke({"question": question, "iteration": 0})

    print("\n============ VALIDATED RESULT ============")
    print(f"Iterations : {res['iteration']}")
    print(f"Quality score : {res['score']}/10")
    print(f"Control report :\n{res['critic_notes']}")
    print(f"\nFinal answer :\n{res['final_answer']}")

if __name__ == "__main__":
    main()