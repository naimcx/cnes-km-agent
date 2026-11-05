from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tools import search_docs
from state import AgentState

load_dotenv()
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

# NODE 1 : Researcher
def researcher_node(state: AgentState):
    """Queries the vector memory with the question."""
    retrieved = search_docs.invoke(state["question"], k=2)
    return {
        "context": retrieved,
        "iteration": state.get("iteration", 0) + 1
    }

# NODE 2 : Analyst
def analyst_node(state: AgentState):
    """Draft a technical answer based STRICTLY on the retrieved context."""
    context_str = "\n".join(state["context"])
    feedback = state.get("critic_notes", "")
    
    prompt = (
        f"You are a technical engineer at CNES Space Transport.\n"
        f"Documentary context verified :\n{context_str}\n\n"
        f"Question : {state['question']}\n"
    )
    if feedback:
        prompt += f"Warning, critic signaled the following issue in the previous round : {feedback}\n"
        
    prompt += "Respond in a factual and technical manner. Do not invent any figures or names."
    
    response = llm.invoke(prompt)
    return {"draft": response.content}

# NODE 3 : Critic / Fact-Checker
def critic_node(state: AgentState):
    """Verifies strict compliance with the documents and assigns a score (anti-hallucination)."""
    context_str = "\n".join(state["context"])
    draft = state["draft"]
    
    prompt = (
        f"You are the quality control officer for space data.\n"
        f"Source documents :\n{context_str}\n\n"
        f"Written response :\n{draft}\n\n"
        f"Instructions :\n"
        f"1. Give a score from 0 to 10 for the strict fidelity to the documents.\n"
        f"2. Write 'VALID' if the score is >= 7, otherwise explain the error.\n"
        f"3. Follow the imposed format.\n"
        f"SCORE: <note>\n"
        f"NOTICE: <VALID or explanation of the error>\n"
    )
    
    eval_res = llm.invoke(prompt).content
    
    # Extract score from the evaluation response
    score = 5
    for line in eval_res.split("\n"):
        if "SCORE:" in line:
            try:
                score = int(line.replace("SCORE:", "").strip())
            except ValueError:
                score = 7
                
    return {
        "critic_notes": eval_res,
        "score": score,
        "final_answer": draft
    }