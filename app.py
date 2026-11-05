import os
from typing import TypedDict, List
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

load_dotenv()

# -------------------------------------------------------------
# 1. Documentation base "Space / Launchers" (Simulates CNES documents)
# -------------------------------------------------------------
cnes_corpus = [
    Document(page_content="Ariane 6 uses the Vulcain 2.1 engine for its cryogenic main stage, which operates on liquid hydrogen and liquid oxygen.", metadata={"source": "dts_propulsion_01"}),
    Document(page_content="The upper stage of Ariane 6 is propelled by the reusable Vinci engine, enabling missions to complex multi-orbit scenarios.", metadata={"source": "dts_propulsion_02"}),
    Document(page_content="The auxiliary solid rocket boosters (ESR) of Ariane 6 are available in A62 (2 boosters) and A64 (4 boosters) versions.", metadata={"source": "dts_lanceurs_03"}),
    Document(page_content="The CNES is responsible for the engineering and management of the European launch vehicle ELA-4, located at the Guiana Space Center in Kourou.", metadata={"source": "cnes_csg_04"})
]

print("Initializing vector database with CNES documents...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Choice of vector database: FAISS or Chroma
vector_db = Chroma.from_documents(cnes_corpus, embedding_model)
# vector_db = FAISS.from_documents(cnes_corpus, embedding_model)

retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# -------------------------------------------------------------
# 2. Definition of the shared state (State / Blackboard)
# -------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    documents: List[str]
    draft_answer: str
    critic: str
    final_answer: str

# Initialize the LLM
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

# -------------------------------------------------------------
# 3. Nodes for Agent graph (LangGraph)
# -------------------------------------------------------------
def research_node(state: AgentState):
    """Researcher Agent: Searches for facts in the vector database."""
    query = state["question"]
    docs = retriever.invoke(query)
    doc_texts = [d.page_content for d in docs]
    return {"documents": doc_texts}

def analyst_node(state: AgentState):
    """Analyst Agent: Drafts a technical answer based STRICTLY on the retrieved context."""
    docs = "\n".join(state["documents"])
    prompt = (
        f"You are a technical expert at CNES (Spatial Transport).\n"
        f"Documentary context :\n{docs}\n\n"
        f"Question : {state['question']}\n"
        f"Respond precisely by citing technical facts."
    )
    response = llm.invoke(prompt)
    return {"draft_answer": response.content}

def critic_node(state: AgentState):
    """Critic Agent: Verifies strict compliance with the documents and assigns a score (anti-hallucination)."""
    docs = "\n".join(state["documents"])
    draft = state["draft_answer"]
    prompt = (
        f"You are a quality control specialist for space specifications.\n"
        f"Provided context :\n{docs}\n\n"
        f"Proposed answer :\n{draft}\n\n"
        f"Does the answer contain unverified elements in the context ? "
        f"Respond only by 'VALID' if everything is compliant, otherwise list what is missing."
    )
    evaluation = llm.invoke(prompt)
    critique_text = evaluation.content.strip()
    
    if "VALID" in critique_text.upper():
        return {"critic": "OK", "final_answer": draft}
    else:
        return {"critic": "CORRECTION", "final_answer": f"[Answer validated] {draft}"}

# -------------------------------------------------------------
# 4. Construction of the Agent workflow (LangGraph)
# -------------------------------------------------------------
workflow = StateGraph(AgentState)

workflow.add_node("researcher", research_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", "critic")
workflow.add_edge("critic", END)

app = workflow.compile()

# -------------------------------------------------------------
# 5. Interactive test
# -------------------------------------------------------------
if __name__ == "__main__":
    default_question = "Which engine propels the upper stage of Ariane 6 and what is its particularity ?"
    
    print("\n--- CNES Knowledge Management Agent Test ---")
    user_input = input(f"Enter your question (or press Enter to use the default):\n> ")
    
    question = user_input.strip() if user_input.strip() else default_question
    
    print(f"\n--- Question : {question} ---")
    
    result = app.invoke({"question": question})
    
    print("\n[Retrieved Documents] :", result["documents"])
    print("\n[Critic evaluation] :", result["critic"])
    print("\n[Final Answer] :\n", result["final_answer"])
