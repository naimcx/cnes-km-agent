from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.tools import search_docs, search_web, fetch_wikipedia
from src.state import AgentState

load_dotenv()

TOOLS_MAP = {
    "search_docs": search_docs,
    "search_web": search_web,
    "fetch_wikipedia": fetch_wikipedia
}

tools = list(TOOLS_MAP.values())
llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# Node 1 : Orchestrator node
def orchestrator_node(state: AgentState):
    """Orchestrates the workflow based on the question type."""
    prompt = (
        "You are an orchestrator for a CNES technical assistant."
        f"Question : {state['question']}\n"
        "Define a brief search and analysis plan"
    )
    response = llm.invoke(prompt)
    return {"plan": response.content}


# NODE 2 : Researcher
def researcher_node(state: AgentState):
    """Queries the vector memory with the question."""
    question = state["question"]
    plan = state.get("plan", "")

    prompt = (
        f"You are a researcher agent for CNES."
        f"Question: {question}\n"
        f"Use the best tools provided to answer the question."
    )
    response = llm_with_tools.invoke(prompt)

    context = []
    if response.tool_calls:
        for tool in response.tool_calls:
            tool_name = tool["name"]
            tool_args = tool["args"]

            if tool_name in TOOLS_MAP:
                tool_result = TOOLS_MAP[tool_name].invoke(tool_args)
                context.append(f"[{tool_name}]: {tool_result}")
    else:
        context.append(response.content)

    return {
        "context": context,
        "iteration": state.get("iteration", 0) + 1
    }

# NODE 3 : Analyst
def analyst_node(state: AgentState):
    """Draft a technical answer based STRICTLY on the retrieved context."""
    context_str = "\n".join(state["context"])
    feedback = state.get("critic", "")
    
    prompt = (
        f"You are a technical engineer at CNES Space Transport.\n"
        f"Documentary context verified :\n{context_str}\n\n"
        f"Question : {state['question']}\n"
    )
    if feedback:
        prompt += f"Warning, critic signaled the following issue in the previous round : {feedback}\n"
        
    prompt += "Respond in a factual and technical manner. Do not invent any figures or names."
    
    response = llm.invoke(prompt)
    return {"analysis": response.content}

# NODE 4 : Critic
def critic_node(state: AgentState):
    """Verifies strict compliance with the documents and assigns a score (anti-hallucination)."""
    context_str = "\n".join(state["context"])
    draft = state.get("analysis", "")
    
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
    
    score = 5
    for line in eval_res.split("\n"):
        if "SCORE:" in line:
            try:
                score = int(line.replace("SCORE:", "").strip())
            except ValueError:
                score = 7
                
    return {
        "critic": eval_res,
        "score": score,
    }


# Node 5 : Synthesizer
def synthesizer_node(state: AgentState):
    """Synthesizes the final answer based on the critic's evaluation and the analyst's draft."""
    prompt = (
        f"You are a synthesizer for a multi-agent system."
        f"Synthesise a clear and complete final answer."
        f"Question: {state['question']}\n"
        f"Research: {state['context']}\n"
        f"Analysis: {state['draft']}\n"
        f"Critic: {state['critic_notes']}\n"
        f"Format the answer with clear sections."
    )
    response = llm.invoke(prompt)
    return {"final_answer": response.content}


# Node 6 : SelfReflection node
def self_reflection_node(state: AgentState):
    """Analyze all exchange history to modify the research strategy and improve the next answer."""
