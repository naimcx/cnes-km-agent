import re
import json
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from src.tools import ALL_TOOLS
from src.config import llm
from src.state import AgentState
from schema.pydantic_schema import CriticOutput, ReflectionOutput, OrchestratorPlan
from src.services.prompt_loader import get_node_prompt, get_skill_prompt


# =============================================
# LOAD ENVIRONMENT VARIABLES AND INITIALIZE LLM
# =============================================

TOOLS_REGISTRY = {tool.name: tool for tool in ALL_TOOLS}

# Bind the LLM with all tools
llm_with_tools = llm.bind_tools(ALL_TOOLS)

# Prepare structured output evaluators for Critic, Self-Reflection, and Orchestrator nodes
critic_evaluator = llm.with_structured_output(CriticOutput)
reflection_evaluator = llm.with_structured_output(ReflectionOutput)
orchestrator_evaluator = llm.with_structured_output(OrchestratorPlan)


# ===========================================
# TOOL CALL PARSING HELPERS
# ===========================================

def extract_fallback_tool_calls(text: str) -> list[dict]:
    """
    Parses raw textual JSON tool calls from the LLM output when a list of tool calls is not returned.
    Extracts tool name and arguments dictionary from code blocks or inline JSON structures.
    Returns a list of dictionaries with keys: "name", "args", and "id".
    """
    calls = []
    # Match patterns {"name": "...", "args": {...}} or {"name": "...", "parameters": {...}}
    patterns = re.findall(
        r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"(?:parameters|args)"\s*:\s*(\{.*?\})\s*\}',
        text,
        re.DOTALL
    )
    for name, args_str in patterns:
        try:
            args = json.loads(args_str)
            calls.append({"name": name, "args": args, "id": f"call_{name}"})
        except Exception:
            continue
    return calls


# ===========================================
# NODE 1 : ORCHESTRATOR
# ===========================================

def orchestrator_node(state: AgentState):
    """Orchestrates the workflow based on the question type."""
    system_prompt = get_node_prompt("orchestrator")
    user_prompt = (
        f"Question : {state['question']}\n"
    )

    try:
        plan_output: OrchestratorPlan = orchestrator_evaluator.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        return {
            "plan": plan_output.search_plan,
            "routing": plan_output.routing
        }
    except Exception:
        return {
            "plan": "Search internal documentation first, fallback to web if missing.",
            "routing": "internal"
        }


# ===========================================
# NODE 2 : RESEARCHER
# ===========================================

def researcher_node(state: AgentState):
    """
    Executes the search plan and retrieves relevant context from internal and external sources.
    The researcher can call multiple tools in sequence
    """
    question = state["question"]
    plan = state.get("plan", "")
    critic_feedback = state.get("critic", "")

    system_prompt = get_node_prompt("researcher")
    calc_skill = get_skill_prompt("aerospace-calc")
    system_prompt += f"\n\n{calc_skill}\n"
    user_prompt = f"Target Question: {question}\nPlan: {plan}\n"

    if critic_feedback:
        user_prompt += f"\nPrevious critic Alert: {critic_feedback}\n"

    history = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    context = list(state.get("context", []))

    # Multi-step iterative research loop: the researcher can call multiple tools in sequence, up to 3 iterations
    for step in range(3):
        response = llm_with_tools.invoke(history)
        history.append(response)

        # 1. Retrieve tool calls from the LLM response, including fallback parsing if the structured output is missing
        tool_calls = list(response.tool_calls or [])
        if not tool_calls and response.content:
            fallback = extract_fallback_tool_calls(response.content)
            if fallback:
                tool_calls.extend(fallback)

        # If no tool calls are returned, assume the LLM has completed its research and append the final content to context
        if not tool_calls:
            if not context and response.content:
                context.append(response.content)
            break

        # 2. Iterate through each tool call, invoke the corresponding tool, and append the results to context and history
        for tool_call in tool_calls:
            t_name = tool_call["name"]
            t_args = tool_call.get("args", {}) or tool_call.get("parameters", {})
            t_id = tool_call.get("id", f"call_{t_name}_{step}")

            if t_name == "inspect_pdf_page_image" and "question" not in t_args:
                t_args["question"] = question

            print(f"\n[Tool Call] {t_name}({t_args})")
            if t_name in TOOLS_REGISTRY:
                t_res = TOOLS_REGISTRY[t_name].invoke(t_args)
            else:
                t_res = f"Error: Tool {t_name} not found."

            context.append(f"[{t_name}]: {t_res}")
            history.append(ToolMessage(content=str(t_res), tool_call_id=t_id))

    return {
        "context": context,
    }


# ===========================================
# NODE 3 : ANALYST
# ===========================================

def analyst_node(state: AgentState):
    """
    Draft a technical answer based STRICTLY on the retrieved context.
    The analyst must not hallucinate or invent any data, and must not extrapolate or calculate values with invented parameters.
    """
    feedback = state.get("critic", "")

    system_prompt = get_node_prompt("analyst")
    user_prompt = (
        f"Question : {state['question']}\n"
        f"Documentary context verified & Visual Context :\n{state['context']}\n\n"
    )
    
    if feedback:
        user_prompt += f"Warning, critic signaled the following issue in the previous round : {feedback}\n"
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    return {
        "analysis": response.content,
    }


# ===========================================
# NODE 4 : CRITIC
# ===========================================

def critic_node(state: AgentState):
    """Verifies strict compliance with the documents and assigns a score (anti-hallucination)."""
    system_prompt = get_node_prompt("critic")
    user_prompt = (
        f"Question : {state['question']}\n"
        f"Source Context :\n{state['context']}\n\n"
        f"Draft response to evaluate :\n{state['analysis']}\n"
    )

    try:
        evaluation: CriticOutput = critic_evaluator.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)]
        )

        return {
            "critic": evaluation.feedback_message,
            "is_valid": evaluation.is_valid,
        }
    except Exception as e:
        return {
            "critic": f"Error during critic evaluation: {e}",
            "is_valid": False
        }


# ===========================================
# NODE 5 : SELF-REFLECTION
# ===========================================

def self_reflection_node(state: AgentState):
    """
    Diagnose the root cause of why the draft was flagged by the critic and provide a corrective plan.
    1. If the critic flagged missing data points, suggest a re-search plan with specific queries.
    2. If the critic flagged hallucinations or misformatting, provide clear instructions for the analyst to correct the draft.
    3. Return a structured output with the root cause, action type, and corrective plan
    """
    system_prompt = get_node_prompt("self_reflection")
    user_prompt = (
        f"User Question: {state['question']}\n\n"
        f"Available Context: {state.get('context', [])}\n\n"
        f"Draft Analysis: {state.get('analysis', '')}\n\n"
        f"Critic Feedback: {state.get('critic', '')}\n"
    )

    try:
        reflection: ReflectionOutput = reflection_evaluator.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        if reflection.action_type == "re_search":
            new_plan = f"Missing data points to find: {', '.join(reflection.new_search_queries)}"
        else:
            new_plan = f"Analyst correction instructions: {reflection.critique_for_analyst}"

        return {
            "reflection": reflection.root_cause,
            "action_type": reflection.action_type,
            "critic": reflection.critique_for_analyst if reflection.action_type == "re_write" else state.get("critic", ""),
            "plan": new_plan,
            "reflection_cycle": state.get("reflection_cycle", 0) + 1
        }
    except Exception as e:
        return {
            "reflection": f"Evaluation error: {e}",
            "action_type": "re_search",
            "plan": "Re-try searching for missing technical elements.",
            "reflection_cycle": state.get("reflection_cycle", 0) + 1
        }


# ===========================================
# NODE 6 : SYNTHESIZER
# ===========================================

def synthesizer_node(state: AgentState):
    """
    Drafts the final answer in a clear, structured format with Markdown tables and bullet points, strictly based on verified context.
    The synthesizer must not hallucinate or invent any data, and must not extrapolate or calculate values with invented parameters.
    Extracts certified images referenced in the context and appends them as Markdown image links with captions derived from the file name without hallucinating or inventing URL paths.
    """
    system_prompt = get_node_prompt("synthesizer")
    context_list = state.get("context", [])
    context_str = "\n".join(context_list)

    key_findings = [
        c for c in context_list
        if "[inspect_pdf_page_image]" in c or "[calculate_aerospace_math]" in c
    ]
    findings_str = "\n\n".join(key_findings) if key_findings else context_str

    user_prompt = (
        f"Question: {state['question']}\n\n"
        f"Verified Visual & Mathematical Findings:\n{findings_str}\n\n"
        f"Verified Technical Data:\n{state['analysis']}\n\n"
        f"Answer ONLY the user's question clearly and concisely using the verified numbers above."
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    final_output = response.content

    # Extract certified image paths recorded during document retrieval and visual inspection
    context_str = "\n".join(state.get("context", []))

    # Prioritize images explicitly inspected by the vision tool, or take the last retrieved image
    inspected_imgs = re.findall(r'\[Visual Inspection Page \d+\]:.*?\[Associated Page Image\]:\s*(\S+)', context_str, re.DOTALL)
    all_imgs = re.findall(r'\[Associated Page Image\]:\s*(\S+)', context_str)
    real_image_url = inspected_imgs[-1] if inspected_imgs else (all_imgs[-1] if all_imgs else "")

    # Programmatically append the first certified visual attachment to avoid hallucinated URLs
    if real_image_url:
        caption = Path(real_image_url).stem.replace("_", " ")
        final_output += f"\n\n### Document Reference\n![{caption}]({real_image_url})\n"

    return {"final_answer": final_output}
