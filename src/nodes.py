import re
import json
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from src.tools import search_docs_hybrid, inspect_pdf_page_image, search_web, fetch_wikipedia
from src.config import llm
from src.state import AgentState
from schema.pydantic_schema import CriticOutput, ReflectionOutput, OrchestratorPlan


# =============================================
# LOAD ENVIRONMENT VARIABLES AND INITIALIZE LLM
# =============================================

load_dotenv()

INTERNAL_TOOLS = [search_docs_hybrid, inspect_pdf_page_image]
EXTERNAL_TOOLS = [search_web, fetch_wikipedia]
ALL_TOOLS = [search_docs_hybrid, inspect_pdf_page_image, search_web, fetch_wikipedia]
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
    system_prompt = (
        "You are the chief mission planner for a CNES technical assistant.\n"
        "Classify the query scope:\n"
        "- 'internal': specifically about Ariane 6, its manual, stages, fairing, interfaces, or CNES procedures.\n"
        "- 'external': physics theory, general astrodynamics, space law, or launch competitors (Falcon 9, Vulcan, New Glenn, Starship).\n"
        "- 'hybrid': cross-comparison between Ariane 6 and external rockets or external physics principles.\n"
        "CRITICAL SEARCH LANGUAGE RULES:\n"
        "Formulate search queries strictly in concise English technical keywords"
    )
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

    system_prompt = (
        "You are an expert technical researcher for CNES.\n"
        "Perform precise, fact-grounded investigation to collect verified documentary evidence.\n\n"
        "WORKFLOW:\n"
        "1. Issue targeted queries with 'search_docs_hybrid' using English technical terms.\n"
        "2. Examine returned chunks: it's announced figures that are printed on the subsequent page "
        "(e.g. text on page 83 announces Figure 5.3.1a on page 84; "
        "3. When search results reference a diagram, figure, or drawing (e.g. Figure 5.3.1a on page 84, "
        "or Picture 5.3.4.a on page 89), IMMEDIATELY call 'inspect_pdf_page_image' with that page number to extract dimensions, mechanisms or to see if the drawing is relevant.\n"
        "4. Once both text and visual measurements are gathered, conclude with a factual synthesis once all requested metrics are found."
    )

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
            context.append(response.content)
            break

        # 2. Iterate through each tool call, invoke the corresponding tool, and append the results to context and history
        for tool_call in tool_calls:
            t_name = tool_call["name"]
            t_args = tool_call.get("args", {}) or tool_call.get("parameters", {})
            t_id = tool_call.get("id", f"call_{t_name}_{step}")

            # Inject the original question into the tool arguments if the tool is 'inspect_pdf_page_image' and no question is provided
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

    system_prompt = (
        "You are a technical analyst for CNES Space Transport.\n"
        "DATA FIDELITY & INTEGRITY RULES:\n"
        "1. Every single parameter, component name, dimension (Ø), or unit MUST be transcribed strictly from context.\n"
        "2. GROUND vs FLIGHT HARDWARE: Never confuse ground integration buildings (e.g. clean rooms, BAF-HE) "
        "with launcher flight hardware (fairing, stage structures, adapter interfaces).\n"
        "3. ACCESSIBILITY vs SEPARATION: Differentiate access door panels (and their explicit functions) from "
        "pyrotechnic separation lines (HSS/VSS) and separation actuators (spring packages).\n"
        "4. VOLUMES & ENVELOPES: If a user manual defines an envelope via boundary dimensions (diameters, lengths) "
        "without publishing an integrated volume in m³, explicitly state that the envelope is defined geometrically.\n"
        "5. NEVER extrapolate or calculate values with invented parameters. Provide verified sources for every claim."
    )
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
    system_prompt = (
        "You are the quality control officer for CNES space documentation.\n"
        "Audit the analyst's draft against the source context:\n"
        "- Flag any invented dimension, assumed height/volume, or unverified specification.\n"
        "- Ensure access doors are not confused with separation systems.\n"
        "- Approve only if every claim is grounded in the text or visual reports."
    )
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
    system_prompt = (
        "You are a self-reflection supervisor for a CNES technical assistant.\n"
        "Determine the root cause of why the draft was flagged by the critic:\n"
        "- Choose 're_search': if ANY specific metric, figure, or requirement requested by the user "
        "is missing from the context (whether it requires text retrieval or diagram inspection). "
        "Provide clear suggestions for what to look up.\n"
        "- Choose 're_write': ONLY if all required data points already exist clearly within the retrieved context, "
        "but the analyst hallucinated, misformatted, or made an arithmetic error."
    )
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
    system_prompt = (
        "You are an expert technical communicator for CNES.\n"
        "Format the verified dimensions into clear markdown tables and bullet points.\n"
        "STRICT VISUAL LINK RULES:\n"
        "DO NOT output any image markdown link (e.g., ![...](...)). "
        "The system automatically appends verified document figures to the end of your response."
    )
    user_prompt = (
        f"Question: {state['question']}\n\n"
        f"Verified Technical Data:\n{state['analysis']}\n\n"
        f"Review: {state.get('critic', 'Approved')}\n"
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    final_output = response.content

    # Extract certified image paths recorded during document retrieval and visual inspection
    context_str = "\n".join(state.get("context", []))

    # Use regex to find the first occurrence of an associated page image in the context
    match_img = re.search(r'\[Associated Page Image\]:\s*(\S+)', context_str)
    real_image_url = match_img.group(1) if match_img else ""

    # Programmatically append the first certified visual attachment to avoid hallucinated URLs
    if real_image_url:
        caption = Path(real_image_url).stem.replace("_", " ")
        final_output += f"\n\n### Document Reference\n![{caption}]({real_image_url})\n"

    return {"final_answer": final_output}
