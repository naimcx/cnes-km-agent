from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.nodes import (
    orchestrator_node,
    researcher_node,
    analyst_node,
    critic_node,
    self_reflection_node,
    synthesizer_node
)

# ===========================================
# WORKFLOW EXECUTION PARAMETERS
# ===========================================

# Maximum number of iterations for the Actor-Critic loop
MAX_REFLECTION_CYCLES = 3


# ===========================================
# GRAPH DEFINITION
# ===========================================

# 1. Instantiate the StateGraph with the AgentState type
workflow = StateGraph(AgentState)

# 2. Add nodes to the graph
workflow.add_node("orchestrator", orchestrator_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("critic", critic_node)
workflow.add_node("self_reflection", self_reflection_node)
workflow.add_node("synthesizer", synthesizer_node)

# 3. Define the entry point of the workflow
workflow.set_entry_point("orchestrator")

# 4. Establish edges between nodes to define the flow of the workflow
workflow.add_edge("orchestrator", "researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", "critic")


# ===========================================
# CONDITIONAL ROUTING BASED ON NODE OUTPUTS
# ===========================================

def decide_after_critic(state: AgentState):
    """Decides whether to go to the synthesizer or to the self-reflection node based on the critic's feedback."""
    is_valid = state.get("is_valid", False)
    cycles = state.get("reflection_cycle", 0)
    
    # Exit condition
    if is_valid or cycles >= MAX_REFLECTION_CYCLES:
        return "synthesizer"
    return "self_reflection"

workflow.add_conditional_edges(
    "critic",
    decide_after_critic,
    {
        "synthesizer": "synthesizer",
        "self_reflection": "self_reflection"
    }
)

def decide_after_self_reflection(state: AgentState):
    """Decides whether to go back to the researcher for re-searching or to the analyst for re-writing."""
    action_type = state.get("action_type", "re_write")
    if action_type == "re_search":
        print("[Self-reflection] suggests re-searching. Going back to the researcher.")
        return "researcher"
    else:
        print("[Self-reflection] suggests re-writing. Going back to the analyst.")
        return "analyst"

workflow.add_conditional_edges(
    "self_reflection",
    decide_after_self_reflection,
    {
        "researcher": "researcher",
        "analyst": "analyst"
    }
)

# Terminate workflow execution after the synthesizer node
workflow.add_edge("synthesizer", END)


# ===========================================
# GRAPH COMPILATION
# ===========================================

app = workflow.compile()
