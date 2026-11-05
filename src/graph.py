from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import researcher_node, analyst_node, critic_node

# 1. Instantiate the StateGraph with the AgentState type
workflow = StateGraph(AgentState)

# 2. Add nodes to the graph
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)
workflow.add_node("critic", critic_node)

# 3. Define the entry point of the workflow
workflow.set_entry_point("researcher")

# 4. Establish edges between nodes to define the flow of the workflow
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", "critic")

# 5. Conditional edges (Actor-Critic loop)
def decide_next_step(state: AgentState):
    """Decides whether to validate the answer or have the analyst redo it."""
    score = state.get("score", 0)
    iterations = state.get("iteration", 0)
    
    # If the score is good or if we have already made 2 attempts, we stop (safeguard)
    if score >= 7 or iterations >= 2:
        return END
    else:
        print(f"\n[Critical Alert] Insufficient score ({score}/10). Re-routing to the analyst...")
        return "analyst"

workflow.add_conditional_edges(
    "critic",
    decide_next_step,
    {
        END: END,
        "analyst": "analyst"
    }
)

# 6. Graph compilation
app = workflow.compile()

# 7. Interactive test
if __name__ == "__main__":
    question = "What is the different types of Ariane 6 and what engines do they use ?"
    print(f"--- Starting the query : {question} ---")
    
    initial_state = {
        "question": question,
        "iteration": 0
    }
    
    result = app.invoke(initial_state)
    
    print("\n================ FINAL RESULT ================")
    print(f"Number of iterations : {result['iteration']}")
    print(f"Validation score : {result['score']}/10")
    print(f"Reviewer's note :\n{result['critic_notes']}")
    print(f"\nValidated response :\n{result['final_answer']}")