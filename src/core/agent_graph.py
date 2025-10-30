from langgraph.graph import StateGraph, END
from src.core.models import AgentState
from src.chains.summarizer_refiner import (
    summarizer_node,
    refinement_node,
    judge_node,
    decide_to_continue,
)


def create_agent_graph():
    """
    Assembles the LangGraph StateGraph defining the iterative critique and refinement loop.

    The flow is: Summarizer -> Judge -> (Loop or End).
    """

    # 1. Initialize the StateGraph
    workflow = StateGraph(AgentState)

    # 2. Add the worker nodes (We are explicitly NOT adding 'decide_to_continue' as a node,
    # as it should only be used as a conditional router function.)
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("judge", judge_node)
    workflow.add_node("refine", refinement_node)

    # 3. Define Edges

    # Start: Summarizer node generates initial draft
    workflow.set_entry_point("summarizer")

    # Edge 1: After Summarizer, always go to Judge for critique
    workflow.add_edge("summarizer", "judge")

    # Edge 2: After Refinement, always go back to Judge
    workflow.add_edge("refine", "judge")

    # Edge 3: Conditional edge after Judge to determine next step
    # The 'decide_to_continue' function (which is mocked in the test)
    # is executed to choose the next node ("refine" or END).
    workflow.add_conditional_edges(
        "judge",
        decide_to_continue,
        {
            "refine": "refine",
            "end": END,
        },
    )

    # Compile the graph
    app = workflow.compile()
    print("LangGraph agent compiled successfully.")

    return app


# The final callable graph object
agent_graph = create_agent_graph()
