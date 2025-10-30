import pytest
from unittest.mock import patch

from src.core.models import AgentState, JudgeResult
from src.core.agent_graph import create_agent_graph

# --- Mock JudgeResults for predictable flow control (used for consistency in mocks) ---

mock_judge_result_refine = JudgeResult(
    critique="Initial draft fails quality check. Must be refined.",
    score=4,
    should_refine=True,
)

mock_judge_result_pass = JudgeResult(
    critique="Final draft passes all quality checks.", score=9, should_refine=False
)


@pytest.fixture
def graph_initial_state() -> AgentState:
    """Provides the required initial state for invoking the full graph."""
    return {
        "input_text": "This is a test document with enough content to be summarized.",
        "summary_draft": "",
        "document_chunks": [
            "Chunk 1: The main idea is about the self-correcting loop.",
            "Chunk 2: The judge will be lenient on the first two steps.",
            "Chunk 3: The process should end after the third refinement.",
        ],
        "judge_result": None,
        "refinement_count": 0,
        "max_refinement_steps": 3,
    }


# 🌟 FIX: Patching all worker nodes at their assumed source (src.chains.summarizer_refiner)
@patch("src.core.agent_graph.refinement_node")
@patch("src.core.agent_graph.judge_node")
@patch("src.core.agent_graph.summarizer_node")
def test_full_graph_runs_to_end(
    mock_summarizer_node, mock_judge_node, mock_refinement_node, graph_initial_state
):
    """
    Tests the LangGraph flow by mocking all logic, verifying only the connections/wiring.

    We use a context manager to patch the conditional function (router) and recompile the
    graph to ensure the mock is correctly referenced by the conditional edge.
    """

    # 1. Configure the Mocks for the Nodes

    # MOCK SUMMARIZER (Node 1): Guarantees a summary_draft is immediately set.
    mock_summarizer_node.return_value = {
        "summary_draft": "This is an initial mocked summary."
    }

    # MOCK JUDGE (Nodes 2 & 4): Return fixed results for consistency.
    mock_judge_node.side_effect = [
        {"judge_result": mock_judge_result_refine},
        {"judge_result": mock_judge_result_pass},
    ]

    # MOCK REFINEMENT (Node 3): Guarantees the counter is incremented and summary is updated.
    mock_refinement_node.return_value = {
        "refinement_count": 1,
        "summary_draft": "Refined Summary V1 (Mocked)",
    }

    print(
        "\n[Mocks Configured: Router forced to execute one refinement loop via context manager and recompilation.]"
    )

    # 2. Patch the Conditional Router Function and RECOMPILE THE GRAPH
    # This is the "nuclear option" fix for patching conflicts in LangGraph.
    with patch("src.core.agent_graph.decide_to_continue") as mock_decide_to_continue:
        # MOCK ROUTER: Forces the graph to loop once, then end.
        mock_decide_to_continue.side_effect = ["refine", "end"]

        # 3. Recompile the graph so it captures the patched decide_to_continue reference.
        test_graph = create_agent_graph()

        # 4. Run the entire compiled graph
        final_state = test_graph.invoke(graph_initial_state)

        # 5. Assertions to confirm the loop worked

        # Assert that the refinement loop executed exactly once
        assert final_state["refinement_count"] == 1

        # The final summary draft should be the one from the refinement mock
        assert final_state["summary_draft"] == "Refined Summary V1 (Mocked)"

        # Assert that the worker nodes were called the correct number of times
        assert mock_summarizer_node.call_count == 1
        assert mock_refinement_node.call_count == 1
        assert mock_judge_node.call_count == 2
        assert mock_decide_to_continue.call_count == 2
