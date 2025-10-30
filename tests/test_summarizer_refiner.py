import pytest
from typing import Dict, Any

from src.core.models import AgentState, JudgeResult
from src.chains.summarizer_refiner import (
    summarizer_node,
    judge_node,
    refinement_node,
    decide_to_continue,
)


@pytest.fixture
def initial_state_fixture() -> AgentState:
    """Fixture for a standard initial state."""
    return {
        "document": "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.",
        "summary_draft": "",
        "document_chunks": [
            "Chunk 1: The quick brown fox jumps over the lazy dog. It is very fast.",
            "Chunk 2: The brown fox then runs to the river. It lives in the woods.",
        ],
        "judge_result": None,
        "refinement_count": 0,
        "max_refinement_steps": 3,
    }


# Mock the JudgeResult for predictable testing
mock_judge_result_refine = JudgeResult(
    critique="Summary is too short and misses key details.", score=6, should_refine=True
)

mock_judge_result_pass = JudgeResult(
    critique="Summary is excellent and meets all criteria.",
    score=9,
    should_refine=False,
)


def test_summarizer_node_returns_summary(initial_state_fixture):
    """Tests that the summarizer node executes and updates the 'summary_draft' key."""

    # Mock the LLM to return a predictable string to testing purposes
    # mock_chain_result = "This is a generated initial summary draft."

    # Mock the internal LLM chain call within the node
    # def mock_invoke(*args, **kwargs):
    #     return mock_chain_result

    new_state: Dict[str, Any] = summarizer_node(initial_state_fixture)

    # Assert that the new state contains the expected key
    assert "summary_draft" in new_state
    assert isinstance(new_state["summary_draft"], str)
    assert len(new_state["summary_draft"]) > 0


def test_judge_node_updates_judge_result(initial_state_fixture):
    """Tests that the judge node updates the judge_result field."""

    # # Mock the LLM to return a predictable JudgeResult object (simulating structured output)
    #     def mock_structured_invoke(*args, **kwards):
    #         return mock_judge_result_refine

    initial_state_fixture["Current_summary"] = "A draft to be judged."
    new_state: Dict[str, Any] = judge_node(initial_state_fixture)

    # Assert that the new state contains the expected key and type
    assert "judge_result" in new_state
    assert isinstance(new_state["judge_result"], JudgeResult)


def test_decide_to_continue_refine(initial_state_fixture):
    """Tests that decision is 'refine' when should_refine is True."""
    initial_state_fixture["judge_result"] = mock_judge_result_refine
    result = decide_to_continue(initial_state_fixture)
    assert result == "refine"


def test_refinement_node_increments_count(initial_state_fixture):
    """Tests that the refinement node increments the refinement count."""

    initial_state_fixture["summary_draft"] = "Old summary."
    initial_state_fixture["judge_result"] = mock_judge_result_refine

    # Simulate first refinement
    new_state: Dict[str, Any] = refinement_node(initial_state_fixture)
    assert "refinement_count" in new_state
    assert new_state.get("refinement_count") == 1


def test_refinement_node_stops_at_max_steps(initial_state_fixture):
    """Tests that the refinement node stops the loop if max steps"""

    initial_state_fixture["refinement_count"] = 3
    initial_state_fixture["msx_refinement_steps"] = 3
    initial_state_fixture["judge_result"] = mock_judge_result_refine

    new_state: Dict[str, Any] = refinement_node(initial_state_fixture)

    # Assert that the resulting judge_result force an exit
    judge_result: JudgeResult = new_state["judge_result"]
    assert not judge_result.should_refine
    assert judge_result.critique == "Refinement limit reached."

    # Verify the conditional edge respects the new state
    final_decision: str = decide_to_continue(new_state)
    assert final_decision == "end"
