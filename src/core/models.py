from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field


# Judge Result Schema (Pydantic Model)
class JudgeResult(BaseModel):
    """
    Structured critique and refinement decision from the Judge Agent.
    This schema is used to ensure LLM output is predictable
    """

    critique: str = Field(
        description="Detailed critique of the current summary and specific improvement suggestions."
    )
    score: int = Field(
        description="Numerical score (e.g., 1-10) reflecting the quality of the summary."
    )
    should_refine: bool = Field(
        description="Boolean indicating whether the summary needs refinement (True) or is acceptable (False)."
    )


# Agent State Schema (TypedDict)
# This dictionary holds the state information passed between agents in the LangGraph.
class AgentState(TypedDict):
    """
    The state for the summarization LangGraph nodes.

    This is passed between agents to maintain context and track progress.
    """

    # Required Input: The document to be summarized
    input_text: str

    # Internal Tracking: The full document broken into mangeable chunks
    document_chunks: List[str]

    # Output: The current summary being refined
    summary_draft: str

    # Internal Tracking: The result from the Judge Agent
    judge_result: Optional[JudgeResult]

    # Internal Tracking: Number of refinement iterations completed
    refinement_count: int

    # Configuration: Maximum number of refinement steps allowed
    max_refinement_steps: int
