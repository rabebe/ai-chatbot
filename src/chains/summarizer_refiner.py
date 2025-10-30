import os
import logging
import json
from typing import Dict, Any

from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from pydantic import ValidationError

from ..core.models import AgentState, JudgeResult

# --- Configuration ---
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM (Ensure GOOGLE_API_KEY is set in your environment)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# --- Prompts ---

# 1. Initial Summarization Prompt
INITIAL_SUMMARY_PROMPT = """You are an expert technical writer. Your task is to generate a comprehensive, coherent, and highly accurate summary of the provided document chunks.

The summary must be easy to read and synthesize information from all chunks.

Document Chunks:
{document_chunks}

---
Initial Summary Draft:"""

# 2. Refinement Prompt
REFINEMENT_PROMPT = """You are an expert editor specializing in technical documentation. Your task is to refine the existing summary draft based on the Judge's specific critique.

The goal is to produce a final, high-quality, and accurate summary.

Critique:
{critique}

Current Summary Draft:
{summary_draft}

---
Revised Summary:"""

# 3. Judge Prompt (Critique and Decision)
JUDGE_PROMPT = """You are the ultimate quality control Judge. Your role is to evaluate a summary draft against the original document chunks.

Your decision must be returned STRICTLY as a JSON object that matches the JudgeResult schema.

Evaluation Criteria:
1. Accuracy: Does the summary contain any factual errors or misrepresentations?
2. Completeness: Does the summary cover all major points mentioned in the document?
3. Coherence: Is the summary well-written, logically structured, and easy to understand?
4. Conciseness: Is the summary efficient and to the point?

Your final output MUST be a JSON object with three fields:
- critique: Your detailed feedback on the current summary draft.
- score: An integer score from 1 (Poor) to 10 (Perfect).
- should_refine: A boolean (true/false). Set to 'true' if the score is less than 8 or if any major errors exist. Set to 'false' if the summary is acceptable.

Document Chunks:
{document_chunks}

Current Summary Draft to be Judged:
{summary_draft}

---
Judge Result (JSON format ONLY):
"""

# --- Worker Nodes ---


def summarizer_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates the initial draft of the summary from the document chunks.
    """
    logger.info("---EXECUTING SUMMARIZER NODE---")

    # Extract needed state variables
    document_chunks = state["document_chunks"]

    # Combine chunks into a single string for the prompt
    document_text = "\n\n---\n\n".join(document_chunks)

    # Prepare the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are an expert technical writer."),
            HumanMessage(
                content=INITIAL_SUMMARY_PROMPT.format(document_chunks=document_text)
            ),
        ]
    )

    # Create the runnable chain
    chain = prompt | llm | StrOutputParser()

    try:
        # Invoke the chain
        summary = chain.invoke({})
        logger.info("Initial summary generated.")

        # Update the state
        return {"summary_draft": summary}
    except Exception as e:
        logger.error(f"Summarizer failed: {e}")
        # In case of failure, prevent infinite loop and use error message as summary
        return {
            "summary_draft": f"ERROR: Summarizer failed to generate summary. {e}",
            "refinement_count": state.get("refinement_count", 0) + 1,
        }


def refinement_node(state: AgentState) -> Dict[str, Any]:
    """
    Refines the summary draft based on the judge's critique.
    """
    logger.info("---EXECUTING REFINEMENT NODE---")

    # Extract needed state variables
    summary_draft = state["summary_draft"]
    judge_result = state["judge_result"]
    critique = (
        judge_result.critique if judge_result else "No specific critique provided."
    )

    # Check for loop limit
    refinement_count = state.get("refinement_count", 0) + 1
    max_steps = state.get("max_refinement_steps", 3)

    if refinement_count > max_steps:
        logger.warning(f"Refinement limit reached ({max_steps} steps). Exiting loop.")
        # If max steps reached, set should_refine to False to end the graph
        judge_result = JudgeResult(
            critique="Refinement limit reached.",
            score=state.get(
                "judge_result", JudgeResult(critique="", score=0, should_refine=False)
            ).score,
            should_refine=False,
        )
        return {"judge_result": judge_result, "refinement_count": refinement_count}

    # Prepare the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are an expert editor."),
            HumanMessage(
                content=REFINEMENT_PROMPT.format(
                    critique=critique, summary_draft=summary_draft
                )
            ),
        ]
    )

    # Create the runnable chain
    chain = prompt | llm | StrOutputParser()

    try:
        # Invoke the chain
        revised_summary = chain.invoke({})
        logger.info(f"Summary refined (Attempt {refinement_count}).")

        # Update the state
        return {
            "summary_draft": revised_summary,
            "refinement_count": refinement_count,
        }
    except Exception as e:
        logger.error(f"Refiner failed: {e}")
        # In case of failure, prevent infinite loop
        judge_result = JudgeResult(
            critique=f"Refiner failed: {e}", score=0, should_refine=False
        )
        return {"judge_result": judge_result, "refinement_count": refinement_count}


def judge_node(state: AgentState) -> Dict[str, Any]:
    """
    Critiques the current summary draft and decides whether further refinement is needed.
    """
    logger.info("---EXECUTING JUDGE NODE---")

    # Extract needed state variables
    document_chunks = state["document_chunks"]
    summary_draft = state["summary_draft"]

    # Combine chunks into a single string for the judge
    document_text = "\n\n---\n\n".join(document_chunks)

    # Prepare the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="You are the ultimate quality control Judge. Your output must be valid JSON matching the JudgeResult schema."
            ),
            HumanMessage(
                content=JUDGE_PROMPT.format(
                    document_chunks=document_text, summary_draft=summary_draft
                )
            ),
        ]
    )

    # Create the structured output chain
    structured_llm = llm.with_structured_output(JudgeResult)
    chain = prompt | structured_llm

    # Attempt to invoke and parse the structured output
    try:
        result: JudgeResult = chain.invoke({})
        logger.info(
            f"Judge decision: Score={result.score}, Refine={result.should_refine}"
        )

        # Update the state
        return {"judge_result": result}

    except (
        OutputParserException,
        ValidationError,
        json.JSONDecodeError,
        Exception,
    ) as e:
        # Handle cases where the LLM fails to output valid JSON
        logger.error(
            f"Judge failed to produce valid structured output: {e}. Forcing refinement=False."
        )

        # Create a fallback result that stops the loop
        fallback_result = JudgeResult(
            critique=f"Judge failed to parse output ({type(e).__name__}). Stopping loop.",
            score=0,
            should_refine=False,
        )
        return {"judge_result": fallback_result}


# --- Conditional Edges ---


def decide_to_continue(state: AgentState) -> str:
    """
    Conditional edge function that determines the next step based on the Judge's result.
    """
    judge_result = state.get("judge_result")
    if not judge_result:
        logger.warning("Judge result missing. Forcing exit.")
        return "end"

    # If the Judge says we need refinement, go to the refinement node
    if judge_result.should_refine:
        # Check refinement count to prevent infinite loops
        refinement_count = state.get("refinement_count", 0)
        max_steps = state.get("max_refinement_steps", 3)

        if refinement_count < max_steps:
            logger.info("Refinement needed. Continuing loop.")
            return "refine"
        else:
            logger.warning(
                f"Max refinement steps ({max_steps}) reached. Ending process."
            )
            return "end"
    else:
        logger.info("Summary acceptable. Ending process.")
        return "end"
