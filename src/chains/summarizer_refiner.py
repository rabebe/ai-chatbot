import os
import logging
from typing import Dict, Any
from difflib import SequenceMatcher

from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from ..core.models import AgentState, JudgeResult
from ..core.utils import init_cache_db, fuzzy_match_cache, save_to_cache

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# Initialize cache database
init_cache_db()


# --- Utility ---
def compute_diff_ratio(a: str, b: str) -> float:
    """Return similarity ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()


# --- Prompts ---
INITIAL_SUMMARY_PROMPT = """You are an expert technical writer. Your task is to generate a comprehensive, coherent, and highly accurate summary of the provided document chunks.

Document Chunks:
{document_chunks}

---
Initial Summary Draft:"""

REFINEMENT_PROMPT = """You are an expert editor specializing in technical documentation. Your task is to refine the existing summary draft based on the Judge's specific critique.

Critique:
{critique}

Current Summary Draft:
{summary_draft}

---
Revised Summary:"""


# --- Worker Nodes ---
def summarizer_node(state: AgentState) -> Dict[str, Any]:
    logger.info("---EXECUTING SUMMARIZER NODE---")
    document_chunks = state["document_chunks"]
    document_text = "\n\n---\n\n".join(document_chunks)

    # --- Get user_id from the state ---
    user_id = state["user_id"]

    # --- Check cache first ---
    # NOTE: The cache key should include the document text, but fuzzy_match_cache
    # takes user_id and the input text (which is the combined document text here).
    cached_output = fuzzy_match_cache(user_id, document_text)

    if cached_output:
        # fuzzy_match_cache returns (old_input, old_output). We only want the output.
        logger.info("Found cached summary (fuzzy match). Skipping LLM call.")
        # Assuming cached_output is a tuple (old_input, old_output) or None
        if isinstance(cached_output, tuple):
            summary = cached_output[1]
        else:
            summary = (
                cached_output  # fallback if fuzzy_match only returned the output string
            )

        return {
            "summary_draft": summary,
            "summary_history": [summary],
            "fixes": [],
            "from_cache": True,
            "user_id": user_id,  # Ensure user_id is passed through
        }

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are an expert technical writer."),
            HumanMessage(
                content=INITIAL_SUMMARY_PROMPT.format(document_chunks=document_text)
            ),
        ]
    )

    chain = prompt | llm | StrOutputParser()

    try:
        summary = chain.invoke({})
        logger.info("Initial summary generated.")

        # Save to cache (using the input document text as the fuzzy matching key)
        save_to_cache(user_id, document_text, summary)

        return {
            "summary_draft": summary,
            "summary_history": [summary],
            "fixes": [],
            "user_id": user_id,  # Ensure user_id is passed through
        }
    except Exception as e:
        logger.error(f"Summarizer failed: {e}")
        # Note: Added user_id to the error return state
        return {
            "summary_draft": f"ERROR: {e}",
            "refinement_count": state.get("refinement_count", 0) + 1,
            "user_id": user_id,
        }


def refinement_node(state: AgentState) -> Dict[str, Any]:
    logger.info("---EXECUTING REFINEMENT NODE---")
    summary_draft = state["summary_draft"]
    judge_result = state.get("judge_result")
    critique = (
        judge_result.critique if judge_result else "No specific critique provided."
    )
    refinement_count = state.get("refinement_count", 0) + 1
    max_steps = state.get("max_refinement_steps", 3)

    if refinement_count > max_steps:
        logger.warning(f"Refinement limit reached ({max_steps} steps).")
        fallback_score = judge_result.score if judge_result else 0
        judge_result = JudgeResult(
            critique="Refinement limit reached.",
            score=fallback_score,
            should_refine=False,
        )
        return {
            "judge_result": judge_result,
            "refinement_count": refinement_count,
            "summary_draft": summary_draft,
            "user_id": state["user_id"],  # Ensure user_id is passed through
        }

    user_id = state["user_id"]

    # --- Check cache for refined summary ---
    # Key includes both the summary and the critique for specific matching
    cache_input_key = f"CRITIQUE:{critique} ||| DRAFT:{summary_draft}"
    cached_refined_tuple = fuzzy_match_cache(user_id, cache_input_key)

    if cached_refined_tuple:
        # fuzzy_match_cache returns (old_input, old_output). We only want the output.
        revised_summary = cached_refined_tuple[1]
        logger.info("Found cached refined summary. Skipping LLM call.")

        return {
            "summary_draft": revised_summary,
            "summary_history": state.get("summary_history", []) + [revised_summary],
            "fixes": state.get("fixes", []),
            "from_cache": True,
            "refinement_count": refinement_count,
            "user_id": user_id,  # Ensure user_id is passed through
        }

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

    chain = prompt | llm | StrOutputParser()

    try:
        revised_summary = chain.invoke({})
        logger.info(f"Summary refined (Attempt {refinement_count}).")

        # Track fixes
        fixes = state.get("fixes", [])
        change_ratio = compute_diff_ratio(summary_draft, revised_summary)
        if change_ratio < 0.98:
            fixes.append(
                {
                    "iteration": refinement_count,
                    "before": summary_draft,
                    "after": revised_summary,
                    "change_ratio": change_ratio,
                }
            )

        # Update history
        history = state.get("summary_history", [])
        history.append(revised_summary)

        # Save to cache (using the input key used for fuzzy matching)
        save_to_cache(user_id, cache_input_key, revised_summary)

        return {
            "summary_draft": revised_summary,
            "refinement_count": refinement_count,
            "fixes": fixes,
            "summary_history": history,
            "user_id": user_id,
        }
    except Exception as e:
        logger.error(f"Refiner failed: {e}")
        fallback_result = JudgeResult(
            critique=f"Refiner failed: {e}", score=0, should_refine=False
        )
        return {
            "judge_result": fallback_result,
            "refinement_count": refinement_count,
            "summary_draft": summary_draft,
            "user_id": user_id,
        }


def judge_node(state: AgentState) -> Dict[str, Any]:
    """
    Judge node evaluates the current summary against the original document chunks.
    """
    logger.info("---EXECUTING JUDGE NODE (STRICT PROFESSOR)---")

    document_chunks = state["document_chunks"]
    summary_draft = state["summary_draft"]
    document_text = "\n\n---\n\n".join(document_chunks)
    user_id = state["user_id"]

    STRICT_JUDGE_PROMPT = f"""
You are a university professor evaluating a student's summary against the original document.
Be extremely critical and thorough. Check for:
- Contradictions or conflicting statements
- Missing key points
- Repetition or redundant information
- Coherence, clarity, and conciseness
- Accuracy with respect to the original text

Document Chunks:
{document_text}

Summary Draft:
{summary_draft}

Your output MUST be a JSON object with:
- critique: detailed feedback pointing out errors, contradictions, or missing content
- score: integer 1-10
- should_refine: true if any major issues exist, false otherwise
Respond strictly — do NOT inflate scores.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are an expert university professor grader."),
            HumanMessage(content=STRICT_JUDGE_PROMPT),
        ]
    )

    structured_llm = llm.with_structured_output(JudgeResult)
    chain = prompt | structured_llm

    try:
        result: JudgeResult = chain.invoke({})
        logger.info(
            f"Judge decision: Score={result.score}, Refine={result.should_refine}"
        )

        # Safeguard logic
        if result.score <= 8:
            result.should_refine = True
        else:
            result.should_refine = False

        # Track history
        history = state.get("judge_history", [])
        history.append(
            {
                "iteration": state.get("refinement_count", 0),
                "score": result.score,
                "critique": result.critique,
                "should_refine": result.should_refine,
            }
        )

        return {
            "judge_result": result,
            "summary_draft": summary_draft,
            "judge_history": history,
            "user_id": user_id,
        }

    except (Exception, ValidationError) as e:
        logger.error(f"Judge failed: {e}. Stopping refinement.")
        fallback_result = JudgeResult(
            critique=f"Judge failed ({type(e).__name__}).", score=0, should_refine=False
        )
        history = state.get("judge_history", [])
        history.append(
            {
                "iteration": state.get("refinement_count", 0),
                "score": 0,
                "critique": str(e),
                "should_refine": False,
            }
        )
        return {
            "judge_result": fallback_result,
            "summary_draft": summary_draft,
            "judge_history": history,
            "user_id": user_id,
        }


def decide_to_continue(state: AgentState) -> str:
    judge_result: JudgeResult = state.get("judge_result")
    if not judge_result:
        logger.warning("Judge result missing. Ending loop.")
        return "end"

    refinement_count = state.get("refinement_count", 0)
    max_steps = state.get("max_refinement_steps", 3)

    logger.info(
        f"Decide to continue? should_refine={judge_result.should_refine}, "
        f"refinement_count={refinement_count}, max_steps={max_steps}, score={judge_result.score}"
    )

    #

    if judge_result.score < 8:
        if refinement_count < max_steps:
            return "refine"
        else:
            logger.warning(
                f"Max refinement steps ({max_steps}) reached. Ending process."
            )
            return "end"
    else:
        return "end"
