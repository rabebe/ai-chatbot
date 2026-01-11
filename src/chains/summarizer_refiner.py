import os
import logging
from typing import Dict, Any
from difflib import SequenceMatcher

from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from ..core.models import AgentState, JudgeResult
from ..core.utils import fuzzy_match_summary, save_summary

# -------------------
# Logging
# -------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------
# Initialize LLM
# -------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


# -------------------
# Utility Functions
# -------------------
def compute_diff_ratio(a: str, b: str) -> float:
    """Return similarity ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()


# -------------------
# Prompts
# -------------------
INITIAL_SUMMARY_PROMPT = """You are an expert technical writer.
Document Chunks:
{document_chunks}
---
Initial Summary Draft:"""

REFINEMENT_PROMPT = """You are an expert editor specializing in technical documentation.
Critique:
{critique}
Current Summary Draft:
{summary_draft}
---
Revised Summary:"""


# -------------------
# Worker Nodes
# -------------------
def summarizer_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- EXECUTING SUMMARIZER NODE ---")

    document_text = "\n\n---\n\n".join(state["document_chunks"])
    user_id = state["user_id"]

    # Check fuzzy cache first
    cached_output = fuzzy_match_summary(user_id, document_text)
    if cached_output:
        _, summary, score, critique_text = cached_output
        logger.info("Returning cached summary from fuzzy DB")
        return {
            "summary_draft": summary,
            "summary_history": [
                {"summary": summary, "critique": critique_text, "score": score}
            ],
            "fixes": [],
            "from_cache": True,
            "user_id": user_id,
        }

    # Build prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are an expert technical writer."),
            HumanMessage(
                content=INITIAL_SUMMARY_PROMPT.format(document_chunks=document_text)
            ),
        ]
    )

    # Invoke LLM safely
    try:
        summary = (prompt | llm | StrOutputParser()).invoke({})
        logger.info(f"LLM generated summary: {summary!r}")
    except Exception:
        logger.exception("LLM call failed")
        summary = "LLM returned no content. Check API or prompt."

    # Fallback if summary is empty
    if not summary.strip():
        summary = "LLM returned no content. Check API or prompt."

    save_summary(
        user_id=user_id,
        input_text=document_text,
        output_text=summary,
        score=0,
        critique_text="",
    )

    return {
        "summary_draft": summary,
        "summary_history": [{"summary": summary, "critique": "", "score": 0}],
        "fixes": [],
        "user_id": user_id,
    }


def judge_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- EXECUTING JUDGE NODE ---")

    document_text = "\n\n---\n\n".join(state["document_chunks"])
    summary_draft = state.get("summary_draft", "")
    if not summary_draft.strip():
        summary_draft = "No summary generated yet."

    STRICT_JUDGE_PROMPT = f"""
You are a strict professor evaluating a summary.

Document:
{document_text}

Summary:
{summary_draft}

Return JSON with:
- critique (string)
- score (integer 1–10)
- should_refine (boolean)
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are an expert university professor grader."),
            HumanMessage(content=STRICT_JUDGE_PROMPT),
        ]
    )

    try:
        result: JudgeResult = (prompt | llm.with_structured_output(JudgeResult)).invoke(
            {}
        )
        logger.info(f"Judge returned: {result}")
    except Exception:
        logger.exception("Judge LLM failed")
        result = JudgeResult(
            critique="Error generating critique", score=0, should_refine=False
        )

    # Convert to dict
    result_dict = result.model_dump()
    SCORE_THRESHOLD = 7
    result_dict["should_refine"] = result_dict.get("score", 0) < SCORE_THRESHOLD

    # Update summary and judge history
    summary_history = state.get("summary_history", [])
    if summary_history:
        summary_history[-1]["critique"] = result_dict.get("critique")
        summary_history[-1]["score"] = result_dict.get("score")
    else:
        summary_history.append(
            {
                "summary": summary_draft,
                "critique": result_dict.get("critique"),
                "score": result_dict.get("score"),
            }
        )

    judge_history = state.get("judge_history", [])
    judge_history.append(
        {
            "iteration": state.get("refinement_count", 0),
            "score": result_dict.get("score"),
            "critique": result_dict.get("critique"),
            "should_refine": result_dict.get("should_refine"),
        }
    )

    return {
        "judge_result": result_dict,
        "summary_draft": summary_draft,
        "summary_history": summary_history,
        "judge_history": judge_history,
        "user_id": state["user_id"],
    }


def refinement_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- EXECUTING REFINEMENT NODE ---")

    judge_result = state.get("judge_result", {})
    summary_draft = state.get("summary_draft", "")
    critique = judge_result.get("critique", "")

    refinement_count = state.get("refinement_count", 0) + 1
    max_steps = state.get("max_refinement_steps", 2)
    user_id = state["user_id"]

    if refinement_count > max_steps:
        logger.info("Max refinement steps reached.")
        return {
            "refinement_count": refinement_count,
            "summary_draft": summary_draft,
            "user_id": user_id,
            "judge_result": judge_result,
        }

    cache_key = f"CRITIQUE:{critique}|||DRAFT:{summary_draft}"
    cached_refined = fuzzy_match_summary(user_id, cache_key)
    if cached_refined:
        _, revised_summary, score, critique_text = cached_refined
        history = state.get("summary_history", [])
        history.append(
            {"summary": revised_summary, "critique": critique_text, "score": score}
        )
        return {
            "summary_draft": revised_summary,
            "summary_history": history,
            "from_cache": True,
            "refinement_count": refinement_count,
            "user_id": user_id,
        }

    # Prompt for refinement
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

    try:
        revised_summary = (prompt | llm | StrOutputParser()).invoke({})
        logger.info(f"Refinement node output: {revised_summary!r}")
    except Exception:
        logger.exception("Refinement LLM failed")
        revised_summary = summary_draft  # fallback

    # Track changes
    change_ratio = compute_diff_ratio(summary_draft, revised_summary)
    fixes = state.get("fixes", [])
    if change_ratio < 0.9:
        fixes.append(
            {
                "iteration": refinement_count,
                "before": summary_draft,
                "after": revised_summary,
                "change_ratio": change_ratio,
            }
        )

    # Save
    save_summary(
        user_id=user_id,
        input_text=cache_key,
        output=revised_summary,
        score=0,
        critique_text=critique or "",
    )

    history = state.get("summary_history", [])
    history.append({"summary": revised_summary, "critique": critique, "score": None})

    return {
        "summary_draft": revised_summary,
        "summary_history": history,
        "fixes": fixes,
        "refinement_count": refinement_count,
        "user_id": user_id,
    }


def decide_to_continue(state: AgentState) -> str:
    judge_result = state.get("judge_result")
    if not judge_result:
        return "end"
    if judge_result.get("should_refine") and state.get(
        "refinement_count", 0
    ) < state.get("max_refinement_steps", 2):
        return "refine"
    return "end"


def finalize_summaries(state: AgentState) -> Dict[str, Any]:
    logger.info("--- FINALIZING SUMMARIES ---")
    state["summary_history"] = state.get("summary_history", [])[-3:]
    return state
