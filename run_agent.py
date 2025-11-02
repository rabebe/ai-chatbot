"""
Main execution script for the Self-Correcting Summarization Agent.

This script handles command-line arguments, loads the input document,
and runs the LangGraph workflow to generate a refined summary.
"""

import argparse
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project imports
# Suppression required for block: load_dotenv() MUST run first to set Gemini API key.
from src.core.models import AgentState  # noqa: E402
from src.core.utils import setup_logging, ensure_data_directory  # noqa: E402
from src.core.document_processor import process_document  # noqa: E402
from src.core.agent_graph import agent_graph  # noqa: E402
from langchain_core.documents import Document  # noqa: E402

logger = logging.getLogger(__name__)


def load_document_text(filepath: Path) -> str:
    """Load the content of a document file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Input file not found: {filepath}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        sys.exit(1)


def run_agent(input_file: Path, max_steps: int):
    """
    Main function to execute the LangGraph summarization agent.

    Args:
        input_file: Path to the document to be summarized.
        max_steps: Maximum number of refinement iterations allowed.
    """
    logger.info(f"--- Starting Agent: Summarizing {input_file.name} ---")
    logger.info(f"Configuration: Max refinement steps set to {max_steps}")

    # 1. Load and Process Document
    document_text = load_document_text(input_file)
    document_chunks_objects: list[Document] = process_document(document_text)

    document_chunks_str: list[str] = [
        doc.page_content for doc in document_chunks_objects
    ]
    # 2. Check for sufficient content
    if not document_chunks_str:
        logger.error("Document is empty or too small to process.")
        return

    # 3. Prepare Initial State for LangGraph
    initial_state: AgentState = {
        # input_text is used by the judge node for context
        "input_text": document_text,
        "document_chunks": document_chunks_str,
        # The first chunk is used as the initial context for the first summarization step
        "summary_draft": "",
        "judge_result": None,
        "refinement_count": 0,
        "max_refinement_steps": max_steps,
    }

    # 4. Invoke the Compiled LangGraph
    logger.info("Invoking LangGraph agent...")
    try:
        # The .invoke() method runs the graph synchronously
        final_state = agent_graph.invoke(initial_state)

    except Exception as e:
        logger.error(f"An error occurred during LangGraph execution: {e}")
        return

    # 5. Report Final Results
    final_summary = final_state.get("summary_draft", "Summary not available.")
    final_count = final_state.get("refinement_count", 0)
    final_judge = final_state.get("judge_result")

    print("\n" + "=" * 80)
    print(
        f"       ✅ AGENT COMPLETE: Final Summary Generated after {final_count} Refinements"
    )
    print("=" * 80 + "\n")
    print(final_summary)
    print("\n" + "-" * 80)

    if final_judge:
        print(f"Final Judge Score: {final_judge.score} / 10")
        print(f"Final Judge Critique: {final_judge.critique}")

    logger.info(f"Agent finished in {final_count} steps.")


def main():
    """Parse arguments and start the agent."""
    parser = argparse.ArgumentParser(
        description="Run the Self-Correcting Summarization Agent using LangGraph."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the document file (.txt) to be summarized.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=3,
        help="Maximum number of refinement steps (loops) allowed. Defaults to 3.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level.",
    )

    args = parser.parse_args()

    # Setup environment and logging first
    ensure_data_directory()  # Ensures the 'data' dir exists (imported from utils)
    setup_logging(log_level=args.log_level)  # Sets up logging (imported from utils)

    # Run the main agent function
    run_agent(Path(args.input_file), args.max_steps)


if __name__ == "__main__":
    # Ensure correct working directory context
    # This prevents path issues when running from project root
    main()
