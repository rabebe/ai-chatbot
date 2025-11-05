import logging
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

# --- Configuration & Setup ---

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file (for local testing)
load_dotenv()

# --- Import the necessary components from your project ---
# Suppression required for block: load_dotenv() MUST run first to set Gemini API key.
from src.core.agent_graph import agent_graph  # noqa: E402
from src.core.models import AgentState, JudgeResult  # noqa: E402
from src.core.document_processor import process_document  # noqa: E402


# --- PATH CONFIGURATION ---
# Calculate the absolute paths to the templates and static folders
script_dir = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(script_dir, "templates")
static_path = os.path.join(script_dir, "static")  # <-- THIS IS THE PATH FOR YOUR LOGO

# Initialize Flask app
app = Flask(
    __name__,
    template_folder=template_path,
    static_folder=static_path,  # <-- THIS TELLS FLASK WHERE TO FIND 'logo.png'
)

logger.info(f"Flask template folder set to: {app.template_folder}")
logger.info(f"Flask static folder set to: {app.static_folder}")

CORS(app)
logger.info("Flask-CORS initialized, allowing all origins (*).")

# --- API Endpoints ---


@app.route("/", methods=["GET"])
@app.route("/index.html", methods=["GET"])
def serve_index():
    """Returns the main index.html file from the 'templates' folder."""
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize_document():
    """
    Accepts a document string, runs it through the self-correcting LangGraph agent,
    and returns the final summary and process details.
    """
    logger.info("Received request for summarization.")

    # 1. Input Validation
    try:
        data = request.get_json()
        document = data.get("document")

        # --- FIX: Get max_refinement_steps from the user's request ---
        max_steps = data.get("max_refinement_steps", 3)  # Default to 3 if not provided

        # Check for minimum length to ensure we have something meaningful to summarize
        if not document or not isinstance(document, str) or len(document) < 100:
            return jsonify(
                {
                    "error": "Missing or invalid 'document' field. Minimum 100 characters required.",
                    "usage": "POST body must contain {'document': 'Your long text here...'}",
                }
            ), 400

    except Exception as e:
        return jsonify({"error": f"Invalid JSON input: {e}"}), 400

    # 2. Pre-processing (Chunking)
    try:
        document_docs = process_document(document)
        chunks = [doc.page_content for doc in document_docs]

        if not chunks:
            return jsonify(
                {"error": "Document could not be processed into chunks."}
            ), 400

        logger.info(f"Document split into {len(chunks)} chunks.")

        # 3. Define Initial State for LangGraph
        initial_state = AgentState(
            input_text=document,
            document_chunks=chunks,
            summary_draft="",
            judge_result=None,
            refinement_count=0,
            # --- FIX: Use the user's input for max_refinement_steps ---
            max_refinement_steps=max_steps,
        )

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return jsonify({"error": f"Preprocessing failed: {e}"}), 500

    # 4. Agent Execution (Core Logic)
    try:
        # --- FIX: Pass the 'recursion_limit' to the invoke call ---
        # This ensures the graph stops after the specified number of steps
        config = {"recursion_limit": max_steps + 5}  # Add a buffer

        # We need to capture the full log. We'll manually stream and collect it.
        full_execution_log = []

        # Stream the execution to get intermediate steps
        for chunk in agent_graph.stream(initial_state, config=config):
            # 'chunk' will be a dictionary with the node name as the key
            node_name = list(chunk.keys())[0]
            node_result = list(chunk.values())[0]

            # Get the current state
            current_loop_count = node_result.get("refinement_count", 0)

            # Store the log entry
            full_execution_log.append(
                {
                    "node_name": node_name,
                    "loop_count": current_loop_count,
                    "result": node_result.get(
                        "judge_result"
                    ),  # Only 'judge' will have this
                }
            )

        # The final state is the last item in the log
        final_state = list(full_execution_log[-1].values())[0]

        # 5. Extract Final Result and Metadata
        final_summary = final_state.get("summary_draft", "Summary not found.")
        final_judge_result: JudgeResult = final_state.get("judge_result")

        critique_details = {}
        if final_judge_result:
            critique_details = {
                "score": final_judge_result.score,
                "critique": final_judge_result.critique,
                "refinement_needed": final_judge_result.should_refine,
            }
        else:
            critique_details = {
                "score": "N/A",  # <-- FIX: Corrected typo 'scpre'
                "critique": "Agent stopped before final critique (max steps likely reached).",
            }

        # 6. Return structured output to the frontend
        return (
            jsonify(
                {
                    "status": "success",
                    "final_summary": final_summary,
                    "refinement_steps_taken": final_state.get("refinement_count", 0),
                    "final_judge_result": critique_details,
                    "full_execution_log": full_execution_log,  # <-- FIX: Send the log to the frontend
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"LangGraph execution failed: {e}")
        return jsonify({"error": f"LangGraph execution failed: {e}"}), 500


# --- Server Start ---
if __name__ == "__main__":
    logger.info("Starting Flask application on http://127.0.0.1:5001...")
    app.run(host="0.0.0.0", port=5001, debug=True)
