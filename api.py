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
static_path = os.path.join(script_dir, "static")

# Initialize Flask app
app = Flask(
    __name__,
    template_folder=template_path,
    static_folder=static_path,
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


@app.route("/<path:path>", methods=["GET"])
def serve_app_routes(path):
    """
    Catch-all route to serve the SPA's main HTML file for any client-side route.
    """
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize_document():
    """
    Accepts a document string, runs it through the self-correcting LangGraph agent,
    and returns the final summary and process details, including the execution log.
    """
    logger.info("Received request for summarization.")

    # 1. Input Validation
    try:
        data = request.get_json()
        document = data.get("document")

        # Get max_refinement_steps from the user's request (default to 3)
        max_steps = data.get("max_refinement_steps", 3)

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
            max_refinement_steps=max_steps,  # Use the user's input
        )

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return jsonify({"error": f"Preprocessing failed: {e}"}), 500

    # 4. Agent Execution (Core Logic)
    try:
        # Pass the 'recursion_limit' to the invoke call
        config = {"recursion_limit": max_steps + 5}  # Add a buffer for safety

        full_execution_log = []

        # Stream the execution to get intermediate steps and collect them
        for chunk in agent_graph.stream(initial_state, config=config):
            # 'chunk' will be a dictionary with the node name as the key
            node_name = list(chunk.keys())[0]
            node_result = list(chunk.values())[0]

            # The last chunk's value is the final state update
            final_state = node_result

            # Get the current loop count
            current_loop_count = node_result.get("refinement_count", 0)

            # --- FIX: Serialize the JudgeResult object for the log entry ---
            judge_result_obj: JudgeResult = node_result.get("judge_result")

            # Convert the custom Pydantic object (JudgeResult) to a serializable dictionary
            serialized_result = None
            if judge_result_obj:
                serialized_result = {
                    "score": judge_result_obj.score,
                    "critique": judge_result_obj.critique,
                    "refinement_needed": judge_result_obj.should_refine,
                }
            # ----------------------------------------------------------------

            # Store the log entry, using the serialized result
            full_execution_log.append(
                {
                    "node_name": node_name,
                    "loop_count": current_loop_count,
                    "result": serialized_result,
                }
            )

        # 5. Extract Final Result and Metadata

        # --- DEBUG LOG ADDED HERE ---
        # Log the keys of the final state to see what the graph returned
        logger.info(
            f"DEBUG: Final state keys returned by LangGraph stream: {list(final_state.keys()) if final_state and isinstance(final_state, dict) else 'Not a dictionary or None'}"
        )
        # ---------------------------

        final_summary = final_state.get("summary_draft", "Summary not found.")
        final_judge_result: JudgeResult = final_state.get("judge_result")

        # The final result is manually serialized here as well, which is correct.
        critique_details = {}
        if final_judge_result:
            critique_details = {
                "score": final_judge_result.score,
                "critique": final_judge_result.critique,
                "refinement_needed": final_judge_result.should_refine,
            }
        else:
            # Handle case where the graph finished without hitting the judge node
            critique_details = {
                "score": "N/A",
                "critique": "Agent stopped before final critique (max steps likely reached).",
                "refinement_needed": True,
            }

        # 6. Return structured output to the frontend
        return (
            jsonify(
                {
                    "status": "success",
                    "final_summary": final_summary,
                    "refinement_steps_taken": final_state.get("refinement_count", 0),
                    "final_judge_result": critique_details,
                    "full_execution_log": full_execution_log,  # Send the log to the frontend
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
