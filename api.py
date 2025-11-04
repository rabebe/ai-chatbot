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


# Initialize Flask app
app = Flask(__name__)
# NOTE: Using CORS for development purposes to allow the frontend (index.html)
# to talk to this API from a different port/origin.

# --- TEMPLATE PATH FIX ---
# Calculate the absolute path to the templates folder and set it explicitly
script_dir = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(script_dir, "templates")
app.template_folder = template_path
logger.info(f"Flask template folder manually set to: {app.template_folder}")

CORS(app)
logger.info("Flask-CORS initialized, allowing all origins (*).")

# --- API Endpoint ---


@app.route("/", methods=["GET"])
@app.route("/index.html", methods=["GET"])
def serve_index():
    """Returns the main index.html file from the 'template' folder."""
    # This check runs every time the route is hit and will confirm the file's presence.
    full_path_check = os.path.join(app.template_folder, "index.html")
    if not os.path.exists(full_path_check):
        logger.error(f"Template NOT FOUND at expected path: {full_path_check}")
        # Raising an error here so the user sees the path in the traceback immediately
        raise Exception(
            f"Configuration Error: 'index.html' not found. Expected path: {full_path_check}"
        )
    else:
        logger.info(f"Template check SUCCESS: 'index.html' found at: {full_path_check}")
    # --- END CRITICAL DIAGNOSTIC CHECK ---

    # This is the most reliable way to serve the HTML using Flask's templating engine
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
        # Use the utility function to process the document into chunks
        document_docs = process_document(document)

        # Convert LangChain Document objects to a simple list of strings for the state
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
            max_refinement_steps=3,  # Loop limit to prevent infinite loops
        )

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return jsonify({"error": f"Preprocessing failed: {e}"}), 500

    # 4. Agent Execution (Core Logic)
    try:
        # Run the compiled LangGraph agent synchronously
        final_state = agent_graph.invoke(initial_state)

        # 5. Extract Final Result and Metadata
        final_summary = final_state.get("summary_draft", "Summary not found.")
        final_judge_result: JudgeResult = final_state.get("judge_result")

        critique_details = {}
        if final_judge_result:
            critique_details = {
                "score": final_judge_result.score,
                "critique": final_judge_result.critique,
                # This should always be False if the process ended successfully
                "refinement_needed": final_judge_result.should_refine,
            }
        else:
            critique_details = {
                "scpre": "N/A",
                "critique": "Agent stopped before final critique",
            }

        # 6. Return structured output to the frontend
        return jsonify(
            {
                "status": "success",
                "final_summary": final_summary,
                "refinement_steps_taken": final_state.get("refinement_count", 0),
                "final_judge_result": critique_details,
            }
        ), 200

    except Exception as e:
        # Catch any errors from the LangGraph execution itself
        logger.error(f"LangGraph execution failed: {e}")
        return jsonify({"error": f"LangGraph execution failed: {e}"}), 500


# --- Server Start ---
if __name__ == "__main__":
    logger.info("Starting Flask application on http://127.0.0.1:5001...")
    # Using 0.0.0.0 for broader access if run in a container/VM
    app.run(host="0.0.0.0", port=5001, debug=True)
