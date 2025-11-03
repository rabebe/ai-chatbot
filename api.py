import logging
from flask import Flask, request, jsonify
from dotenv import load_dotenv

try:
    from flask_cors import CORS

    HAS_CORS = True
except ImportError:
    HAS_CORS = False

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
if HAS_CORS:
    CORS(app)
    logger.info("Flask-CORS enabled.")
else:
    logger.warning(
        "Flask-CORS not installed. You may encounter CORS issues if running frontend locally."
    )

# --- API Endpoint ---


@app.route("/", methods=["GET"])
def root_status():
    """Returns a simple status message for the root path."""
    return jsonify(
        {
            "status": "LangGraph Summarization Service is running",
            "message": "Use a POST request to the /summarize endpoint to process documents.",
        }
    ), 200


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
                "status": "Process completed without final judge result."
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
