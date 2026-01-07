import logging
import json
from datetime import datetime
from flask import (
    Blueprint,
    request,
    jsonify,
    make_response,
    Response,
    stream_with_context,
)
from functools import wraps
import jwt
import hashlib

from models import User, Summary
from extensions import db
from src.core.agent_graph import agent_graph
from src.core.document_processor import process_document
from src.core.utils import (
    decrement_and_check_quota,
    refund_user_quota,
    get_remaining_quota,
    fuzzy_match_cache,
    MIN_CHARS,
    CACHE_TTL_SECONDS,
    save_to_cache,
)
from src.core.redis_client import redis_client, HAS_REDIS
from src.core.models import JudgeResult

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Blueprint for all API routes ---
routes = Blueprint(
    "routes", __name__, url_prefix="/api"
)  # ✅ prefix all API routes with /api
SECRET_KEY = "supersecret"


# -----------------------
# Authentication Decorator
# -----------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.cookies.get("access_token")
        if not token:
            return jsonify({"error": "Authentication required"}), 401
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            user_id = payload["user_id"]
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401
        return f(user_id=user_id, *args, **kwargs)

    return decorated_function


# -----------------------
# Utility functions
# -----------------------
def generate_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def now_iso():
    return datetime.utcnow().isoformat() + "Z"


# -----------------------
# Auth Routes
# -----------------------
@routes.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username, password = data.get("username"), data.get("password")
    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        token = jwt.encode({"user_id": user.id}, SECRET_KEY, algorithm="HS256")
        resp = make_response(jsonify({"message": "Login successful"}))
        resp.set_cookie("access_token", token, httponly=True)
        return resp
    return jsonify({"error": "Invalid credentials"}), 401


@routes.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    username, password, email = (
        data.get("username"),
        data.get("password"),
        data.get("email"),
    )
    if not username or not password or not email:
        return jsonify({"error": "Username, password, and email required"}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email already registered"}), 400

    new_user = User(username=username, email=email)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"message": "User registered successfully"}), 201


@routes.route("/logout", methods=["POST"])
def logout():
    resp = make_response(jsonify({"message": "Logged out successfully"}))
    resp.set_cookie("access_token", "", httponly=True, expires=0)
    return resp


# -----------------------
# User info & quota
# -----------------------
@routes.route("/me", methods=["GET"])
@login_required
def me(user_id):
    return jsonify({"user_id": user_id})


@routes.route("/me/quota", methods=["GET"])
@login_required
def get_my_quota(user_id):
    limit = 3
    remaining = get_remaining_quota(user_id, limit=limit)
    return jsonify({"count": limit - remaining, "remaining": remaining, "limit": limit})


# -----------------------
# Dashboard (history)
# -----------------------
@routes.route("/dashboard", methods=["GET"])
@login_required
def get_dashboard(user_id):
    try:
        history = (
            Summary.query.filter_by(user_id=user_id)
            .order_by(Summary.created_at.desc())
            .all()
        )
        data = [
            {
                "id": item.id,
                "input_text": item.input_text,
                "output_text": item.output_text,
                "created_at": item.created_at.isoformat() if item.created_at else None,
            }
            for item in history
        ]
        return jsonify(data), 200
    except Exception as e:
        logger.error(f"Dashboard fetch failed: {e}")
        return jsonify({"error": str(e)}), 500


# -----------------------
# Summarization (non-streaming)
# -----------------------
@routes.route("/summarize", methods=["POST"])
@login_required
def summarize_document(user_id):
    data = request.get_json()
    document = data.get("document")
    max_steps = data.get("max_refinement_steps", 3)

    if not document or len(document.strip()) < MIN_CHARS:
        return jsonify(
            {"error": f"Document too short. Minimum {MIN_CHARS} characters required."}
        ), 400

    document_content = document.strip()
    document_hash = generate_content_hash(document_content)

    # Redis / SQLite cache
    if HAS_REDIS:
        cached_summary_data = redis_client.get(document_hash)
        if cached_summary_data:
            cached_data = json.loads(cached_summary_data)
            return jsonify({"status": "cached", "evaluation": cached_data["judge"]})

    sqlite_match = fuzzy_match_cache(user_id, document_content)
    if sqlite_match:
        old_input, old_output, old_score, old_critique_text = sqlite_match
        return jsonify(
            {
                "status": "sqlite_fuzzy_cache",
                "final_summary": old_output,
                "refinement_steps_taken": 0,
                "final_judge_result": {
                    "score": old_score,
                    "critique_text": old_critique_text,
                    "refinement_needed": old_score < 7
                    if old_score is not None
                    else True,
                },
            }
        )

    if not decrement_and_check_quota(user_id, limit=3):
        remaining = get_remaining_quota(user_id, limit=3)
        return jsonify(
            {"error": "Daily summary limit reached", "remaining_quota": remaining}
        ), 429

    try:
        document_docs = process_document(document)
        chunks = [doc.page_content for doc in document_docs]

        initial_state = {
            "user_id": user_id,
            "input_text": document,
            "document_chunks": chunks,
            "summary_draft": "",
            "judge_result": None,
            "summary_history": [],
            "refinement_count": 0,
            "max_refinement_steps": max_steps,
        }

        final_state = agent_graph.invoke(initial_state)
        final_summary = final_state.get("summary_draft", "")
        refinement_count = final_state.get("refinement_count", 0)
        final_judge_result: JudgeResult | None = final_state.get("judge_result")

        # Fallback to summary_history if judge missing
        if not final_judge_result and final_state.get("summary_history"):
            last_summary = final_state["summary_history"][-1]
            final_judge_result = JudgeResult(
                score=last_summary.get("score"),
                critique=last_summary.get("critique"),
                should_refine=(last_summary.get("score", 0) < 7),
            )

        critique_details = {
            "score": final_judge_result.score if final_judge_result else None,
            "critique_text": final_judge_result.critique
            if final_judge_result
            else None,
            "refinement_needed": final_judge_result.should_refine
            if final_judge_result
            else True,
        }

        # Save to DB & cache
        db.session.add(
            Summary(input_text=document, output_text=final_summary, user_id=user_id)
        )
        db.session.commit()

        save_to_cache(
            user_id,
            document_content,
            final_summary,
            score=final_judge_result.score if final_judge_result else None,
            critique_text=final_judge_result.critique if final_judge_result else None,
        )

        if HAS_REDIS:
            cache_data = {
                "summary": final_summary,
                "steps": refinement_count,
                "judge": critique_details,
            }
            redis_client.set(
                document_hash, json.dumps(cache_data), ex=CACHE_TTL_SECONDS
            )

        return jsonify(
            {
                "status": "success",
                "final_summary": final_summary,
                "refinement_steps_taken": refinement_count,
                "final_judge_result": critique_details,
            }
        )

    except Exception as e:
        logger.error(f"LangGraph execution failed for user {user_id}: {e}")
        refund_user_quota(user_id)
        return jsonify({"error": f"LangGraph execution failed: {e}"}), 500


# ---------------------
# Streaming Summarization
# ---------------------
@routes.route("/summarize_stream", methods=["POST"])
@login_required
def summarize_document_stream(user_id):
    data = request.get_json(force=True)
    document = data.get("document")
    max_steps = data.get("max_refinement_steps", 3)

    if not document or len(document.strip()) < MIN_CHARS:
        return jsonify(
            {"error": f"Document too short. Minimum {MIN_CHARS} characters required."}
        ), 400

    document_content = document.strip()
    document_hash = generate_content_hash(document_content)

    # ---------------------
    # 1. Cached Response
    # ---------------------
    if HAS_REDIS and redis_client.get(document_hash):
        cached_data = json.loads(redis_client.get(document_hash))
        # Ensure key is 'critique_text'
        judge = cached_data.get("judge", {})
        final_judge_result = {
            "score": judge.get("score"),
            "critique_text": judge.get("critique_text"),
            "refinement_needed": judge.get("refinement_needed"),
        }
        return jsonify(
            {
                "status": "cached",
                "final_summary": cached_data.get("summary"),
                "refinement_steps_taken": cached_data.get("steps"),
                "final_judge_result": final_judge_result,
            }
        )

    elif not HAS_REDIS:
        sqlite_match = fuzzy_match_cache(user_id, document_content)
        if sqlite_match:
            old_input, old_output, old_score, old_critique_text = sqlite_match
            return jsonify(
                {
                    "status": "sqlite_fuzzy_cache",
                    "message": "85%+ Similarity Match found in history",
                    "final_summary": old_output,
                    "refinement_steps_taken": 0,
                    "final_judge_result": {
                        "score": old_score,
                        "critique_text": old_critique_text,
                        "refinement_needed": old_score < 7
                        if old_score is not None
                        else True,
                    },
                }
            )

    # ---------------------
    # 2. Quota Check
    # ---------------------
    if not decrement_and_check_quota(user_id, limit=3):
        remaining = get_remaining_quota(user_id, limit=3)
        return jsonify(
            {"error": "Daily summary limit reached", "remaining_quota": remaining}
        ), 429

    # ---------------------
    # 3. Process Document
    # ---------------------
    document_docs = process_document(document)
    chunks = [doc.page_content for doc in document_docs]
    initial_state = {
        "user_id": user_id,
        "input_text": document,
        "document_chunks": chunks,
        "summary_draft": "",
        "judge_result": None,
        "refinement_count": 0,
        "max_refinement_steps": max_steps,
    }

    def generator():
        final_summary = initial_state["summary_draft"]
        final_judge = None
        refinement_count = 0

        try:
            sent_initial = False
            for node_output in agent_graph.stream(
                initial_state, config={"recursion_limit": max_steps + 5}
            ):
                node_name = list(node_output.keys())[0]
                node_state = list(node_output.values())[0]
                summary_draft = node_state.get("summary_draft", "")
                judge_obj = node_state.get("judge_result")

                # Initial summary
                if node_name == "summarizer" and not sent_initial:
                    yield (
                        json.dumps(
                            {
                                "event": "initial_summary",
                                "summary": summary_draft,
                                "timestamp": now_iso(),
                            }
                        )
                        + "\n"
                    )
                    sent_initial = True
                    continue

                # Judge decision
                if node_name == "judge":
                    yield (
                        json.dumps(
                            {
                                "event": "judge_decision",
                                "score": getattr(judge_obj, "score", None)
                                if judge_obj
                                else None,
                                "critique_text": getattr(judge_obj, "critique", None)
                                if judge_obj
                                else None,
                                "refinement_needed": getattr(
                                    judge_obj, "should_refine", None
                                )
                                if judge_obj
                                else None,
                                "timestamp": now_iso(),
                            }
                        )
                        + "\n"
                    )
                    continue

                # Refinement step
                if node_name == "refine":
                    yield (
                        json.dumps(
                            {
                                "event": "refined_summary",
                                "summary": summary_draft,
                                "timestamp": now_iso(),
                            }
                        )
                        + "\n"
                    )
                    continue

            # ---------------------
            # Final summary
            # ---------------------
            final_summary = (
                node_state.get("summary_draft", "") if node_state else final_summary
            )
            final_judge = node_state.get("judge_result") if node_state else None
            refinement_count = (
                node_state.get("refinement_count", 0) if node_state else 0
            )

            final_evt = {
                "event": "final_summary",
                "summary": final_summary,
                "timestamp": now_iso(),
            }

            if final_judge:
                final_evt.update(
                    {
                        "score": getattr(final_judge, "score", None),
                        "critique_text": getattr(final_judge, "critique", None),
                        "refinement_needed": getattr(
                            final_judge, "should_refine", None
                        ),
                    }
                )

            yield json.dumps(final_evt) + "\n"

            # ---------------------
            # Save DB & Cache
            # ---------------------
            db.session.add(
                Summary(input_text=document, output_text=final_summary, user_id=user_id)
            )
            db.session.commit()

            # SQLite cache
            save_to_cache(
                user_id,
                document_content,
                final_summary,
                score=getattr(final_judge, "score", None),
                critique_text=getattr(final_judge, "critique", None),
            )

            # Redis cache
            if HAS_REDIS:
                redis_client.set(
                    document_hash,
                    json.dumps(
                        {
                            "summary": final_summary,
                            "steps": refinement_count,
                            "judge": {
                                "score": getattr(final_judge, "score", None),
                                "critique_text": getattr(final_judge, "critique", None),
                                "refinement_needed": getattr(
                                    final_judge, "should_refine", None
                                ),
                            },
                        }
                    ),
                    ex=CACHE_TTL_SECONDS,
                )

        except Exception as exc:
            logger.exception(f"Streaming summarization failed for user {user_id}.")
            yield (
                json.dumps(
                    {"event": "error", "message": str(exc), "timestamp": now_iso()}
                )
                + "\n"
            )
            refund_user_quota(user_id)

    return Response(stream_with_context(generator()), mimetype="application/x-ndjson")
