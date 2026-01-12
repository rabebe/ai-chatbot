from dotenv import load_dotenv
import os
import logging
import json
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from functools import wraps

import jwt
from flask import (
    Blueprint,
    request,
    jsonify,
    make_response,
    Response,
    stream_with_context,
)

from models import User, Summary
from extensions import db

from src.core.agent_graph import agent_graph
from src.core.document_processor import process_document
from src.core.models import JudgeResult
from src.core.email_service import send_verification_email
from src.core.redis_client import redis_client, HAS_REDIS
from src.core.utils import (
    decrement_and_check_quota,
    refund_user_quota,
    get_remaining_quota,
    fuzzy_match_summary,
    MIN_CHARS,
    CACHE_TTL_SECONDS,
    save_summary,
)

load_dotenv()

FLASK_ENV = os.getenv("FLASK_ENV", "production")
secure_cookie = False if FLASK_ENV == "development" else True

# -----------------------
# Logging
# -----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------
# Blueprint
# -----------------------
routes = Blueprint("routes", __name__, url_prefix="/api")
SECRET_KEY = os.getenv("SECRET_KEY")


# -----------------------
# Authentication Decorator
# -----------------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get("access_token")
        if not token:
            return jsonify({"error": "Authentication required"}), 401
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

        return f(user_id=payload["user_id"], *args, **kwargs)

    return decorated


# -----------------------
# Utilities
# -----------------------
def generate_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat() + "Z"


# -----------------------
# Auth Routes
# -----------------------
@routes.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data.get("username")).first()

    if not user or not user.check_password(data.get("password")):
        return jsonify({"error": "Invalid credentials"}), 401

    if not user.is_verified:
        return jsonify({"error": "Email not verified. Please check your inbox."}), 403

    token = jwt.encode({"user_id": user.id}, SECRET_KEY, algorithm="HS256")
    resp = make_response(jsonify({"message": "Login successful"}))
    resp.set_cookie(
        "access_token", token, httponly=True, samesite="None", secure=secure_cookie
    )
    return resp


@routes.route("/register", methods=["POST"])
def register():
    data = request.get_json()

    if User.query.filter(
        (User.username == data["username"]) | (User.email == data["email"])
    ).first():
        return jsonify({"error": "Username exists"}), 400

    user = User(
        username=data["username"],
        email=data["email"],
        is_verified=False,
        daily_summary_count=0,
        last_summary_date=None,
    )

    user.set_password(data["password"])

    token = secrets.token_urlsafe(32)
    user.verification_token = token
    user.token_expiry = datetime.now(timezone.utc) + timedelta(hours=24)

    db.session.add(user)
    db.session.commit()

    # Send verification email
    try:
        send_verification_email(user.email, token)
    except Exception:
        # Log error but don't block registration
        logger.exception("Failed to send verification email")
        return jsonify(
            {"message": "User registered, but verification email failed to send"}
        ), 201

    return jsonify({"message": "User registered", "verification_token": token}), 201


@routes.route("/verify", methods=["GET"])
def verify_email():
    token = request.args.get("token")

    if not token:
        return jsonify({"error": "Missing token"}), 400

    # Find user by token
    user = User.query.filter_by(verification_token=token).first()
    if not user:
        return jsonify({"error": "Invalid token"}), 400

    # Ensure token_expiry is timezone-aware
    expiry = user.token_expiry
    if expiry is None:
        return jsonify(
            {"error": "Token expired. Please request a new verification email."}
        ), 400
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)

    if expiry < datetime.now(timezone.utc):
        return jsonify(
            {"error": "Token expired. Please request a new verification email."}
        ), 400

    # Mark user as verified
    user.is_verified = True
    user.verification_token = None
    user.token_expiry = None
    db.session.commit()

    return jsonify({"message": "Email verified successfully"}), 200


@routes.route("/resend-verification", methods=["POST"])
def resend_verification():
    data = request.get_json()
    email = data.get("email")

    if not email:
        return jsonify({"error": "Email required"}), 400

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"error": "Email not found"}), 404

    if user.is_verified:
        return jsonify({"message": "Email already verified"}), 200

    # Generate new token
    token = secrets.token_urlsafe(32)
    user.verification_token = token
    user.token_expiry = datetime.now(timezone.utc) + timedelta(hours=24)
    db.session.commit()

    try:
        send_verification_email(user.email, token)
    except Exception as e:
        return jsonify({"error": f"Failed to send verification email: {str(e)}"}), 500

    return jsonify({"message": "Verification email resent"}), 200


@routes.route("/logout", methods=["POST"])
def logout():
    resp = make_response(jsonify({"message": "Logged out"}))
    resp.set_cookie(
        "access_token",
        "",
        expires=0,
        httponly=True,
        samesite="None",
        secure=secure_cookie,
    )
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
def quota(user_id):
    limit = 3
    remaining = get_remaining_quota(user_id, limit)
    return jsonify({"remaining": remaining, "limit": limit})


# -----------------------
# Dashboard
# -----------------------
@routes.route("/dashboard", methods=["GET"])
@login_required
def dashboard(user_id):
    summaries = (
        Summary.query.filter_by(user_id=user_id)
        .order_by(Summary.created_at.desc())
        .all()
    )

    return jsonify(
        [
            {
                "id": s.id,
                "input_text": s.input_text,
                "output_text": s.output_text,
                "score": s.score,
                "critique_text": s.critique_text,
                "created_at": s.created_at.isoformat(),
            }
            for s in summaries
        ]
    )


# -----------------------
# Summarize (non-stream)
# -----------------------
@routes.route("/summarize", methods=["POST"])
@login_required
def summarize(user_id):
    data = request.get_json()
    document = data.get("document", "").strip()
    max_steps = data.get("max_refinement_steps", 3)

    if len(document) < MIN_CHARS:
        return jsonify({"error": "Document too short"}), 400

    doc_hash = generate_content_hash(document)

    # Redis cache
    if HAS_REDIS:
        cached = redis_client.get(doc_hash)
        if cached:
            return jsonify(json.loads(cached))

    # DB fuzzy cache
    match = fuzzy_match_summary(user_id, document)
    if match:
        _, output, score, critique = match
        return jsonify(
            {
                "status": "cached",
                "final_summary": output,
                "final_judge_result": {
                    "score": score,
                    "critique_text": critique,
                    "refinement_needed": score < 7 if score else True,
                },
            }
        )

    if not decrement_and_check_quota(user_id, limit=3):
        return jsonify({"error": "Quota exceeded"}), 429

    try:
        docs = process_document(document)
        state = {
            "user_id": user_id,
            "input_text": document,
            "document_chunks": [d.page_content for d in docs],
            "summary_draft": "",
            "refinement_count": 0,
            "max_refinement_steps": max_steps,
        }

        final = agent_graph.invoke(state)
        summary = final["summary_draft"]
        judge: JudgeResult = final.get("judge_result")

        save_summary(
            user_id=user_id,
            input_text=document,
            output_text=summary,
            score=judge.score if judge else None,
            critique_text=judge.critique if judge else None,
        )

        response = {
            "status": "success",
            "final_summary": summary,
            "final_judge_result": {
                "score": judge.score if judge else None,
                "critique_text": judge.critique if judge else None,
                "refinement_needed": judge.should_refine if judge else True,
            },
        }

        if HAS_REDIS:
            redis_client.set(doc_hash, json.dumps(response), ex=CACHE_TTL_SECONDS)

        return jsonify(response)

    except Exception as e:
        refund_user_quota(user_id)
        logger.exception("Summarization failed")
        return jsonify({"error": str(e)}), 500


# -----------------------
# Streaming Summarization
# -----------------------
@routes.route("/summarize_stream", methods=["POST"])
@login_required
def summarize_stream(user_id):
    data = request.get_json(force=True)
    document = data.get("document", "").strip()
    max_steps = data.get("max_refinement_steps", 3)

    if len(document) < MIN_CHARS:
        return jsonify({"error": "Document too short"}), 400

    doc_hash = generate_content_hash(document)

    # -------------------
    # Check Redis cache
    # -------------------
    if HAS_REDIS:
        cached = redis_client.get(doc_hash)
        if cached:
            cached_data = json.loads(cached)
            summary = cached_data.get("final_summary", "")
            judge = cached_data.get("final_judge_result", {}) or {}
            critique = judge.get("critique_text", "")
            score = judge.get("score", 0)

            if summary:

                def cached_stream():
                    logger.info("Returning cached summary from Redis")
                    yield (
                        json.dumps(
                            {
                                "event": "final_summary",
                                "summary": summary,
                                "critique": critique,
                                "score": score,
                            }
                        )
                        + "\n"
                    )

                return Response(
                    stream_with_context(cached_stream()),
                    mimetype="application/x-ndjson",
                )

    # -------------------
    # Check fuzzy DB cache
    # -------------------
    match = fuzzy_match_summary(user_id, document)
    if match:
        _, output, score, critique = match
        output = output or ""
        score = score if isinstance(score, int) else 0
        critique = critique or ""

        if output:

            def fuzzy_stream():
                logger.info("Returning cached summary from fuzzy DB")
                yield (
                    json.dumps(
                        {
                            "event": "final_summary",
                            "summary": output,
                            "critique": critique,
                            "score": score,
                        }
                    )
                    + "\n"
                )

            return Response(
                stream_with_context(fuzzy_stream()), mimetype="application/x-ndjson"
            )

    # -------------------
    # QUOTA CHECK for new text
    # -------------------
    if not decrement_and_check_quota(user_id, limit=3):
        return jsonify({"error": "Quota exceeded"}), 429

    # -------------------
    # Process document and stream AI updates
    # -------------------
    docs = process_document(document)
    state = {
        "user_id": user_id,
        "document_chunks": [d.page_content for d in docs],
        "summary_draft": "",
        "refinement_count": 0,
        "max_refinement_steps": max_steps,
    }

    def extract_summary(update):
        """Extract summary draft from known keys/nodes."""
        if not isinstance(update, dict):
            return None
        # Top-level summary
        if "summary_draft" in update and update["summary_draft"]:
            return update["summary_draft"]
        # Summarizer node
        if "summarizer" in update and "summary_draft" in update["summarizer"]:
            return update["summarizer"]["summary_draft"]
        # Judge node
        if "judge" in update and "summary_draft" in update["judge"]:
            return update["judge"]["summary_draft"]
        return None

    def stream():
        final_summary = None
        final_score = 0
        final_critique = ""

        try:
            for update in agent_graph.stream(state):
                logger.info(f"Agent update: {update}")

                # Convert Pydantic model to dict if needed
                from src.core.models import JudgeResult

                if isinstance(update, JudgeResult):
                    update = update.model_dump()

                # -----------------
                # Draft summaries
                # -----------------
                draft = extract_summary(update)
                if isinstance(draft, str) and draft.strip():
                    final_summary = draft.strip()
                    logger.info("Streaming draft summary")
                    yield (
                        json.dumps(
                            {"event": "refined_summary", "summary": final_summary}
                        )
                        + "\n"
                    )

                # -----------------
                # Judge events
                # -----------------
                judge = update.get("judge_result") or update.get("judge", {}).get(
                    "judge_result"
                )
                if judge:
                    score = judge.get("score")
                    critique = judge.get("critique")

                    if isinstance(score, int):
                        final_score = score
                    if critique:
                        final_critique = critique

                    logger.info(
                        f"Streaming judge decision: score={final_score}, critique={final_critique}"
                    )

                    yield (
                        json.dumps(
                            {
                                "event": "judge_decision",
                                "score": final_score,
                                "critique": final_critique,
                            }
                        )
                        + "\n"
                    )

            # -----------------
            # Final validation
            # -----------------
            if not final_summary:
                raise RuntimeError("No summary was generated by the agent graph")

            logger.info("Streaming final summary")
            yield (
                json.dumps(
                    {
                        "event": "final_summary",
                        "summary": final_summary,
                        "score": final_score,
                        "critique": final_critique,
                    }
                )
                + "\n"
            )

            # -----------------
            # Save final summary
            # -----------------
            save_summary(
                user_id=user_id,
                input_text=document,
                output_text=final_summary,
                score=final_score,
                critique_text=final_critique,
            )

            # -----------------
            # Cache in Redis
            # -----------------
            if HAS_REDIS and final_summary:
                redis_client.set(
                    doc_hash,
                    json.dumps(
                        {
                            "final_summary": final_summary,
                            "final_judge_result": {
                                "score": final_score,
                                "critique_text": final_critique,
                                "output_text": final_summary,
                            },
                        }
                    ),
                    ex=CACHE_TTL_SECONDS,
                )

        except Exception as e:
            refund_user_quota(user_id)
            logger.exception("Streaming failed")
            yield json.dumps({"event": "error", "message": str(e)}) + "\n"

    return Response(stream_with_context(stream()), mimetype="application/x-ndjson")
