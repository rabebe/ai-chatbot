"""
Utility functions for logging, filesystem paths, user identification,
Postgres caching with fuzzy matching, and atomic Redis quota management.
"""

import logging
import hashlib
import time
from typing import Optional, Tuple, List
from pathlib import Path
from difflib import SequenceMatcher
from fastapi import Request

from extensions import db
from models import Summary
from .redis_client import redis_client, HAS_REDIS

# --- Constants ---
QUOTA_TTL_SECONDS = 24 * 60 * 60  # 24 hours
CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days
MIN_CHARS = 100  # Minimum characters for input text to be cached


# ----------------------------------------------------
# LOGGING
# ----------------------------------------------------
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


# ----------------------------------------------------
# PROJECT PATH HELPERS
# ----------------------------------------------------
def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def ensure_logs_directory() -> Path:
    logs_dir = get_project_root() / "logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


# ----------------------------------------------------
# USER IDENTIFICATION
# ----------------------------------------------------
def get_user_id(request: Request) -> str:
    ip = request.client.host if request.client else "unknown_ip"
    ua = request.headers.get("user-agent", "unknown_ua")
    time_window = int(time.time()) // 86400
    raw = f"{ip}-{ua}-{time_window}"
    return hashlib.sha256(raw.encode()).hexdigest()


# ----------------------------------------------------
# FUZZY MATCHING (Postgres)
# ----------------------------------------------------
def fuzzy_match_summary(
    user_id: int, new_text: str, threshold: float = 0.85
) -> Optional[Tuple[str, str, int, str]]:
    """
    Search Postgres summary table for similar text for the user.
    Returns (input_text, output_text, score, critique_text) if similarity >= threshold.
    """
    summaries = Summary.query.filter_by(user_id=user_id).all()

    for row in summaries:
        similarity = SequenceMatcher(None, new_text, row.input_text).ratio()
        if similarity >= threshold:
            return row.input_text, row.output_text, row.score, row.critique_text
    return None


# ----------------------------------------------------
# SAVE SUMMARY
# ----------------------------------------------------
def save_summary(
    user_id: int, input_text: str, output_text: str, score: int, critique_text: str
) -> None:
    """
    Save or update a summary for the user in Postgres.
    """
    try:
        existing = Summary.query.filter_by(
            user_id=user_id, input_text=input_text
        ).first()
        if existing:
            existing.output_text = output_text
            existing.score = score
            existing.critique_text = critique_text
        else:
            new_summary = Summary(
                user_id=user_id,
                input_text=input_text,
                output_text=output_text,
                score=score,
                critique_text=critique_text,
            )
            db.session.add(new_summary)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error saving summary: {e}")


# ----------------------------------------------------
# USER HISTORY
# ----------------------------------------------------
def get_user_history(user_id: int, limit: int = 50) -> List[Tuple[str, str, str]]:
    """
    Returns recent summaries for a user (newest → oldest).
    """
    rows = (
        Summary.query.filter_by(user_id=user_id)
        .order_by(Summary.created_at.desc())
        .limit(limit)
        .all()
    )
    return [(r.input_text, r.output_text, r.created_at) for r in rows]


# ----------------------------------------------------
# REDIS-BASED USER QUOTA
# ----------------------------------------------------
def get_quota_key(user_id: str) -> str:
    return f"user_quota:{user_id}"


def check_remaining_quota(user_id: str, limit: int = 3) -> int:
    if not HAS_REDIS:
        return limit
    count = redis_client.get(get_quota_key(user_id))
    return max(0, limit - int(count)) if count else limit


def decrement_and_check_quota(user_id: str, limit: int = 3) -> bool:
    if not HAS_REDIS:
        return True

    key = get_quota_key(user_id)
    current_count = redis_client.incr(key)
    if current_count == 1:
        redis_client.expire(key, QUOTA_TTL_SECONDS)

    if current_count <= limit:
        logging.info(
            f"Quota granted for user {user_id}. Count: {current_count}/{limit}"
        )
        return True
    else:
        redis_client.decr(key)
        logging.warning(f"Quota denied for user {user_id}. Limit {limit} reached.")
        return False


def refund_user_quota(user_id: str) -> None:
    if not HAS_REDIS:
        return
    key = get_quota_key(user_id)
    new_count = redis_client.decr(key)
    if new_count < 0:
        redis_client.set(key, 0)
    logging.info(f"Quota refunded for user {user_id}. New Count: {max(0, new_count)}")


def check_user_quota(user_id: str, limit: int = 3) -> bool:
    logging.warning("Please use 'decrement_and_check_quota' for API calls.")
    return decrement_and_check_quota(user_id, limit)


def get_remaining_quota(user_id: str, limit: int = 3) -> int:
    return check_remaining_quota(user_id, limit)
