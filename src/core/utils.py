"""
Utility functions for logging, filesystem paths, user identification,
and SQLite caching with fuzzy matching, and atomic Redis quota management.
"""

import logging
import sqlite3
import hashlib
from typing import Optional, Tuple, List
from pathlib import Path
from difflib import SequenceMatcher
from fastapi import Request

# Import Redis client setup from its dedicated file
from .redis_client import redis_client, HAS_REDIS

# --- Constants ---
# Time-To-Live for the quota key (24 hours in seconds)
QUOTA_TTL_SECONDS = 24 * 60 * 60

# Time-To-Live for cached summaries (7 days in seconds)
# This controls how long a summary remains valid in Redis before the system regenerates it.
CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # 604,800 seconds for 7 days

MIN_CHARS = 100  # Minimum characters for input text to be cached


# ----------------------------------------------------
# LOGGING
# ----------------------------------------------------
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    # Create logs directory if it doesn't exist
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
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
    """
    Get the project root directory.

    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent.parent


def ensure_data_directory() -> Path:
    """
    Ensure the data directory exists.

    Returns:
        Path to data directory
    """
    data_dir = get_project_root() / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def ensure_logs_directory() -> Path:
    """
    Ensure the logs directory exists.

    Returns:
        Path to logs directory
    """
    logs_dir = get_project_root() / "logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


# ----------------------------------------------------
# USER IDENTIFICATION
# ----------------------------------------------------
def get_user_id(request: Request) -> str:
    """
    Returns a stable, anonymous user ID.
    Uses IP + user-agent hashed to protect privacy.
    """
    ip = request.client.host if request.client else "unknown_ip"
    ua = request.headers.get("user-agent", "unknown_ua")
    raw = f"{ip}-{ua}"

    return hashlib.sha256(raw.encode()).hexdigest()


# ----------------------------------------------------
# SQLITE CACHE INITIALIZATION
# ----------------------------------------------------
cache_conn = None


def init_cache_db():
    global cache_conn

    db_path = ensure_data_directory() / "summary_cache.db"
    cache_conn = sqlite3.connect(db_path, check_same_thread=False)

    cursor = cache_conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS summary_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            input_text TEXT NOT NULL,
            output_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, input_text)
        )
        """
    )
    cache_conn.commit()


# ----------------------------------------------------
# FUZZY MATCHING
# ----------------------------------------------------
def fuzzy_match_cache(
    user_id: str, new_text: str, threshold: float = 0.85
) -> Optional[Tuple[str, str]]:
    """
    Searches cached inputs for similar text.
    Returns (input_text, output_text) if similarity above threshold.
    """
    if not cache_conn:
        # Ensure connection is initialized before use
        init_cache_db()

    cursor = cache_conn.cursor()
    cursor.execute(
        "SELECT input_text, output_text FROM summary_cache WHERE user_id = ?",
        (user_id,),
    )

    all_rows = cursor.fetchall()

    for old_input, old_output in all_rows:
        similarity = SequenceMatcher(None, new_text, old_input).ratio()
        if similarity >= threshold:
            return old_input, old_output

    return None


# ----------------------------------------------------
# CACHE SAVE
# ----------------------------------------------------
def save_to_cache(user_id: str, input_text: str, output_text: str) -> None:
    if not cache_conn:
        init_cache_db()

    cursor = cache_conn.cursor()
    try:
        cursor.execute(
            """
            INSERT OR REPLACE INTO summary_cache (user_id, input_text, output_text)
            VALUES (?, ?, ?)
            """,
            (user_id, input_text, output_text),
        )
        cache_conn.commit()
    except Exception as e:
        logging.error(f"Error saving to cache: {e}")


# ----------------------------------------------------
# USER HISTORY
# ----------------------------------------------------
def get_user_history(user_id: str, limit: int = 50) -> List[Tuple[str, str, str]]:
    """
    Returns recent summaries:
        input_text, output_text, created_at
    Ordered newest → oldest.
    """
    if not cache_conn:
        init_cache_db()

    cursor = cache_conn.cursor()
    cursor.execute(
        """
        SELECT input_text, output_text, created_at
        FROM summary_cache
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    )

    return cursor.fetchall()


# ----------------------------------------------------
# REDIS-BASED USER QUOTA (Atomic Implementation)
# ----------------------------------------------------


def get_quota_key(user_id: str) -> str:
    """Standardizes the Redis key format for user quota."""
    return f"user_quota:{user_id}"


def check_remaining_quota(user_id: str, limit: int = 3) -> int:
    """Calculates the remaining quota without affecting the counter."""
    if not HAS_REDIS:
        # If Redis is unavailable, assume full quota availability for this check
        return limit

    key = get_quota_key(user_id)
    count = redis_client.get(key)

    if count is None:
        return limit

    return max(0, limit - int(count))


# Note: This function replaces the original `check_user_quota` for API access logic.
def decrement_and_check_quota(user_id: str, limit: int = 3) -> bool:
    """
    Atomically decrements (increments the count) and checks the limit.
    Returns True if the user is granted access, False otherwise.
    """
    if not HAS_REDIS:
        # Fallback: Assume success if Redis is down (fail-open)
        return True

    key = get_quota_key(user_id)

    # 1. ATOMICALLY INCREMENT the counter
    current_count = redis_client.incr(key)

    # 2. Set/Reset Expiration
    # Only set TTL if the key was just created (count == 1)
    if current_count == 1:
        redis_client.expire(key, QUOTA_TTL_SECONDS)

    # 3. Check limit
    if current_count <= limit:
        logging.info(
            f"Quota granted for user {user_id}. Count: {current_count}/{limit}"
        )
        return True
    else:
        # If the count exceeds the limit, immediately refund the increment
        redis_client.decr(key)
        logging.warning(f"Quota denied for user {user_id}. Limit {limit} reached.")
        return False


def refund_user_quota(user_id: str) -> None:
    """
    Refunds a single quota unit for a user.
    Used when an expensive AI job fails after the quota was successfully reserved.
    """
    if not HAS_REDIS:
        return

    key = get_quota_key(user_id)

    # Atomically decrement the counter
    new_count = redis_client.decr(key)

    # Safety check: Prevent the counter from going below zero
    if new_count < 0:
        # If it goes below zero, immediately set it back to zero
        redis_client.set(key, 0)

    logging.info(f"Quota refunded for user {user_id}. New Count: {max(0, new_count)}")


# Retain original function names for compatibility, directing them to the new logic
def check_user_quota(user_id: str, limit: int = 3) -> bool:
    """
    DEPRECATED/WRAPPER: Use decrement_and_check_quota in routes.py instead.
    This wrapper prevents accidental use of non-atomic check logic.
    """
    logging.warning("Please use 'decrement_and_check_quota' for API calls.")
    return decrement_and_check_quota(user_id, limit)


def get_remaining_quota(user_id: str, limit: int = 3) -> int:
    """
    Returns the number of remaining summaries for the user today.
    """
    return check_remaining_quota(user_id, limit)
