"""
Utility functions for logging, filesystem paths, user identification,
and SQLite caching with fuzzy matching, and atomic Redis quota management.
"""

import logging
from typing import Optional
from pathlib import Path

# Import Redis client setup from its dedicated file

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
