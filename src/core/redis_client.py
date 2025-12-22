import os
import redis
import logging

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


try:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    # Test the connection
    redis_client.ping()
    HAS_REDIS = True
    logging.info("Connected to Redis successfully.")
except (redis.ConnectionError, redis.TimeoutError) as e:
    redis_client = None
    HAS_REDIS = False
    logging.warning(f"Redis not available, falling back to SQLite: {e}")
