import os
import redis
import logging
from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv("ENV", "development")

redis_client = None
HAS_REDIS = False

if ENV == "development":
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        HAS_REDIS = True
        logging.info(f"[DEV] Connected to local Redis at {REDIS_URL}")
    except (redis.ConnectionError, redis.TimeoutError) as e:
        logging.warning(f"[DEV] Redis not available: {e}")
        redis_client = None
else:
    # Production: Upstash
    UPSTASH_URL = os.getenv("UPSTASH_URL")
    try:
        redis_client = redis.Redis.from_url(UPSTASH_URL, decode_responses=True)
        redis_client.ping()
        HAS_REDIS = True
        logging.info(f"[PROD] Connected to Upstash Redis at {UPSTASH_URL}")
    except (redis.ConnectionError, redis.TimeoutError) as e:
        logging.warning(f"[PROD] Upstash Redis not available: {e}")
        redis_client = None
