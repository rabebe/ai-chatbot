import pytest
import time
import os

os.environ["REDIS_URL"] = "redis://localhost:6379/1"

# Import the functions and variables we want to test
from src.core.utils import (
    decrement_and_check_quota,
    refund_user_quota,
    get_remaining_quota,
    get_quota_key,
    QUOTA_TTL_SECONDS,
)
from src.core.redis_client import redis_client, HAS_REDIS

# --- Test Setup/Fixtures ---

TEST_USER_ID = "test_user_quota_12345"
QUOTA_LIMIT = 3


@pytest.fixture(autouse=True)
def cleanup_redis_key():
    """Fixture to ensure the test user's quota key is clean before and after each test."""
    key = get_quota_key(TEST_USER_ID)

    # Setup: Delete the key before running the test
    if HAS_REDIS:
        redis_client.delete(key)

    yield  # Run the test

    # Teardown: Delete the key after the test finishes
    if HAS_REDIS:
        redis_client.delete(key)


# Skip all tests if Redis is not available
if not HAS_REDIS:
    pytest.skip(
        "Skipping Redis quota tests because HAS_REDIS is False.",
        allow_module_level=True,
    )


# -----------------------------------------------------------------
# 1. Test decrement_and_check_quota (Atomic Check and Reserve)
# -----------------------------------------------------------------


def test_01_quota_initial_grant_and_expiry_set():
    """Tests the first call grants access and sets the TTL."""
    key = get_quota_key(TEST_USER_ID)

    # 1. First call should succeed
    assert decrement_and_check_quota(TEST_USER_ID, QUOTA_LIMIT) is True

    # 2. Verify the counter is 1
    assert int(redis_client.get(key)) == 1

    # 3. Verify the TTL is set (Redis returns time remaining in seconds)
    ttl = redis_client.ttl(key)
    # The TTL should be close to 24 hours (QUOTA_TTL_SECONDS)
    assert ttl > QUOTA_TTL_SECONDS - 5 and ttl <= QUOTA_TTL_SECONDS

    # 4. Verify remaining quota is correct
    assert get_remaining_quota(TEST_USER_ID, QUOTA_LIMIT) == QUOTA_LIMIT - 1


def test_02_quota_success_up_to_limit():
    """Tests subsequent calls succeed without resetting TTL."""
    key = get_quota_key(TEST_USER_ID)

    # First call: set counter to 1 and TTL
    decrement_and_check_quota(TEST_USER_ID, QUOTA_LIMIT)
    initial_ttl = redis_client.ttl(key)

    # Wait for 1 second to guarantee time has elapsed.
    time.sleep(1)

    # Second and third calls should succeed
    assert decrement_and_check_quota(TEST_USER_ID, QUOTA_LIMIT) is True
    assert decrement_and_check_quota(TEST_USER_ID, QUOTA_LIMIT) is True

    # Counter should be at the limit (3)
    assert int(redis_client.get(key)) == QUOTA_LIMIT

    # Remaining quota should be 0
    assert get_remaining_quota(TEST_USER_ID, QUOTA_LIMIT) == 0

    # TTL should not have been reset significantly
    assert redis_client.ttl(key) < initial_ttl


def test_03_quota_denied_past_limit():
    """Tests that the quota is denied and the counter is refunded."""
    key = get_quota_key(TEST_USER_ID)

    # Grant access up to the limit (3 times)
    for _ in range(QUOTA_LIMIT):
        decrement_and_check_quota(TEST_USER_ID, QUOTA_LIMIT)

    # Fourth call (past limit) should fail
    assert decrement_and_check_quota(TEST_USER_ID, QUOTA_LIMIT) is False

    # CRITICAL CHECK: Counter should have been incremented (to 4) and then refunded (back to 3)
    assert int(redis_client.get(key)) == QUOTA_LIMIT
    assert get_remaining_quota(TEST_USER_ID, QUOTA_LIMIT) == 0


# -----------------------------------------------------------------
# 2. Test refund_user_quota (Atomic Refund)
# -----------------------------------------------------------------


def test_04_refund_on_success():
    """Tests refunding a successful reservation."""
    key = get_quota_key(TEST_USER_ID)

    # Grant quota (Count is 1)
    decrement_and_check_quota(TEST_USER_ID, QUOTA_LIMIT)
    assert int(redis_client.get(key)) == 1

    # Refund quota (Simulating AI job failure)
    refund_user_quota(TEST_USER_ID)

    # Counter should be back to 0
    assert int(redis_client.get(key)) == 0
    assert get_remaining_quota(TEST_USER_ID, QUOTA_LIMIT) == QUOTA_LIMIT


def test_05_refund_does_not_go_below_zero():
    """Tests the safety check to prevent counter from going below zero."""
    key = get_quota_key(TEST_USER_ID)

    # Ensure key exists and is 0
    redis_client.set(key, 0)

    # Try to refund (should fail silently or revert the count)
    refund_user_quota(TEST_USER_ID)

    # Counter should remain at 0 (or be set to 0 by the safety check)
    assert int(redis_client.get(key)) == 0
    assert get_remaining_quota(TEST_USER_ID, QUOTA_LIMIT) == QUOTA_LIMIT
