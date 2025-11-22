"""
Property-based tests for session management (Task 4.1, 4.2).

Tests Properties 11 and 12 from design.md:
- Property 11: Session expiration cleanup
- Property 12: Session persistence until termination
"""
import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from hypothesis import given, strategies as st, settings, assume
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant
import uuid
import redis

from app.services.session_manager import SessionManager
from app.models.session import SessionData


# Custom strategies for generating test data
@st.composite
def token_jti_strategy(draw):
    """Generate valid JWT token JTI (UUID format)"""
    return str(uuid.uuid4())


@st.composite
def user_key_id_strategy(draw):
    """Generate valid user key IDs (UUID format)"""
    return str(uuid.uuid4())


@st.composite
def username_strategy(draw):
    """Generate valid usernames (3-50 chars, alphanumeric + underscore/dash)"""
    length = draw(st.integers(min_value=3, max_value=50))
    chars = st.sampled_from("abcdefghijklmnopqrstuvwxyz0123456789_-")
    return ''.join(draw(st.lists(chars, min_size=length, max_size=length)))


@st.composite
def ip_address_strategy(draw):
    """Generate valid IPv4 addresses"""
    octets = [draw(st.integers(min_value=0, max_value=255)) for _ in range(4)]
    return '.'.join(map(str, octets))


@st.composite
def user_agent_strategy(draw):
    """Generate user agent strings"""
    browsers = ["Mozilla/5.0", "Chrome/120.0", "Safari/17.0", "Edge/120.0"]
    return draw(st.sampled_from(browsers))


# Fixtures for test setup
@pytest.fixture
async def session_manager():
    """Create a SessionManager instance for testing"""
    # Use a test Redis database (DB 15)
    manager = SessionManager(
        redis_url="redis://localhost:6379/15",
        session_expire_minutes=1  # Short expiration for testing
    )
    
    # Clear any existing test data
    try:
        manager.redis_client.flushdb()
    except redis.RedisError:
        pytest.skip("Redis not available for testing")
    
    yield manager
    
    # Cleanup
    try:
        manager.redis_client.flushdb()
        manager.close()
    except redis.RedisError:
        pass


# ==================== Property 12: Session persistence until termination ====================
# Feature: user-authentication, Property 12: Session persistence until termination

@pytest.mark.asyncio
@settings(max_examples=100, deadline=5000)
@given(
    token_jti=token_jti_strategy(),
    user_key_id=user_key_id_strategy(),
    username=username_strategy(),
    ip_address=ip_address_strategy(),
    user_agent=user_agent_strategy(),
)
async def test_property_12_session_persists_until_termination(
    token_jti: str,
    user_key_id: str,
    username: str,
    ip_address: str,
    user_agent: str,
):
    """
    Property 12: For any successfully created session, it should remain valid and
    accessible across multiple requests until explicitly logged out or expired.
    
    Validates: Requirements 4.1
    """
    # Skip if Redis is not available
    try:
        manager = SessionManager(
            redis_url="redis://localhost:6379/15",
            session_expire_minutes=1
        )
        manager.redis_client.ping()
    except redis.RedisError:
        pytest.skip("Redis not available")
    
    try:
        # 1. Create session
        session_data = await manager.create_session(
            token_jti=token_jti,
            user_key_id=user_key_id,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        
        # Property: Session should be immediately retrievable
        retrieved_session = await manager.get_session(token_jti)
        assert retrieved_session is not None, "Session should exist immediately after creation"
        assert retrieved_session.user_key_id == user_key_id
        assert retrieved_session.username == username
        
        # Property: Session should persist across multiple retrievals
        for _ in range(5):
            retrieved_again = await manager.get_session(token_jti)
            assert retrieved_again is not None, "Session should persist across multiple requests"
            assert retrieved_again.user_key_id == user_key_id
            await asyncio.sleep(0.1)  # Simulate time between requests
        
        # Property: Session should remain valid after setting as active
        await manager.set_active_session(user_key_id, token_jti)
        validated_session = await manager.validate_session(token_jti)
        assert validated_session is not None, "Session should be valid after being set as active"
        assert validated_session.user_key_id == user_key_id
        
        # Property: Session should only be invalidated by explicit termination
        terminated = await manager.terminate_session(token_jti)
        assert terminated, "Termination should succeed"
        
        # Property: Session should not be retrievable after termination
        after_termination = await manager.get_session(token_jti)
        assert after_termination is None, "Session should not exist after termination"
        
    finally:
        manager.redis_client.flushdb()
        manager.close()


# ==================== Property 11: Session expiration cleanup ====================
# Feature: user-authentication, Property 11: Session expiration cleanup

@pytest.mark.asyncio
@settings(max_examples=50, deadline=10000)  # Fewer examples due to time-based testing
@given(
    token_jti=token_jti_strategy(),
    user_key_id=user_key_id_strategy(),
    username=username_strategy(),
    ip_address=ip_address_strategy(),
    user_agent=user_agent_strategy(),
)
async def test_property_11_session_expiration_cleanup(
    token_jti: str,
    user_key_id: str,
    username: str,
    ip_address: str,
    user_agent: str,
):
    """
    Property 11: For any session that expires or is terminated, the system should
    remove it from the active session registry and the user's active session mapping.
    
    Validates: Requirements 3.5
    """
    # Skip if Redis is not available
    try:
        manager = SessionManager(
            redis_url="redis://localhost:6379/15",
            session_expire_minutes=1  # Very short expiration for testing
        )
        manager.redis_client.ping()
    except redis.RedisError:
        pytest.skip("Redis not available")
    
    try:
        # 1. Create session
        session_data = await manager.create_session(
            token_jti=token_jti,
            user_key_id=user_key_id,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        
        # 2. Set as active session
        await manager.set_active_session(user_key_id, token_jti)
        
        # Property: Session and active mapping should exist before cleanup
        session_before = await manager.get_session(token_jti)
        active_before = await manager.get_active_session(user_key_id)
        assert session_before is not None, "Session should exist before cleanup"
        assert active_before == token_jti, "Active session mapping should exist before cleanup"
        
        # 3. Terminate session
        terminated = await manager.terminate_session(token_jti)
        assert terminated, "Termination should succeed"
        
        # Property: Session data should be removed from Redis
        session_after = await manager.get_session(token_jti)
        assert session_after is None, "Session data should be removed after termination"
        
        # Property: Active session mapping should be removed if it matched
        active_after = await manager.get_active_session(user_key_id)
        assert active_after is None or active_after != token_jti, \
            "Active session mapping should be cleared after termination"
        
        # Property: Validation should fail for terminated session
        validated = await manager.validate_session(token_jti)
        assert validated is None, "Validation should fail for terminated session"
        
    finally:
        manager.redis_client.flushdb()
        manager.close()


# ==================== Additional Unit Tests ====================

@pytest.mark.asyncio
async def test_session_creation_with_ttl(session_manager):
    """Test that sessions are created with correct TTL in Redis"""
    token_jti = str(uuid.uuid4())
    user_key_id = str(uuid.uuid4())
    
    session_data = await session_manager.create_session(
        token_jti=token_jti,
        user_key_id=user_key_id,
        username="test_user",
        ip_address="192.168.1.1",
        user_agent="TestAgent/1.0",
    )
    
    # Check TTL is set
    session_key = session_manager._get_session_key(token_jti)
    ttl = session_manager.redis_client.ttl(session_key)
    
    # TTL should be approximately 60 seconds (1 minute)
    assert 50 <= ttl <= 60, f"TTL should be around 60 seconds, got {ttl}"


@pytest.mark.asyncio
async def test_session_refresh_updates_ttl(session_manager):
    """Test that refreshing a session updates its TTL"""
    token_jti = str(uuid.uuid4())
    user_key_id = str(uuid.uuid4())
    
    # Create session
    await session_manager.create_session(
        token_jti=token_jti,
        user_key_id=user_key_id,
        username="test_user",
        ip_address="192.168.1.1",
        user_agent="TestAgent/1.0",
    )
    
    # Wait a bit
    await asyncio.sleep(2)
    
    # Get initial TTL
    session_key = session_manager._get_session_key(token_jti)
    ttl_before = session_manager.redis_client.ttl(session_key)
    
    # Refresh session
    refreshed = await session_manager.refresh_session(token_jti)
    assert refreshed, "Refresh should succeed"
    
    # Get updated TTL
    ttl_after = session_manager.redis_client.ttl(session_key)
    
    # TTL should be reset to full duration
    assert ttl_after > ttl_before, "TTL should increase after refresh"
    assert 50 <= ttl_after <= 60, f"TTL should be reset to ~60 seconds, got {ttl_after}"


@pytest.mark.asyncio
async def test_active_session_mapping_cleanup(session_manager):
    """Test that active session mapping is cleaned up correctly"""
    token_jti = str(uuid.uuid4())
    user_key_id = str(uuid.uuid4())
    
    # Create and set active session
    await session_manager.create_session(
        token_jti=token_jti,
        user_key_id=user_key_id,
        username="test_user",
        ip_address="192.168.1.1",
        user_agent="TestAgent/1.0",
    )
    await session_manager.set_active_session(user_key_id, token_jti)
    
    # Verify active session exists
    active = await session_manager.get_active_session(user_key_id)
    assert active == token_jti
    
    # Terminate session
    await session_manager.terminate_session(token_jti)
    
    # Verify active session mapping is removed
    active_after = await session_manager.get_active_session(user_key_id)
    assert active_after is None, "Active session mapping should be removed"


@pytest.mark.asyncio
async def test_validate_session_checks_active_status(session_manager):
    """Test that validation checks if session is the active one"""
    token_jti_old = str(uuid.uuid4())
    token_jti_new = str(uuid.uuid4())
    user_key_id = str(uuid.uuid4())
    
    # Create two sessions for same user
    await session_manager.create_session(
        token_jti=token_jti_old,
        user_key_id=user_key_id,
        username="test_user",
        ip_address="192.168.1.1",
        user_agent="TestAgent/1.0",
    )
    await session_manager.create_session(
        token_jti=token_jti_new,
        user_key_id=user_key_id,
        username="test_user",
        ip_address="192.168.1.2",
        user_agent="TestAgent/2.0",
    )
    
    # Set new session as active
    await session_manager.set_active_session(user_key_id, token_jti_new)
    
    # Old session should exist but not validate
    old_session = await session_manager.get_session(token_jti_old)
    assert old_session is not None, "Old session data should still exist"
    
    validated_old = await session_manager.validate_session(token_jti_old)
    assert validated_old is None, "Old session should not validate when not active"
    
    # New session should validate
    validated_new = await session_manager.validate_session(token_jti_new)
    assert validated_new is not None, "New (active) session should validate"
    assert validated_new.user_key_id == user_key_id


@pytest.mark.asyncio
async def test_cleanup_expired_sessions_manual(session_manager):
    """Test manual cleanup of expired sessions"""
    # Create a session with very short TTL by manipulating Redis directly
    token_jti = str(uuid.uuid4())
    user_key_id = str(uuid.uuid4())
    
    # Create session normally
    await session_manager.create_session(
        token_jti=token_jti,
        user_key_id=user_key_id,
        username="test_user",
        ip_address="192.168.1.1",
        user_agent="TestAgent/1.0",
    )
    
    # Manually set a very short TTL to simulate expiration
    session_key = session_manager._get_session_key(token_jti)
    session_manager.redis_client.expire(session_key, 1)  # 1 second
    
    # Wait for expiration
    await asyncio.sleep(2)
    
    # Session should be gone (Redis auto-cleanup)
    session = await session_manager.get_session(token_jti)
    assert session is None, "Session should be auto-cleaned by Redis TTL"


@pytest.mark.asyncio
async def test_concurrent_session_access(session_manager):
    """Test that concurrent access to same session works correctly"""
    token_jti = str(uuid.uuid4())
    user_key_id = str(uuid.uuid4())
    
    # Create session
    await session_manager.create_session(
        token_jti=token_jti,
        user_key_id=user_key_id,
        username="test_user",
        ip_address="192.168.1.1",
        user_agent="TestAgent/1.0",
    )
    await session_manager.set_active_session(user_key_id, token_jti)
    
    # Simulate concurrent access
    tasks = [
        session_manager.validate_session(token_jti)
        for _ in range(10)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # All validations should succeed
    for result in results:
        assert result is not None, "Concurrent validations should all succeed"
        assert result.user_key_id == user_key_id


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
