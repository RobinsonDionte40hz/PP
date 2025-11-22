"""
Integration example showing how SessionManager works with authentication.

This demonstrates the complete flow:
1. User login -> Create JWT with JTI -> Store session in Redis
2. Protected route access -> Validate JWT -> Verify session in Redis
3. User logout -> Invalidate JWT -> Delete session from Redis
"""
import uuid
from datetime import datetime, timedelta, timezone
from app.services.session_manager import SessionManager, get_session_manager
from app.security import create_access_token, decode_token


async def example_login_flow():
    """Example of login flow with session creation"""
    print("=== Example: Login Flow ===\n")
    
    # Get session manager
    session_manager = get_session_manager()
    
    # Simulate successful authentication
    user_key_id = str(uuid.uuid4())
    username = "john_doe"
    ip_address = "192.168.1.100"
    user_agent = "Mozilla/5.0..."
    
    # Generate unique token ID
    token_jti = str(uuid.uuid4())
    
    # Create JWT access token
    token_data = {
        "sub": user_key_id,
        "username": username,
        "jti": token_jti,
    }
    access_token = create_access_token(token_data)
    print(f"1. Generated JWT access token with JTI: {token_jti[:8]}...")
    
    # Check for existing active session and enforce single-session constraint
    await session_manager.enforce_single_session(user_key_id, token_jti)
    print(f"2. Enforced single-session constraint for user: {username}")
    
    # Create session in Redis
    session_data = await session_manager.create_session(
        token_jti=token_jti,
        user_key_id=user_key_id,
        username=username,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    print(f"3. Created session in Redis:")
    print(f"   - User: {session_data.username}")
    print(f"   - Expires: {session_data.expires_at.isoformat()}")
    print(f"   - IP: {session_data.ip_address}")
    
    return access_token, token_jti, user_key_id


async def example_protected_route_access(access_token: str):
    """Example of accessing a protected route"""
    print("\n=== Example: Protected Route Access ===\n")
    
    session_manager = get_session_manager()
    
    # 1. Extract and validate JWT token
    try:
        payload = decode_token(access_token)
        print(f"1. JWT token validated successfully")
        print(f"   - User: {payload['username']}")
        print(f"   - JTI: {payload['jti'][:8]}...")
    except Exception as e:
        print(f"❌ JWT validation failed: {e}")
        return False
    
    # 2. Verify session exists in Redis
    token_jti = payload["jti"]
    session_data = await session_manager.validate_session(token_jti)
    
    if not session_data:
        print(f"❌ Session validation failed - session not found or expired")
        return False
    
    print(f"2. Session validated in Redis:")
    print(f"   - User: {session_data.username}")
    print(f"   - Key ID: {session_data.user_key_id}")
    
    # 3. Grant access to protected resource
    print(f"✅ Access granted to protected resource")
    return True


async def example_logout_flow(token_jti: str):
    """Example of logout flow with session termination"""
    print("\n=== Example: Logout Flow ===\n")
    
    session_manager = get_session_manager()
    
    # Terminate session in Redis
    terminated = await session_manager.terminate_session(token_jti)
    
    if terminated:
        print(f"1. Session terminated successfully")
        print(f"   - JTI: {token_jti[:8]}...")
        print(f"   - Session data deleted from Redis")
        print(f"   - Active session mapping cleared")
    else:
        print(f"❌ Session termination failed - session not found")
        return False
    
    # Verify session is gone
    session_data = await session_manager.get_session(token_jti)
    if session_data is None:
        print(f"2. Verified session is no longer accessible")
        print(f"✅ Logout complete")
        return True
    else:
        print(f"❌ Session still exists after termination")
        return False


async def example_single_session_enforcement():
    """Example of single-session-per-user enforcement"""
    print("\n=== Example: Single Session Enforcement ===\n")
    
    session_manager = get_session_manager()
    user_key_id = str(uuid.uuid4())
    username = "jane_doe"
    
    # First login (e.g., from laptop)
    token_jti_1 = str(uuid.uuid4())
    await session_manager.create_session(
        token_jti=token_jti_1,
        user_key_id=user_key_id,
        username=username,
        ip_address="192.168.1.100",
        user_agent="Mozilla/5.0 (Laptop)",
    )
    await session_manager.set_active_session(user_key_id, token_jti_1)
    print(f"1. First login from laptop:")
    print(f"   - JTI: {token_jti_1[:8]}...")
    print(f"   - IP: 192.168.1.100")
    
    # Verify first session is active
    active_1 = await session_manager.get_active_session(user_key_id)
    if active_1:
        print(f"2. Active session: {active_1[:8]}...")
    else:
        print(f"2. No active session found")
    
    # Second login (e.g., from phone) - should terminate first session
    token_jti_2 = str(uuid.uuid4())
    print(f"\n3. Second login from phone (enforcing single session):")
    print(f"   - New JTI: {token_jti_2[:8]}...")
    
    await session_manager.enforce_single_session(user_key_id, token_jti_2)
    await session_manager.create_session(
        token_jti=token_jti_2,
        user_key_id=user_key_id,
        username=username,
        ip_address="10.0.0.50",
        user_agent="Mobile Safari",
    )
    
    # Verify second session is now active
    active_2 = await session_manager.get_active_session(user_key_id)
    if active_2:
        print(f"4. Active session now: {active_2[:8]}...")
    else:
        print(f"4. No active session found")
    
    # Verify first session is no longer valid
    validated_1 = await session_manager.validate_session(token_jti_1)
    validated_2 = await session_manager.validate_session(token_jti_2)
    
    print(f"\n5. Session validation results:")
    print(f"   - First session (laptop): {'❌ Invalid' if not validated_1 else '✅ Valid'}")
    print(f"   - Second session (phone): {'✅ Valid' if validated_2 else '❌ Invalid'}")
    
    if not validated_1 and validated_2:
        print(f"\n✅ Single session enforcement working correctly")
    else:
        print(f"\n❌ Single session enforcement failed")


async def main():
    """Run all examples"""
    print("=" * 60)
    print("Session Management Integration Examples")
    print("=" * 60)
    
    try:
        # Example 1: Complete login -> access -> logout flow
        access_token, token_jti, user_key_id = await example_login_flow()
        
        # Example 2: Access protected route
        access_granted = await example_protected_route_access(access_token)
        
        # Example 3: Logout
        if access_granted:
            await example_logout_flow(token_jti)
        
        # Example 4: Single session enforcement
        await example_single_session_enforcement()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
